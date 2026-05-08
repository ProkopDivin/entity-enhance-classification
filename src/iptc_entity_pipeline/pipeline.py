"""ClearML pipeline orchestration for entity-enhanced IPTC training."""

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from clearml import Task, TaskTypes
from clearml.automation.controller import PipelineDecorator

from iptc_entity_pipeline.article_embeddings import ArticleEmbeddingProvider
from iptc_entity_pipeline.config import BaseCnf, resolve_paths
from iptc_entity_pipeline.data_loading import attach_entities, load_and_normalize, load_wdid_map, sanitize_name
from iptc_entity_pipeline.evaluation_comparison import build_path, compare_runs
from iptc_entity_pipeline.legacy_reuse import evaluateModel
from iptc_entity_pipeline.model_io import save_outputs
from iptc_entity_pipeline.reporting import log_stage, report_cv_fold, report_cv_curve, report_cv_std, report_eval, report_test_curve

from iptc_entity_pipeline.training import train_model
from iptc_entity_pipeline.reporting import conf_logging

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class EvalResult:
    """Outputs of the final evaluation pipeline step."""

    dev_corpora_df: Any
    test_corpora_df: Any
    test_classes_df: Any
    objective_metrics: dict[str, Any]


@PipelineDecorator.component(
    return_values=['corpora'],
    execution_queue='iptc_entity_tasks',
    task_type=TaskTypes.data_processing,
)
def load_data(
    paths_cnf: Mapping[str, Any],
    emb_cnf: Mapping[str, Any],
    downsample_corpora: Mapping[str, float] | None = None,
):
    """Load normalized corpora with linked entities attached to each document."""
    from iptc_entity_pipeline.config import EmbeddingCnf, PathsCnf, conf_from_dict

    conf_logging()
    paths = conf_from_dict(PathsCnf, paths_cnf)
    emb = conf_from_dict(EmbeddingCnf, emb_cnf)
    corpora = load_and_normalize(
        train_csv=paths.train_csv,
        test_csv=paths.test_csv,
        removed_cat_ids=paths.removed_cat_ids,
        downsample_corpora=downsample_corpora or {},
        downsampling_order_cache_json=paths.downsampling_order_cache_json,
    )
    wdid_mapping = load_wdid_map(wdid_mapping_tsv=paths.wdid_mapping_tsv)
    attach_entities(
        corpus=corpora.train,
        csv_path=paths.train_csv,
        wdid_mapping=wdid_mapping,
        min_relevance=emb.entity_relevance_threshold,
    )
    attach_entities(
        corpus=corpora.test,
        csv_path=paths.test_csv,
        wdid_mapping=wdid_mapping,
        min_relevance=emb.entity_relevance_threshold,
    )
    return corpora


@PipelineDecorator.component(
    return_values=['articleEmbeddingStats'],
    execution_queue='iptc_entity_tasks',
    task_type=TaskTypes.data_processing,
)
def prepare_article_embeddings(
    corpora,
    paths_cnf: Mapping[str, Any],
    emb_cnf: Mapping[str, Any],
):
    """Precompute and cache missing article embeddings for all corpora."""
    from iptc_entity_pipeline.config import EmbeddingCnf, PathsCnf, conf_from_dict

    conf_logging()
    logger = logging.getLogger(__name__)
    paths = conf_from_dict(PathsCnf, paths_cnf)
    emb = conf_from_dict(EmbeddingCnf, emb_cnf)
    if not emb.use_article_embeddings:
        logger.info('Article embeddings disabled by config; skipping article embedding preparation')
        return {
            'train': {},
            'test': {},
            'total_docs': 0,
            'total_computed': 0,
            'total_cached_or_present': 0,
        }

    article_provider = ArticleEmbeddingProvider(
        embeddings_dir=paths.article_embeddings_dir,
        model_name=emb.article_model_name,
        embed_svc_url=emb.embed_svc_url,
        embedding_dim=emb.article_embedding_dim,
    )

    train_stats = article_provider.prepare_embeddings(corpus=corpora.train)
    test_stats = article_provider.prepare_embeddings(corpus=corpora.test)
    total_docs = train_stats.total_docs + test_stats.total_docs
    total_computed = train_stats.computed_docs + test_stats.computed_docs
    logger.info(
        'Article embedding preparation progress: %s/%s articles (computed_missing=%s, cached_or_present=%s)',
        total_docs,
        total_docs,
        total_computed,
        total_docs - total_computed,
    )
    
    result = {
        'train': asdict(train_stats),
        'test': asdict(test_stats),
        'total_docs': total_docs,
        'total_computed': total_computed,
        'total_cached_or_present': total_docs - total_computed,
    }

    return result

@PipelineDecorator.component(
    return_values=['trainData', 'testData', 'featureDim'],
    execution_queue='iptc_entity_tasks',
    task_type=TaskTypes.data_processing,
)
def build_dataset(
    corpora,
    paths_cnf: Mapping[str, Any],
    emb_cnf: Mapping[str, Any],
    article_embedding_stats: Mapping[str, Any],
    assembly_cnf: Mapping[str, Any] | None = None,
):
    """Link embeddings and build train/test embedding datasets.

    When ``assembly_cnf`` is provided and its ``enabled`` flag is true, the
    train corpus's ``catList`` is validated against (or written to) the
    shared category-list JSON at ``assembly_cnf['category_list_path']``.
    The orchestrator augments ``assembly_cnf`` with a ``current_member_label``
    key per call so error messages identify the member.
    """
    from pathlib import Path

    from iptc_entity_pipeline.assembly_io import validate_or_write_cat_list
    from iptc_entity_pipeline.build_dataset import no_entities, report_ent_stats, get_pooling
    from iptc_entity_pipeline.config import EmbeddingCnf, PathsCnf, conf_from_dict
    from iptc_entity_pipeline.dataset_builder import build_emb_data
    from iptc_entity_pipeline.entity_embeddings import EntityEmbeddingStore
    from iptc_entity_pipeline.feature_builder import FeatureBuilder

    conf_logging()
    logger = logging.getLogger(__name__)
    paths = conf_from_dict(PathsCnf, paths_cnf)
    emb = conf_from_dict(EmbeddingCnf, emb_cnf)
    if not emb.use_article_embeddings and not emb.use_entity_embeddings:
        raise ValueError('Invalid embedding config: both use_article_embeddings and use_entity_embeddings are False')

    def _maybe_validate_cat_list(*, train_data) -> None:
        if not assembly_cnf or not assembly_cnf.get('enabled'):
            return
        cat_list_path = assembly_cnf.get('category_list_path')
        if not cat_list_path:
            return
        validate_or_write_cat_list(
            path=Path(cat_list_path),
            cat_list=list(train_data.corpus.catList),
            member_label=str(assembly_cnf.get('current_member_label', 'unknown')),
        )

    logger.info('Initializing providers for entity preparation + linking step')
    logger.info(
        'Using prepared article embedding cache from previous step: total=%s, computed_missing=%s',
        article_embedding_stats.get('total_docs'),
        article_embedding_stats.get('total_computed'),
    )
    article_provider = None
    if emb.use_article_embeddings:
        article_provider = ArticleEmbeddingProvider(
            embeddings_dir=paths.article_embeddings_dir,
            model_name=emb.article_model_name,
            embed_svc_url=emb.embed_svc_url,
            embedding_dim=emb.article_embedding_dim,
        )

    if not emb.use_entity_embeddings:
        assert article_provider is not None
        dataset_bundle = no_entities(
            corpora=corpora,
            article_provider=article_provider,
            logger=logger,
        )
        _maybe_validate_cat_list(train_data=dataset_bundle.train_data)
        return dataset_bundle.train_data, dataset_bundle.test_data, dataset_bundle.feature_dim

    selected_langs = tuple(emb.entity_langs) if emb.entity_langs else (emb.entity_lang,)
    logger.info('Using entity embedding languages=%s', selected_langs)
    entity_store = EntityEmbeddingStore(
        root_dir=paths.entity_embeddings_dir,
        langs=selected_langs,
    )
    pooling = get_pooling(emb_cfg=emb, logger=logger)
    builder = FeatureBuilder(
        article_embedding_provider=article_provider,
        entity_embedding_store=entity_store,
        pooling_strategy=pooling,
        use_article_embeddings=emb.use_article_embeddings,
        combine_method=emb.combine_method,
    )

    task = Task.current_task()
    logger.info('Building linked features for train corpus (%s articles)', len(corpora.train))
    x_train, train_stats = builder.build_features(
        corpus=corpora.train,
        clearml_logger=task.get_logger() if task is not None else None,
        return_stats=True,
    )
    logger.info('Building linked features for test corpus (%s articles)', len(corpora.test))
    x_test, test_stats = builder.build_features(
        corpus=corpora.test,
        clearml_logger=task.get_logger() if task is not None else None,
        return_stats=True,
    )
    report_ent_stats(stats=test_stats, clearml_task=task, logger=logger)
    report_ent_stats(stats=train_stats, clearml_task=task, logger=logger)
    train_data = build_emb_data(corpus=corpora.train, x_matrix=x_train)
    test_data = build_emb_data(corpus=corpora.test, x_matrix=x_test)
    feature_dim = int(x_train.shape[1])

    _maybe_validate_cat_list(train_data=train_data)
    return train_data, test_data, feature_dim


@PipelineDecorator.component(
    return_values=['cvResult'],
    execution_queue='iptc_entity_tasks',
    task_type=TaskTypes.training,
)
def run_cv(
    train_data,
    feature_dim: int,
    hparam_cnf: Mapping[str, Any],
    train_cnf: Mapping[str, Any],
    eval_cnf: Mapping[str, Any],
    cv_cnf: Mapping[str, Any],
    optuna_cnf: Mapping[str, Any],
    tuning_cnf: Mapping[str, Any],
    objective_corpora: str,
    print_logs: bool = True,
    debug: bool = False,
    upload_artifacts: bool = False,
):
    """Run mandatory CV over train and select best hyperparameter combination."""
    from dataclasses import asdict

    from iptc_entity_pipeline.config import (
        CvCnf,
        EvaluationCnf,
        HyperparamSpace,
        OptunaCnf,
        ThresholdTuningCnf,
        TrainingCnf,
        conf_from_dict,
    )
    from iptc_entity_pipeline.cross_validation import build_cv_df, build_cv_result, prepare_cv, select_best

    conf_logging()
    logger = logging.getLogger(__name__)
    task = Task.current_task()
    clearml_logger = task.get_logger()

    space = conf_from_dict(HyperparamSpace, hparam_cnf)
    base_training = conf_from_dict(TrainingCnf, train_cnf)
    eval_cfg = conf_from_dict(EvaluationCnf, eval_cnf)
    cv_cfg = conf_from_dict(CvCnf, cv_cnf)
    optuna_cfg = conf_from_dict(OptunaCnf, optuna_cnf)
    tuning_cfg = conf_from_dict(ThresholdTuningCnf, tuning_cnf)

    task.connect(asdict(space), name='hyperparamSpace')
    task.connect(asdict(base_training), name='trainingConfig')
    task.connect(asdict(eval_cfg), name='evaluationConfig')
    task.connect(asdict(cv_cfg), name='cvConfig')
    task.connect(asdict(optuna_cfg), name='optunaConfig')
    task.connect(asdict(tuning_cfg), name='thresholdTuningConfig')

    x_full, y_full = prepare_cv(train_data=train_data)
    logger.info(f'CV data prepared: x_shape={x_full.shape}, y_shape={y_full.shape}, feature_dim={feature_dim}')

    selection = select_best(
        space=space,
        base_training=base_training,
        train_data=train_data,
        x_full=x_full,
        y_full=y_full,
        feature_dim=feature_dim,
        eval_cfg=eval_cfg,
        cv_cfg=cv_cfg,
        optuna_cfg=optuna_cfg,
        tuning_cfg=tuning_cfg,
        objective_corpora=objective_corpora,
        debug=debug,
        print_logs=print_logs,
        clearml_logger=clearml_logger,
    )

    trials_df, folds_df, cv_dev_df = build_cv_df(
        trial_rows=selection.trial_rows,
        fold_rows=selection.fold_rows,
        best_trial=selection.best_trial,
        objective_corpora=objective_corpora,
    )

    report_cv_curve(
        task=task,
        logger=clearml_logger,
        trials_df=trials_df,
        folds_df=folds_df,
        cv_dev_df=cv_dev_df,
        upload_artifacts=upload_artifacts,
    )
    report_cv_fold(logger=clearml_logger, fold_curves=selection.best_fold_curves)
    logger.info(f'CV complete: best_trial F1={selection.best_trial["F1_mean"]:.4f}')

    cv_result = build_cv_result(
        cv_dev_df=cv_dev_df,
        selection=selection,
        patience=base_training.early_stopping_patience,
        tuning_cfg=tuning_cfg,
        cat_list=list(train_data.corpus.catList),
        default_threshold=eval_cfg.threshold_eval,
    )

    if cv_result.threshold_report_df is not None:
        clearml_logger.report_table(
            title='Threshold Tuning',
            series='Per-class thresholds (CV folds)',
            iteration=0,
            table_plot=cv_result.threshold_report_df,
        )
        n_classes = int(cv_result.threshold_report_df['n_folds'].gt(0).sum())
        logger.info(
            f'Threshold tuning: aggregation={tuning_cfg.aggregation}, '
            f'classes_with_tuned_threshold={n_classes}/{len(cv_result.threshold_report_df)}'
        )
        if upload_artifacts:
            task.upload_artifact(
                'threshold_tuning_report', artifact_object=cv_result.threshold_report_df,
            )
            task.upload_artifact(
                'threshold_tuning_thresholds', artifact_object=dict(cv_result.tuned_thresholds or {}),
            )

    return cv_result

@PipelineDecorator.component(
    return_values=['assemblyResult'],
    execution_queue='iptc_entity_tasks',
    task_type=TaskTypes.training,
)
def run_assembly_step(
    member_train_data,
    member_feature_dims,
    member_model_cnfs,
    member_train_cnfs,
    member_thresholds,
    member_labels,
    eval_cnf: Mapping[str, Any],
    cv_cnf: Mapping[str, Any],
    objective_corpora: str,
    print_logs: bool = True,
    debug: bool = False,
    upload_artifacts: bool = False,
    mapping_artifact_name: str = 'assembly_class_to_model',
):
    """Run the assembly CV: trains both members per fold, picks per-class winners.

    Returns an :class:`assembly.AssemblyCvResult` with the same downstream
    surface as ``run_cv``'s ``CvResult``.
    """
    from dataclasses import asdict

    from iptc_entity_pipeline.assembly import run_assembly
    from iptc_entity_pipeline.config import (
        CvCnf,
        EvaluationCnf,
        ModelCnf,
        TrainingCnf,
        conf_from_dict,
    )

    conf_logging()
    logger = logging.getLogger(__name__)
    task = Task.current_task()
    clearml_logger = task.get_logger()

    eval_cfg = conf_from_dict(EvaluationCnf, eval_cnf)
    cv_cfg = conf_from_dict(CvCnf, cv_cnf)
    model_cfgs = [conf_from_dict(ModelCnf, m) for m in member_model_cnfs]
    train_cfgs = [conf_from_dict(TrainingCnf, t) for t in member_train_cnfs]

    task.connect(asdict(eval_cfg), name='evaluationConfig')
    task.connect(asdict(cv_cfg), name='cvConfig')
    for idx, label in enumerate(member_labels):
        task.connect(member_model_cnfs[idx], name=f'modelConfig_{label}')
        task.connect(member_train_cnfs[idx], name=f'trainingConfig_{label}')

    logger.info(
        f'Assembly step: members={list(member_labels)} '
        f'folds={cv_cfg.folds} debug={debug}'
    )

    assembly_result = run_assembly(
        member_train_data=member_train_data,
        member_feature_dims=member_feature_dims,
        member_model_cfgs=model_cfgs,
        member_train_cfgs=train_cfgs,
        member_thresholds=member_thresholds,
        member_labels=member_labels,
        eval_cfg=eval_cfg,
        cv_cfg=cv_cfg,
        objective_corpora=objective_corpora,
        debug=debug,
        print_logs=print_logs,
        clearml_logger=clearml_logger,
    )

    mapping_payload = {
        'schema_version': '1',
        'member_labels': list(assembly_result.class_to_model.member_labels),
        'primary_index': 0,
        'assignments': dict(assembly_result.class_to_model.assignments),
        'stitched_thresholds': dict(assembly_result.tuned_thresholds),
    }
    task.upload_artifact(mapping_artifact_name, artifact_object=mapping_payload)
    if upload_artifacts:
        task.upload_artifact(
            'assembly_per_class_f1', artifact_object=assembly_result.per_class_f1_df,
        )
        task.upload_artifact(
            'assembly_per_fold_f1', artifact_object=assembly_result.per_fold_f1_df,
        )
        task.upload_artifact(
            'assembly_threshold_report', artifact_object=assembly_result.threshold_report_df,
        )

    return assembly_result


@PipelineDecorator.component(
    return_values=['trainedModel'],
    execution_queue='iptc_entity_tasks',
    task_type=TaskTypes.training,
)
def train_best(
    train_data,
    test_data,
    feature_dim: int,
    best_model_cnf: Mapping[str, Any],
    train_cnf: Mapping[str, Any],
    print_logs: bool = True,
):
    """Train final model on full train set with best hyperparams from CV."""
    from iptc_entity_pipeline.config import ModelCnf, TrainingCnf, conf_from_dict

    conf_logging()
    logger = logging.getLogger(__name__)
    task = Task.current_task()
    clearml_logger = task.get_logger()

    model_cfg = conf_from_dict(ModelCnf, best_model_cnf)
    train_cfg = conf_from_dict(TrainingCnf, train_cnf)

    task.connect(best_model_cnf, name='bestModelConfig')
    task.connect(train_cnf, name='bestTrainingConfig')

    logger.info(
        f'Training final model: hidden_dim={model_cfg.hidden_dim}, '
        f'dropouts=({model_cfg.dropouts1}, {model_cfg.dropouts2}), '
        f'lr={train_cfg.learning_rate}, batch_size={train_cfg.batch_size}, '
        f'epochs={train_cfg.epochs}, early_stopping_patience={train_cfg.early_stopping_patience}, '
        f'feature_dim={feature_dim}, train_docs={len(train_data.corpus)}, test_docs={len(test_data.corpus)}'
    )
    if train_cfg.early_stopping_patience != 0:
        raise ValueError('Final retraining may evaluate test curves only when early stopping is disabled')

    result = train_model(
        train_data=train_data,
        dev_data=test_data,
        feature_dim=feature_dim,
        model_config=model_cfg,
        training_config=train_cfg,
        print_logs=print_logs,
        validation_split_name='test',
    )

    report_test_curve(logger=clearml_logger, result=result)
    logger.info(
        f'Final model training complete: epochs={result.epochs_run}, '
        f'final_dev_loss={result.final_dev_loss:.6f}'
    )
    return result.model

@PipelineDecorator.component(
    return_values=['evalResult'],
    execution_queue='iptc_entity_tasks',
    task_type=TaskTypes.testing,
)
def eval_final(
    trained_model,
    cv_dev_df,
    test_data,
    eval_cnf: Mapping[str, Any],
    emb_cnf: Mapping[str, Any],
    objective_corpora: str,
    config_name: str,
    config_mapping: Mapping[str, Any],
    feature_dim: int,
    tuned_thresholds: Mapping[str, float] | None = None,
    threshold_report_df: Any = None,
    upload_artifacts: bool = False,
):
    """Evaluate final model on test and persist CV dev summary with mean/std."""
    import logging
    from dataclasses import asdict
    from pathlib import Path

    from iptc_entity_pipeline.config import EmbeddingCnf, EvaluationCnf, conf_from_dict
    from iptc_entity_pipeline.data_loading import sanitize_name
    from iptc_entity_pipeline.evaluation_comparison import build_path, compare_runs
    from iptc_entity_pipeline.legacy_reuse import evaluateModel
    from iptc_entity_pipeline.model_io import export_eval_excel, save_outputs
    from iptc_entity_pipeline.pipeline import EvalResult
    from iptc_entity_pipeline.reporting import conf_logging, report_eval_scalars, report_eval_tables

    def run_comparison():
        """Run optional baseline comparison and report results to ClearML."""
        comparison_cfg = config_mapping.get('evaluation_comparison') or config_mapping.get('comparison') or {}
        base_run_dir = str(comparison_cfg.get('base_run_dir', '') or eval_cfg.base_run_dir).strip()
        if not base_run_dir:
            return None
        out_dir = Path(save_paths.output_dir)
        result = compare_runs(
            current_run_dir=save_paths.output_dir,
            base_run_dir=base_run_dir,
            threshold_eval=eval_cfg.threshold_eval,
            averaging_type=eval_cfg.averaging_type,
            top_n=int(comparison_cfg.get('top_n', 20)),
            only_diff=bool(comparison_cfg.get('only_diff', False)),
            output_path=build_path(
                output_root=str(comparison_cfg.get('output_root', out_dir)),
                config_name=str(comparison_cfg.get('config_name', config_name)),
            ),
        )
        clearml_logger.report_table(
            title='Evaluation Comparison', series='Summary', iteration=0, table_plot=result.summary_comparison,
        )
        clearml_logger.report_table(
            title='Evaluation Comparison', series='Corpora Comparison', iteration=0,
            table_plot=result.corpora_comparison,
        )
        clearml_logger.report_table(
            title='Evaluation Comparison', series='Classes Comparison', iteration=0,
            table_plot=result.classes_comparison,
        )
        if upload_artifacts:
            task.upload_artifact('evaluation_comparison_summary', artifact_object=result.summary_comparison)
            task.upload_artifact('evaluation_comparison_classes', artifact_object=result.classes_comparison)
            task.upload_artifact('evaluation_comparison_corpora', artifact_object=result.corpora_comparison)
            if result.excel_path is not None:
                task.upload_artifact('evaluation_comparison_xlsx', artifact_object=str(result.excel_path))
        return result

    def upload_eval_artifacts():
        """Upload all eval-step artifacts to ClearML."""
        objective_row_name = f'All-{eval_cfg.averaging_type}'
        task.upload_artifact('dev_corpora_dataframe', artifact_object=cv_dev_df)
        task.upload_artifact('test_corpora_dataframe', artifact_object=df_corpora_test)
        task.upload_artifact('test_classes_dataframe', artifact_object=df_classes_test)
        task.upload_artifact('final_evaluation_tables_xlsx', artifact_object=str(excel_path))
        task.upload_artifact('saved_model_paths', artifact_object=asdict(save_paths))
        task.upload_artifact(
            'evaluation_thresholds',
            artifact_object={
                'threshold_predict': eval_cfg.threshold_predict,
                'threshold_eval': eval_cfg.threshold_eval,
                'objective_corpora': objective_corpora,
                'objective_row_name': objective_row_name,
            },
        )
        # Compatibility aliases kept to match naming from baseline runs.
        task.upload_artifact('Corpora Dataframe', artifact_object=df_corpora_test)
        task.upload_artifact('Classes Dataframe', artifact_object=df_classes_test)

    conf_logging()
    logger = logging.getLogger(__name__)
    task = Task.current_task()
    clearml_logger = task.get_logger()

    eval_cfg = conf_from_dict(EvaluationCnf, eval_cnf)
    emb_cfg = conf_from_dict(EmbeddingCnf, emb_cnf)
    task.connect(eval_cnf, name='evaluationConfig')
    task.connect(emb_cnf, name='embeddingConfig')

    custom_thresholds = dict(tuned_thresholds) if tuned_thresholds else None
    logger.info(
        f'Evaluating final model on test: test_docs={len(test_data.corpus)}, '
        f'objective={objective_corpora}, custom_thresholds='
        f'{len(custom_thresholds) if custom_thresholds else 0} class(es)'
    )

    df_corpora_test, df_classes_test, pred_scores = evaluateModel(
        model=trained_model,
        evalData=test_data,
        evaluation_config=eval_cfg,
        customThresholds=custom_thresholds,
        returnPredictions=True,
    )

    report_eval_scalars(
        clearml_logger=clearml_logger,
        cv_dev_df=cv_dev_df,
        df_corpora_test=df_corpora_test,
        objective_corpora=objective_corpora,
        averaging_type=eval_cfg.averaging_type,
    )
    report_eval_tables(
        clearml_logger=clearml_logger,
        cv_dev_df=cv_dev_df,
        df_corpora_test=df_corpora_test,
        df_classes_test=df_classes_test,
    )

    save_paths = save_outputs(
        model=trained_model,
        test_data=test_data,
        pred_scores=pred_scores,
        eval_cnf=eval_cfg,
        emb_cnf=emb_cfg,
        config_mapping=config_mapping,
        config_name=config_name,
        feature_dim=feature_dim,
        tuned_thresholds=custom_thresholds,
        threshold_report_df=threshold_report_df,
        upload_artifacts=upload_artifacts,
    )

    if threshold_report_df is not None:
        clearml_logger.report_table(
            title='Threshold Tuning',
            series='Per-class thresholds (final eval)',
            iteration=0,
            table_plot=threshold_report_df,
        )

    comparison_result = run_comparison()

    output_dir = Path(save_paths.output_dir)
    model_name = sanitize_name(value=config_name)
    excel_path = output_dir / f'final_evaluation_tables_{model_name}.xlsx'
    export_eval_excel(
        excel_path=excel_path,
        cv_dev_df=cv_dev_df,
        df_corpora_test=df_corpora_test,
        df_classes_test=df_classes_test,
        comparison_result=comparison_result,
    )

    if upload_artifacts:
        upload_eval_artifacts()

    objective_row_name = f'All-{eval_cfg.averaging_type}'
    objective_metrics = (
        df_corpora_test.loc[objective_corpora].to_dict()
        if objective_corpora in df_corpora_test.index
        else df_corpora_test.loc[objective_row_name].to_dict()
    )
    logger.info(f'Evaluation complete: F1={objective_metrics.get("F1", "N/A")}, config={config_name}')
    return EvalResult(
        dev_corpora_df=cv_dev_df,
        test_corpora_df=df_corpora_test,
        test_classes_df=df_classes_test,
        objective_metrics=objective_metrics,
    )


def _run_assembly_training_pipeline(
    *,
    cnf: Mapping[str, Any],
    assembly_cnf: Mapping[str, Any],
    paths_cnf: Mapping[str, Any],
    eval_cnf: Mapping[str, Any],
    cv_cnf: Mapping[str, Any],
    objective_corpora: str,
    downsample_corpora: Mapping[str, float],
    config_name: str,
    print_logs: bool,
    debug: bool,
    upload_artifacts: bool,
) -> None:
    """Run the dual-model assembly variant of the training pipeline.

    Steps:
      1) For each member: load_data, prepare_article_embeddings, build_dataset
         (cat-list validation runs inside ``build_dataset`` for the assembly).
      2) Load each member's per-class thresholds.
      3) Run the assembly CV step (replaces ``run_cv``).
      4) Train each member on full train via ``train_best`` (early stopping
         disabled at the call site).
      5) Wrap the trained members in :class:`AssemblyModel` and run
         ``eval_final`` on it (no changes needed there).
    """
    from dataclasses import asdict, replace as dc_replace
    from pathlib import Path

    from iptc_entity_pipeline.assembly_io import load_thresholds
    from iptc_entity_pipeline.assembly_model import AssemblyModel
    from iptc_entity_pipeline.config import AssemblyMemberCnf, conf_from_dict, get_config

    task = Task.current_task()
    members_raw = list(assembly_cnf.get('members') or ())
    if len(members_raw) < 2:
        raise ValueError('assembly.members must contain at least two members')

    members: list[AssemblyMemberCnf] = [conf_from_dict(AssemblyMemberCnf, m) for m in members_raw]
    member_full_cnfs = [get_config(m.config_name) for m in members]
    member_labels = [m.label or m.config_name for m in members]

    log_stage(
        task=task,
        message=f'Assembly mode: {len(members)} members={member_labels}',
        print_logs=print_logs,
    )

    member_train_data: list[Any] = []
    member_test_data: list[Any] = []
    member_feature_dims: list[int] = []
    for idx, member in enumerate(members):
        m_cnf = member_full_cnfs[idx]
        m_emb = asdict(m_cnf.emb)
        log_stage(
            task=task,
            message=f'Assembly member {idx + 1}/{len(members)} ({member.label}): data prep',
            print_logs=print_logs,
        )
        m_corpora = load_data(
            paths_cnf=paths_cnf,
            emb_cnf=m_emb,
            downsample_corpora=downsample_corpora,
        )
        m_article_stats = prepare_article_embeddings(
            corpora=m_corpora,
            paths_cnf=paths_cnf,
            emb_cnf=m_emb,
        )
        if upload_artifacts:
            task.upload_artifact(
                f'article_embedding_stats_{member.label}',
                artifact_object=dict(m_article_stats),
            )
        m_assembly_cnf = {**dict(assembly_cnf), 'current_member_label': member.label}
        m_train_data, m_test_data, m_feature_dim = build_dataset(
            corpora=m_corpora,
            paths_cnf=paths_cnf,
            emb_cnf=m_emb,
            article_embedding_stats=m_article_stats,
            assembly_cnf=m_assembly_cnf,
        )
        member_train_data.append(m_train_data)
        member_test_data.append(m_test_data)
        member_feature_dims.append(m_feature_dim)

    cat_list = list(member_train_data[0].corpus.catList)
    member_thresholds = [
        load_thresholds(
            path=Path(member.thresholds_path),
            cat_list=cat_list,
            default_threshold=float(eval_cnf.get('threshold_eval', 0.5)),
        )
        for member in members
    ]

    log_stage(
        task=task,
        message=f'Assembly CV: {cv_cnf.get("folds", 5)}-fold per-class selection',
        print_logs=print_logs,
    )
    assembly_result = run_assembly_step(
        member_train_data=member_train_data,
        member_feature_dims=member_feature_dims,
        member_model_cnfs=[asdict(c.model) for c in member_full_cnfs],
        member_train_cnfs=[asdict(c.train) for c in member_full_cnfs],
        member_thresholds=member_thresholds,
        member_labels=member_labels,
        eval_cnf=eval_cnf,
        cv_cnf=cv_cnf,
        objective_corpora=objective_corpora,
        print_logs=print_logs,
        debug=debug,
        upload_artifacts=upload_artifacts,
        mapping_artifact_name=str(assembly_cnf.get('mapping_artifact_name', 'assembly_class_to_model')),
    )

    log_stage(
        task=task,
        message='Assembly final training: training each member on full train',
        print_logs=print_logs,
    )
    trained_members: list[Any] = []
    shared_test_data = member_test_data[0]
    for idx, m_cnf in enumerate(member_full_cnfs):
        member_train_cfg = dc_replace(m_cnf.train, early_stopping_patience=0)
        trained = train_best(
            train_data=member_train_data[idx],
            test_data=shared_test_data,
            feature_dim=member_feature_dims[idx],
            best_model_cnf=asdict(m_cnf.model),
            train_cnf=asdict(member_train_cfg),
            print_logs=print_logs,
        )
        trained_members.append(trained)

    assembled_model = AssemblyModel(
        members=trained_members,
        cat_list=cat_list,
        class_to_model=assembly_result.class_to_model,
    )

    log_stage(
        task=task,
        message='Assembly: evaluating ensembled model on test',
        print_logs=print_logs,
    )
    eval_result = eval_final(
        trained_model=assembled_model,
        cv_dev_df=assembly_result.cv_dev_df,
        test_data=shared_test_data,
        eval_cnf=eval_cnf,
        emb_cnf=asdict(member_full_cnfs[0].emb),
        objective_corpora=objective_corpora,
        config_name=config_name,
        config_mapping=cnf,
        feature_dim=member_feature_dims[0],
        tuned_thresholds=assembly_result.tuned_thresholds,
        threshold_report_df=assembly_result.threshold_report_df,
        upload_artifacts=upload_artifacts,
    )

    logger = task.get_logger()
    objective_row_name = f'All-{eval_cnf["averaging_type"]}'
    if objective_row_name in eval_result.dev_corpora_df.index:
        row_cv = eval_result.dev_corpora_df.loc[objective_row_name].to_dict()
    else:
        row_cv = eval_result.dev_corpora_df.iloc[0].to_dict()
    report_eval(logger=logger, title='Cross Validation Results', row=row_cv, iteration=0)
    if 'Precision_std' in row_cv:
        report_cv_std(logger=logger, row=row_cv, title='Cross Validation Results', iteration=0)
    report_eval(
        logger=logger,
        title='Test Evaluation Results',
        row=eval_result.objective_metrics,
        iteration=0,
    )
    if 'All-micro' in eval_result.test_corpora_df.index:
        report_eval(
            logger=logger,
            title='Test Evaluation Results (micro)',
            row=eval_result.test_corpora_df.loc['All-micro'].to_dict(),
            iteration=0,
        )

    if upload_artifacts:
        task.upload_artifact('pipeline_config', artifact_object=dict(cnf))
        task.upload_artifact('objective_metrics', artifact_object=dict(eval_result.objective_metrics))
        task.upload_artifact('dev_corpora_dataframe', artifact_object=eval_result.dev_corpora_df)
        task.upload_artifact('test_corpora_dataframe', artifact_object=eval_result.test_corpora_df)
        task.upload_artifact('test_classes_dataframe', artifact_object=eval_result.test_classes_df)

    log_stage(
        task=task,
        message='Assembly pipeline finished',
        print_logs=print_logs,
    )


@PipelineDecorator.pipeline(
    name='iptc-entity-enhanced-v1',
    project='iptc/EntityEnhanced',
    version='0.1',
    pipeline_execution_queue='iptc_entity_pipeline',
)
def run_training_pipeline(cnf: Mapping[str, Any]) -> None:
    """Execute full v1 training and evaluation pipeline."""
    conf_logging()
    task = Task.current_task()
    task.connect(cnf, name='pipelineConfig')
    config_name = str(cnf.get('config_name', 'wpentities'))
    task.add_tags([config_name])

    paths_cnf = cnf['paths']
    emb_cnf = cnf['emb']
    train_cnf = cnf['train']
    eval_cnf = cnf['eval']
    cv_cnf = cnf.get('cv', {'folds': 5, 'random_seed': 43})
    optuna_cnf = cnf.get('optuna', {})
    tuning_cnf = cnf.get('tuning', {'enabled': False})
    assembly_cnf = cnf.get('assembly') or {'enabled': False}
    hparam_cnf = cnf['hparam']
    obj_corpora = cnf['objective_corpora']
    down_smpl = cnf.get('downsample_corpora', {})
    use_art_emb = bool(emb_cnf.get('use_article_embeddings', True))
    use_ent_emb = bool(emb_cnf.get('use_entity_embeddings', True))
    print_logs = bool(cnf.get('print_logs', True))
    debug = bool(cnf.get('debug', False))
    upload_artifacts = bool(cnf.get('upload_artifacts', False))

    if assembly_cnf.get('enabled'):
        _run_assembly_training_pipeline(
            cnf=cnf,
            assembly_cnf=assembly_cnf,
            paths_cnf=paths_cnf,
            eval_cnf=eval_cnf,
            cv_cnf=cv_cnf,
            objective_corpora=obj_corpora,
            downsample_corpora=down_smpl,
            config_name=config_name,
            print_logs=print_logs,
            debug=debug,
            upload_artifacts=upload_artifacts,
        )
        return

    if not use_art_emb and not use_ent_emb:
        raise ValueError('Invalid embedding config: both use_article_embeddings and use_entity_embeddings are False')

    log_stage(
        task=task,
        message='Stage 1/6: Loading corpora and article-to-entity mapping',
        print_logs=print_logs,
    )
    corpora = load_data(
        paths_cnf=paths_cnf,
        emb_cnf=emb_cnf,
        downsample_corpora=down_smpl,
    )
    log_stage(
        task=task,
        message=(
            'Stage 2/6: Preparing article embeddings (train/test)'
            if use_art_emb
            else 'Stage 2/6: Skipping article embeddings (article embeddings disabled)'
        ),
        print_logs=print_logs,
    )
    article_embedding_stats = prepare_article_embeddings(
        corpora=corpora,
        paths_cnf=paths_cnf,
        emb_cnf=emb_cnf,
    )
    if upload_artifacts:
        task.upload_artifact('article_embedding_stats', artifact_object=dict(article_embedding_stats))

    log_stage(
        task=task,
        message=(
            'Stage 3/6: Preparing entity embeddings, linking, and building datasets'
            if use_art_emb and use_ent_emb
            else (
                'Stage 3/6: Building entity-only datasets (article embeddings disabled)'
                if use_ent_emb
                else 'Stage 3/6: Building article-only datasets (entity embeddings disabled)'
            )
        ),
        print_logs=print_logs,
    )
    train_data, test_data, feature_dim = build_dataset(
        corpora=corpora,
        paths_cnf=paths_cnf,
        emb_cnf=emb_cnf,
        article_embedding_stats=article_embedding_stats,
    )
    log_stage(
        task=task,
        message=f'Stage 4/6: Running mandatory {cv_cnf.get("folds", 5)}-fold cross-validation on train',
        print_logs=print_logs,
    )
    cv_result = run_cv(
        train_data=train_data,
        feature_dim=feature_dim,
        hparam_cnf=hparam_cnf,
        train_cnf=train_cnf,
        eval_cnf=eval_cnf,
        cv_cnf=cv_cnf,
        optuna_cnf=optuna_cnf,
        tuning_cnf=tuning_cnf,
        objective_corpora=obj_corpora,
        print_logs=print_logs,
        debug=debug,
        upload_artifacts=upload_artifacts,
    )
    if upload_artifacts:
        task.upload_artifact('cv_objective_metrics', artifact_object=dict(cv_result.objective_metrics))

    log_stage(
        task=task,
        message='Stage 5/6: Training final model on full train with fixed CV-derived epochs',
        print_logs=print_logs,
    )
    trained_model = train_best(
        train_data=train_data,
        test_data=test_data,
        feature_dim=feature_dim,
        best_model_cnf=cv_result.best_model_config,
        train_cnf=cv_result.best_training_config,
        print_logs=print_logs,
    )

    log_stage(
        task=task,
        message='Stage 6/6: Evaluating final model on test',
        print_logs=print_logs,
    )
    eval_result = eval_final(
        trained_model=trained_model,
        cv_dev_df=cv_result.cv_dev_df,
        test_data=test_data,
        eval_cnf=eval_cnf,
        emb_cnf=emb_cnf,
        objective_corpora=obj_corpora,
        config_name=config_name,
        config_mapping=cnf,
        feature_dim=feature_dim,
        tuned_thresholds=cv_result.tuned_thresholds,
        threshold_report_df=cv_result.threshold_report_df,
        upload_artifacts=upload_artifacts,
    )

    logger = task.get_logger()
    objective_row_name = f'All-{eval_cnf["averaging_type"]}'
    if objective_row_name in eval_result.dev_corpora_df.index:
        row_cv = eval_result.dev_corpora_df.loc[objective_row_name].to_dict()
    else:
        row_cv = eval_result.dev_corpora_df.iloc[0].to_dict()

    report_eval(
        logger=logger,
        title='Cross Validation Results',
        row=row_cv,
        iteration=0,
    )
    if 'Precision_std' in row_cv:
        report_cv_std(
            logger=logger,
            row=row_cv,
            title='Cross Validation Results',
            iteration=0,
        )

    report_eval(
        logger=logger,
        title='Test Evaluation Results',
        row=eval_result.objective_metrics,
        iteration=0,
    )
    if 'All-micro' in eval_result.test_corpora_df.index:
        report_eval(
            logger=logger,
            title='Test Evaluation Results (micro)',
            row=eval_result.test_corpora_df.loc['All-micro'].to_dict(),
            iteration=0,
        )

    if upload_artifacts:
        task.upload_artifact('pipeline_config', artifact_object=dict(cnf))
        task.upload_artifact('objective_metrics', artifact_object=dict(eval_result.objective_metrics))
        task.upload_artifact('dev_corpora_dataframe', artifact_object=eval_result.dev_corpora_df)
        task.upload_artifact('test_corpora_dataframe', artifact_object=eval_result.test_corpora_df)
        task.upload_artifact('test_classes_dataframe', artifact_object=eval_result.test_classes_df)
        finish_message = 'Pipeline finished: metrics, tables, and artifacts uploaded'
    else:
        finish_message = 'Pipeline finished: metrics and tables reported (artifact uploads disabled)'
    log_stage(
        task=task,
        message=finish_message,
        print_logs=print_logs,
    )


def run_pipeline(
    config: Optional[BaseCnf] = None,
    config_name: str = 'wpentities',
    is_local: bool = False,
) -> Tuple[BaseCnf, Mapping[str, Any]]:
    """
    Resolve config and execute the ClearML pipeline locally.

    :param config: Optional custom pipeline config.
    :param config_name: Name of selected config variant.
    :param is_local: Run pipeline steps locally instead of dispatching to a queue.
    :return: Tuple of resolved config and mapping passed to ClearML.
    """
    pipeline_config = config or BaseCnf()
    resolved_config = resolve_paths(config=pipeline_config, root_dir=Path.cwd())
    config_mapping = asdict(resolved_config)
    config_mapping['config_name'] = config_name
    if is_local:
        PipelineDecorator.run_locally()
    run_training_pipeline(cnf=config_mapping)
    return resolved_config, config_mapping
