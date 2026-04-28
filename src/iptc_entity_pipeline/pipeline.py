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
):
    """Link embeddings and build train/test embedding datasets."""
    from iptc_entity_pipeline.build_dataset import no_entities, report_ent_stats, get_pooling
    from iptc_entity_pipeline.config import EmbeddingCnf, PathsCnf, conf_from_dict
    from iptc_entity_pipeline.dataset_builder import build_emb_data
    from iptc_entity_pipeline.entity_embeddings import EntityEmbeddingStore
    from iptc_entity_pipeline.feature_builder import FeatureBuilder

    conf_logging()
    logger = logging.getLogger(__name__)
    logger.info('Initializing providers for merged entity preparation + linking step')
    logger.info(
        'Using prepared article embedding cache from previous step: total=%s, computed_missing=%s',
        article_embedding_stats.get('total_docs'),
        article_embedding_stats.get('total_computed'),
    )
    paths = conf_from_dict(PathsCnf, paths_cnf)
    emb = conf_from_dict(EmbeddingCnf, emb_cnf)
    article_provider = ArticleEmbeddingProvider(
        embeddings_dir=paths.article_embeddings_dir,
        model_name=emb.article_model_name,
        embed_svc_url=emb.embed_svc_url,
        embedding_dim=emb.article_embedding_dim,
    )
    if not emb.use_entity_embeddings:
        dataset_bundle = no_entities(
            corpora=corpora,
            article_provider=article_provider,
            logger=logger,
        )
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
    objective_corpora: str,
    print_logs: bool = True,
    debug: bool = False,
    upload_artifacts: bool = False,
):
    """Run mandatory CV over train and select best hyperparameter combination."""
    from dataclasses import asdict

    from iptc_entity_pipeline.config import CvCnf, EvaluationCnf, HyperparamSpace, OptunaCnf, TrainingCnf, conf_from_dict
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

    task.connect(asdict(space), name='hyperparamSpace')
    task.connect(asdict(base_training), name='trainingConfig')
    task.connect(asdict(eval_cfg), name='evaluationConfig')
    task.connect(asdict(cv_cfg), name='cvConfig')
    task.connect(asdict(optuna_cfg), name='optunaConfig')

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

    return build_cv_result(cv_dev_df=cv_dev_df, selection=selection)

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
    model_cfg = conf_from_dict(ModelCnf, best_model_cnf)
    train_cfg = conf_from_dict(TrainingCnf, train_cnf)
    result = train_model(
        train_data=train_data,
        dev_data=test_data,
        feature_dim=feature_dim,
        model_config=model_cfg,
        training_config=train_cfg,
        print_logs=print_logs,
    )
    task = Task.current_task()
    logger = task.get_logger()
    report_test_curve(logger=logger, result=result)
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
    upload_artifacts: bool = False,
):
    """Evaluate final model on test and persist CV dev summary with mean/std."""
    from pathlib import Path

    import pandas as pd
    from iptc_entity_pipeline.pipeline import EvalResult
    from iptc_entity_pipeline.config import EmbeddingCnf, EvaluationCnf, conf_from_dict

    conf_logging()
    task = Task.current_task()
    logger = task.get_logger()
    eval_cfg = conf_from_dict(EvaluationCnf, eval_cnf)
    emb_cfg = conf_from_dict(EmbeddingCnf, emb_cnf)
    df_corpora_test, df_classes_test, pred_scores = evaluateModel(
        model=trained_model,
        evalData=test_data,
        evaluation_config=eval_cfg,
        customThresholds=None,
        returnPredictions=True,
    )

    objective_row_name = f'All-{eval_cfg.averaging_type}'
    if objective_corpora in cv_dev_df.index:
        row_cv = cv_dev_df.loc[objective_corpora].to_dict()
    else:
        row_cv = cv_dev_df.iloc[0].to_dict()
    report_eval(logger=logger, title='Dev Cross Validation Mean Results', row=row_cv, iteration=0)
    if 'Precision_std' in row_cv:
        report_cv_std(
            logger=logger,
            row=row_cv,
            title='Dev Cross Validation Mean Results',
            iteration=0,
        )
    row_all_test = df_corpora_test.loc[objective_row_name].to_dict()
    report_eval(logger=logger, title='Test Evaluation Results', row=row_all_test, iteration=0)
    if 'All-micro' in df_corpora_test.index:
        row_micro_test = df_corpora_test.loc['All-micro'].to_dict()
        report_eval(logger=logger, title='Test Evaluation Results (micro)', row=row_micro_test, iteration=0)
    if objective_corpora in df_corpora_test.index:
        row_objective_test = df_corpora_test.loc[objective_corpora].to_dict()
        report_eval(logger=logger, title='Objective Test Evaluation Results', row=row_objective_test, iteration=0)
    else:
        logging.getLogger(__name__).warning(
            'Requested objective_corpora "%s" not found in test corpus index; available=%s',
            objective_corpora,
            list(df_corpora_test.index),
        )

    logger.report_table(title='Cross Validation Summary', series='Mean+Std', iteration=0, table_plot=cv_dev_df)
    logger.report_table(title='Test Evaluation', series='Corpora Dataframe', iteration=0, table_plot=df_corpora_test)
    logger.report_table(title='Test Evaluation', series='Classes Dataframe', iteration=0, table_plot=df_classes_test)

    model_name = sanitize_name(value=config_name)
    save_paths = save_outputs(
        model=trained_model,
        test_data=test_data,
        pred_scores=pred_scores,
        eval_cnf=eval_cfg,
        emb_cnf=emb_cfg,
        config_mapping=config_mapping,
        config_name=config_name,
        feature_dim=feature_dim,
        upload_artifacts=upload_artifacts,
    )
    output_dir = Path(save_paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    excel_path = output_dir / f'final_evaluation_tables_{model_name}.xlsx'

    if upload_artifacts:
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

    comparison_cfg = config_mapping.get('evaluation_comparison') or config_mapping.get('comparison') or {}
    base_run_dir = str(
        comparison_cfg.get('base_run_dir', '') or eval_cfg.base_run_dir
    ).strip()
    comparison_result = None
    if base_run_dir:
        comparison_result = compare_runs(
            current_run_dir=save_paths.output_dir,
            base_run_dir=base_run_dir,
            threshold_eval=eval_cfg.threshold_eval,
            averaging_type=eval_cfg.averaging_type,
            top_n=int(comparison_cfg.get('top_n', 20)),
            only_diff=bool(comparison_cfg.get('only_diff', False)),
            output_path=build_path(
                output_root=str(comparison_cfg.get('output_root', output_dir)),
                config_name=str(comparison_cfg.get('config_name', config_name)),
            ),
        )
        logger.report_table(
            title='Evaluation Comparison',
            series='Summary',
            iteration=0,
            table_plot=comparison_result.summary_comparison,
        )
        logger.report_table(
            title='Evaluation Comparison',
            series='Corpora Comparison',
            iteration=0,
            table_plot=comparison_result.corpora_comparison,
        )
        logger.report_table(
            title='Evaluation Comparison',
            series='Classes Comparison',
            iteration=0,
            table_plot=comparison_result.classes_comparison,
        )
        if upload_artifacts:
            task.upload_artifact('evaluation_comparison_summary', artifact_object=comparison_result.summary_comparison)
            task.upload_artifact('evaluation_comparison_classes', artifact_object=comparison_result.classes_comparison)
            task.upload_artifact('evaluation_comparison_corpora', artifact_object=comparison_result.corpora_comparison)
            if comparison_result.excel_path is not None:
                task.upload_artifact('evaluation_comparison_xlsx', artifact_object=str(comparison_result.excel_path))

    with pd.ExcelWriter(excel_path) as writer:
        cv_dev_df.to_excel(excel_writer=writer, sheet_name='dev_cv_summary')
        df_corpora_test.to_excel(excel_writer=writer, sheet_name='test_corpora')
        df_classes_test.to_excel(excel_writer=writer, sheet_name='test_classes')
        if comparison_result is not None:
            comparison_result.corpora_comparison.to_excel(excel_writer=writer, sheet_name='comparison_corpora', index=False)
            comparison_result.classes_comparison.to_excel(excel_writer=writer, sheet_name='comparison_classes', index=False)
            comparison_result.summary_comparison.to_excel(excel_writer=writer, sheet_name='comparison_summary', index=False)
            comparison_result.top_improved_categories.to_excel(excel_writer=writer, sheet_name='comparison_top_up', index=False)
            comparison_result.top_degraded_categories.to_excel(excel_writer=writer, sheet_name='comparison_top_down', index=False)
            comparison_result.hamming_loss_comparison.to_excel(excel_writer=writer, sheet_name='comparison_hamming', index=False)
            comparison_result.pr_auc_per_class.to_excel(excel_writer=writer, sheet_name='comparison_pr_auc', index=False)
            comparison_result.pr_auc_summary.to_excel(excel_writer=writer, sheet_name='comparison_pr_auc_sum', index=False)
    logger.report_text(f'Saved final evaluation tables to {excel_path}')

    objective_metrics = (
        df_corpora_test.loc[objective_corpora].to_dict()
        if objective_corpora in df_corpora_test.index
        else row_all_test
    )
    return EvalResult(
        dev_corpora_df=cv_dev_df,
        test_corpora_df=df_corpora_test,
        test_classes_df=df_classes_test,
        objective_metrics=objective_metrics,
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
    hparam_cnf = cnf['hparam']
    obj_corpora = cnf['objective_corpora']
    down_smpl = cnf.get('downsample_corpora', {})
    use_ent_emb = bool(emb_cnf.get('use_entity_embeddings', True))
    print_logs = bool(cnf.get('print_logs', True))
    debug = bool(cnf.get('debug', False))
    upload_artifacts = bool(cnf.get('upload_artifacts', False))

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
        message='Stage 2/6: Preparing article embeddings (train/test)',
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
            if use_ent_emb
            else 'Stage 3/6: Building article-only datasets (entity embeddings disabled)'
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
        objective_corpora=obj_corpora,
        print_logs=print_logs,
        debug=debug,
        upload_artifacts=upload_artifacts,
    )
    if upload_artifacts:
        task.upload_artifact('cv_objective_metrics', artifact_object=dict(cv_result.objective_metrics))

    log_stage(
        task=task,
        message='Stage 5/6: Training final model on full train (validation=test)',
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
