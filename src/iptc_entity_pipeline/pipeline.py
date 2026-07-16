"""ClearML pipeline orchestration for entity-enhanced IPTC training."""

import logging
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from iptc_entity_pipeline.article_embeddings import ArticleEmbeddingProvider
from iptc_entity_pipeline.clearml_compat import (
    PipelineDecorator,
    Task,
    TaskTypes,
    get_task_logger,
    is_clearml_available,
    set_local_clearml_bypass,
)
from iptc_entity_pipeline.config import BaseCnf, resolve_paths
from iptc_entity_pipeline.data_loading import attach_entities, load_and_normalize, load_wdid_map, sanitize_name
from iptc_entity_pipeline.evaluation.comparison import build_path, compare_runs
from iptc_entity_pipeline.legacy_reuse import evaluateModel
from iptc_entity_pipeline.model_io import save_outputs
from iptc_entity_pipeline.evaluation.reporting import (
    conf_logging,
    log_stage,
    objective_suffix,
    report_cv,
    report_test_curve,
)

from iptc_entity_pipeline.training import train_model

RV_LOAD_DATA = ['corpora']
RV_PREPARE_ARTICLE_EMBEDDINGS = ['articleEmbeddingStats']
RV_BUILD_DATASET = ['trainData', 'testData', 'featureDim']
RV_RUN_CV = ['cvResult']
RV_RUN_ASSEMBLY = ['assemblyResult']
RV_EVAL_FINAL = ['evalResult']
RV_RUN_COMPARISON = ['comparisonResult']
RV_VALIDATE_MEMBER_CATLISTS = ['catList']
RV_TRAIN_BEST = ['trainedModel']


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class EvalResult:
    """Outputs of the final evaluation pipeline step."""

    test_corpora_df: Any
    test_classes_df: Any
    objective_metrics: dict[str, Any]
    scalar_metrics: dict[str, Any]


@PipelineDecorator.component(
    return_values=RV_LOAD_DATA,
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
        remove_types=emb.remove_types,
    )
    attach_entities(
        corpus=corpora.test,
        csv_path=paths.test_csv,
        wdid_mapping=wdid_mapping,
        min_relevance=emb.entity_relevance_threshold,
        remove_types=emb.remove_types,
    )
    return corpora


@PipelineDecorator.component(
    return_values=RV_PREPARE_ARTICLE_EMBEDDINGS,
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
    return_values=RV_BUILD_DATASET,
    execution_queue='iptc_entity_tasks',
    task_type=TaskTypes.data_processing,
)
def build_dataset(
    corpora,
    paths_cnf: Mapping[str, Any],
    emb_cnf: Mapping[str, Any],
    article_embedding_stats: Mapping[str, Any],
    wait_for: Any = None,
):
    """Link embeddings and build train/test embedding datasets.

    The optional ``wait_for`` argument is unused inside the function and
    exists only to express an explicit data dependency in the ClearML
    pipeline DAG (e.g. so an assembly member's ``build_dataset`` only
    starts after the cat-list consistency check has passed).
    """
    from iptc_entity_pipeline.build_dataset import no_entities, report_ent_stats, get_pooling
    from iptc_entity_pipeline.config import EmbeddingCnf, PathsCnf, conf_from_dict
    from iptc_entity_pipeline.dataset_builder import build_emb_data, build_ragged_emb_data
    from iptc_entity_pipeline.entity_embeddings import EntityEmbeddingStore
    from iptc_entity_pipeline.feature_builder import FeatureBuilder

    conf_logging()
    logger = logging.getLogger(__name__)
    paths = conf_from_dict(PathsCnf, paths_cnf)
    emb = conf_from_dict(EmbeddingCnf, emb_cnf)
    if not emb.use_article_embeddings and not emb.use_entity_embeddings:
        raise ValueError('Invalid embedding config: both use_article_embeddings and use_entity_embeddings are False')

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
        return dataset_bundle.train_data, dataset_bundle.test_data, dataset_bundle.feature_dim

    selected_langs = tuple(emb.entity_langs) if emb.entity_langs else (emb.entity_lang,)
    logger.info(
        'Using entity embedding languages=%s mode=%s',
        selected_langs,
        emb.entity_lang_mode,
    )
    entity_store = EntityEmbeddingStore(
        root_dir=paths.entity_embeddings_dir,
        langs=selected_langs,
        lang_mode=emb.entity_lang_mode,
    )
    entity_store.compute_train_mean_from_corpus(corpus=corpora.train)
    pooling = get_pooling(emb_cfg=emb, logger=logger)
    builder = FeatureBuilder(
        article_embedding_provider=article_provider,
        entity_embedding_store=entity_store,
        pooling_strategy=pooling,
        use_article_embeddings=emb.use_article_embeddings,
        combine_method=emb.combine_method,
    )

    task = Task.current_task()
    if emb.entity_pooling == 'no_pooling':
        logger.info('Building ragged no-pooling features for train corpus (%s articles)', len(corpora.train))
        train_ragged = builder.build_ragged_features(
            corpus=corpora.train,
            clearml_logger=task.get_logger() if task is not None else None,
        )
        logger.info('Building ragged no-pooling features for test corpus (%s articles)', len(corpora.test))
        test_ragged = builder.build_ragged_features(
            corpus=corpora.test,
            clearml_logger=task.get_logger() if task is not None else None,
        )
        report_ent_stats(stats=test_ragged.stats, clearml_task=task, logger=logger)
        report_ent_stats(stats=train_ragged.stats, clearml_task=task, logger=logger)
        train_data = build_ragged_emb_data(
            corpus=corpora.train,
            article_matrix=train_ragged.article_matrix,
            entity_matrices=train_ragged.entity_matrices,
        )
        test_data = build_ragged_emb_data(
            corpus=corpora.test,
            article_matrix=test_ragged.article_matrix,
            entity_matrices=test_ragged.entity_matrices,
        )
        feature_dim = int(train_ragged.article_matrix.shape[1])
        if emb_cnf.get('entity_pooling') == 'no_pooling':
            logger.info('emb_cnf.get(\'entity_pooling\') is working')
        else:
            logger.info('emb_cnf.get(\'entity_pooling\') is not working')

        # this is how fix -because of the way how outputs are picled the pipeline fails with oom error 
        train_data.cache_temporary()
        test_data.cache_temporary()
        entity_store.clear_cache()
        return train_data, test_data, feature_dim

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
    entity_store.clear_cache()
    return train_data, test_data, feature_dim


@PipelineDecorator.component(
    return_values=RV_VALIDATE_MEMBER_CATLISTS,
    execution_queue='iptc_entity_tasks',
    task_type=TaskTypes.data_processing,
)
def validate_member_catlists(corpora_primary, corpora_secondary):
    """Assert assembly members share the same train ``catList`` and that
    their test corpora are row-aligned by document id.

    Used in the assembly pipeline path as a gate between ``load_data`` and
    ``build_dataset``. Two checks must pass:

    1. ``train.catList`` is identical (same ids, same order). Required so
       per-class model selection is meaningful.
    2. ``test`` corpora have the same length and the same ``doc.id``
       sequence in the same order. Required so the per-member predictions
       built later by :class:`AssemblyModel` can be combined row-by-row.

    Returns the canonical category id list so downstream steps can express
    an explicit dependency via ``wait_for=catList``.

    :param corpora_primary: ``CorpusGroup`` from member 0's ``load_data``.
    :param corpora_secondary: ``CorpusGroup`` from member 1's ``load_data``.
    :raises ValueError: On any of the mismatches above.
    :return: The shared train category id list.
    """
    conf_logging()
    logger = logging.getLogger(__name__)
    cat_primary = list(corpora_primary.train.catList)
    cat_secondary = list(corpora_secondary.train.catList)
    if cat_primary != cat_secondary:
        primary_set = set(cat_primary)
        secondary_set = set(cat_secondary)
        missing = sorted(primary_set - secondary_set)
        extra = sorted(secondary_set - primary_set)
        order_changed = primary_set == secondary_set and cat_primary != cat_secondary
        raise ValueError(
            'Assembly catList mismatch between members: '
            f'missing_in_secondary={missing[:10]}{"..." if len(missing) > 10 else ""} '
            f'extra_in_secondary={extra[:10]}{"..." if len(extra) > 10 else ""} '
            f'order_changed={order_changed}'
        )

    primary_test_ids = [str(doc.id) for doc in corpora_primary.test]
    secondary_test_ids = [str(doc.id) for doc in corpora_secondary.test]
    if len(primary_test_ids) != len(secondary_test_ids):
        raise ValueError(
            'Assembly test corpus length mismatch: '
            f'primary={len(primary_test_ids)} secondary={len(secondary_test_ids)}'
        )
    if primary_test_ids != secondary_test_ids:
        diff_idx = next(
            (i for i, (a, b) in enumerate(zip(primary_test_ids, secondary_test_ids)) if a != b),
            None,
        )
        sample = (
            f'first_diff_idx={diff_idx} '
            f'primary={primary_test_ids[diff_idx]!r} '
            f'secondary={secondary_test_ids[diff_idx]!r}'
            if diff_idx is not None else 'order_only_difference'
        )
        raise ValueError(f'Assembly test corpus doc-id order differs: {sample}')

    logger.info(
        f'Assembly: catList consistent (n_classes={len(cat_primary)}) '
        f'and test corpus aligned (n_docs={len(primary_test_ids)})'
    )
    return cat_primary


@PipelineDecorator.component(
    return_values=RV_RUN_CV,
    execution_queue='iptc_entity_tasks',
    task_type=TaskTypes.training,
)
def run_cv(
    train_data,
    feature_dim: int,
    model_cnf: Mapping[str, Any],
    hparam_cnf: Mapping[str, Any],
    train_cnf: Mapping[str, Any],
    eval_cnf: Mapping[str, Any],
    cv_cnf: Mapping[str, Any],
    optuna_cnf: Mapping[str, Any],
    tuning_cnf: Mapping[str, Any],
    objective_row: str,
    random_seed: int,
    print_logs: bool = True,
    upload_artifacts: bool = False,
    eval_thresholds: Mapping[str, float] | None = None,
):
    """Run mandatory CV over train and select best hyperparameter combination.

    :param eval_thresholds: Optional per-class thresholds applied during
        per-fold evaluation. Used by the assembly pipeline so each member's
        per-class CV F1 is measured at that member's externally-loaded
        thresholds. When provided, it is also echoed into
        ``cv.tuned_thresholds`` and threshold tuning is bypassed.
    :return: Pickle-safe :class:`CvOutputs` with all CV results.
    """
    from dataclasses import asdict

    from iptc_entity_pipeline.config import (
        CvCnf,
        EvaluationCnf,
        HyperparamSpace,
        ModelCnf,
        OptunaCnf,
        ThresholdTuningCnf,
        TrainingCnf,
        conf_from_dict,
    )
    from iptc_entity_pipeline.cross_validation import CV
    from iptc_entity_pipeline.seeding import set_global_seed

    conf_logging()
    set_global_seed(seed=int(random_seed))
    task = Task.current_task()
    clearml_logger = get_task_logger(task=task, logger=logging.getLogger(__name__))

    space = conf_from_dict(HyperparamSpace, hparam_cnf)
    base_model = conf_from_dict(ModelCnf, model_cnf)
    base_training = conf_from_dict(TrainingCnf, train_cnf)
    eval_cfg = conf_from_dict(EvaluationCnf, eval_cnf)
    cv_cfg = conf_from_dict(CvCnf, cv_cnf)
    optuna_cfg = conf_from_dict(OptunaCnf, optuna_cnf)
    tuning_cfg = conf_from_dict(ThresholdTuningCnf, tuning_cnf)

    if task is not None:
        task.connect(asdict(space), name='hyperparamSpace')
        task.connect(asdict(base_model), name='modelConfig')
        task.connect(asdict(base_training), name='trainingConfig')
        task.connect(asdict(eval_cfg), name='evaluationConfig')
        task.connect(asdict(cv_cfg), name='cvConfig')
        task.connect(asdict(optuna_cfg), name='optunaConfig')
        task.connect(asdict(tuning_cfg), name='thresholdTuningConfig')
    
    train_data.load_temporary()
        
    cv = CV(
        model_cnf=base_model,
        hparam_cnf=space,
        train_cnf=base_training,
        eval_cnf=eval_cfg,
        cv_cnf=cv_cfg,
        optuna_cnf=optuna_cfg,
        tuning_cnf=tuning_cfg,
        objective_row=objective_row,
        random_seed=int(random_seed),
        eval_thresholds=eval_thresholds,
    )
    cv.fit(
        train_data=train_data,
        feature_dim=feature_dim,
        print_logs=print_logs,
        clearml_logger=clearml_logger,
    )

    report = cv.prepare_report()
    report_cv(
        task=task,
        logger=clearml_logger,
        report=report,
        upload_artifacts=upload_artifacts,
    )

    return cv.export_outputs()

@PipelineDecorator.component(
    return_values=RV_RUN_ASSEMBLY,
    execution_queue='iptc_entity_tasks',
    task_type=TaskTypes.training,
)
def run_assembly_step(
    member_cv_results,
    member_labels,
    cat_list,
    member_loaded_thresholds,
    eval_cnf: Mapping[str, Any],
    objective_row: str,
    print_logs: bool = True,
    upload_artifacts: bool = False,
    mapping_artifact_name: str = 'assembly_class_to_model',
    sign_test: bool = False,
):
    """Build the assembly from each member's pre-computed :class:`CV` result.

    No training of its own: each member must have already gone through
    :func:`run_cv` (with ``eval_thresholds`` set to that member's loaded
    thresholds) so ``per_class_df`` is populated and per-class F1 was
    measured at the production thresholds.

    :param member_cv_results: One fitted :class:`CV` instance per member.
    :param member_labels: Display labels (member 0 is primary; ties on F1
        resolve to it).
    :param cat_list: Shared train ``catList`` (validated upstream).
    :param member_loaded_thresholds: Per-member externally-loaded
        threshold maps. Each selected class's stitched threshold comes
        from the winning member's entry here.
    """
    from iptc_entity_pipeline.assembly import build_assembly_from_cv, report_assembly_tables
    from iptc_entity_pipeline.config import EvaluationCnf, conf_from_dict

    conf_logging()
    logger = logging.getLogger(__name__)
    task = Task.current_task()
    clearml_logger = get_task_logger(task=task, logger=logger)

    eval_cfg = conf_from_dict(EvaluationCnf, eval_cnf)
    if task is not None:
        task.connect(eval_cnf, name='evaluationConfig')

    logger.info(
        f'Assembly step: members={list(member_labels)} '
        f'(consuming per-member CV results)'
    )

    assembly_result = build_assembly_from_cv(
        member_cv_results=list(member_cv_results),
        member_labels=list(member_labels),
        cat_list=list(cat_list),
        eval_cfg=eval_cfg,
        objective_row=objective_row,
        primary_idx=0,
        member_loaded_thresholds=(
            list(member_loaded_thresholds)
            if member_loaded_thresholds is not None else None
        ),
        sign_test=bool(sign_test),
    )

    report_assembly_tables(
        clearml_logger=clearml_logger,
        assembly_result=assembly_result,
        member_labels=list(member_labels),
        print_logs=print_logs,
    )

    mapping_payload = {
        'schema_version': '1',
        'member_labels': list(assembly_result.class_to_model.member_labels),
        'primary_index': 0,
        'assignments': dict(assembly_result.class_to_model.assignments),
        'stitched_thresholds': dict(assembly_result.tuned_thresholds),
    }
    if task is not None:
        task.upload_artifact(mapping_artifact_name, artifact_object=mapping_payload)
    if upload_artifacts and task is not None:
        task.upload_artifact(
            'assembly_per_class_f1', artifact_object=assembly_result.per_class_f1_df,
        )
        task.upload_artifact(
            'assembly_per_corpora', artifact_object=assembly_result.per_corpora_df,
        )
        task.upload_artifact(
            'assembly_threshold_report', artifact_object=assembly_result.threshold_report_df,
        )
        task.upload_artifact(
            'assembly_member_summary', artifact_object=assembly_result.cv_dev_df,
        )

    return assembly_result


@PipelineDecorator.component(
    return_values=RV_TRAIN_BEST,
    execution_queue='iptc_entity_tasks',
    task_type=TaskTypes.training,
)
def train_best(
    train_data,
    feature_dim: int,
    best_model_cnf: Mapping[str, Any],
    train_cnf: Mapping[str, Any],
    random_seed: int,
    test_data=None,
    print_logs: bool = True,
):
    """Train final model on full train set with best hyperparams from CV.

    :param test_data: Optional monitoring split for per-epoch curves. When
        omitted, metrics are computed on ``train_data`` (no test leakage).
    """
    from iptc_entity_pipeline.config import ModelCnf, TrainingCnf, conf_from_dict
    from iptc_entity_pipeline.seeding import set_global_seed

    conf_logging()
    set_global_seed(seed=int(random_seed))
    logger = logging.getLogger(__name__)
    task = Task.current_task()
    clearml_logger = get_task_logger(task=task, logger=logger)

    model_cfg = conf_from_dict(ModelCnf, best_model_cnf)
    train_cfg = conf_from_dict(TrainingCnf, train_cnf)

    if task is not None:
        task.connect(best_model_cnf, name='bestModelConfig')
        task.connect(train_cnf, name='bestTrainingConfig')
    train_data.load_temporary()
    if test_data is not None:
        test_data.load_temporary()
    monitor_data = test_data if test_data is not None else train_data
    curve_label = 'test' if test_data is not None else 'train'
    validation_split_name = 'test' if test_data is not None else 'train'
    logger.info(
        f'Training final model: hidden_dim={model_cfg.hidden_dim}, '
        f'dropouts=({model_cfg.dropouts1}, {model_cfg.dropouts2}), '
        f'lr={train_cfg.learning_rate}, batch_size={train_cfg.batch_size}, '
        f'epochs={train_cfg.epochs}, early_stopping_patience={train_cfg.early_stopping_patience}, '
        f'feature_dim={feature_dim}, train_docs={len(train_data.corpus)}, '
        f'monitor_docs={len(monitor_data.corpus)}, monitor_split={curve_label}'
    )
    if test_data is not None and train_cfg.early_stopping_patience != 0:
        raise ValueError('Final retraining on test monitor split requires early stopping to be disabled')

    try:
        result = train_model(
            train_data=train_data,
            dev_data=monitor_data,
            feature_dim=feature_dim,
            model_config=model_cfg,
            training_config=train_cfg,
            print_logs=print_logs,
            validation_split_name=validation_split_name,
        )
    finally:
        train_data.cleanup_temporary()

    report_test_curve(logger=clearml_logger, result=result, dev_series=curve_label)
    logger.info(
        f'Final model training complete: epochs={result.epochs_run}, '
        f'final_dev_loss={result.final_dev_loss:.6f}'
    )
    return result.model

@PipelineDecorator.component(
    return_values=RV_EVAL_FINAL,
    execution_queue='iptc_entity_tasks',
    task_type=TaskTypes.testing,
)
def eval_final(
    trained_model,
    test_data,
    eval_cnf: Mapping[str, Any],
    emb_cnf: Mapping[str, Any],
    objective_row: str,
    config_name: str,
    config_mapping: Mapping[str, Any],
    feature_dim: int,
    tuned_thresholds: Mapping[str, float] | None = None,
    threshold_report_df: Any = None,
    upload_artifacts: bool = False,
):
    """Evaluate final model on test; CV summaries are reported only in ``run_cv``."""
    import logging
    from dataclasses import asdict
    from pathlib import Path

    from iptc_entity_pipeline.config import EmbeddingCnf, EvaluationCnf, conf_from_dict
    from iptc_entity_pipeline.data_loading import sanitize_name
    from iptc_entity_pipeline.evaluation.comparison import build_path, compare_runs
    from iptc_entity_pipeline.legacy_reuse import evaluateModel
    from iptc_entity_pipeline.model_io import export_eval_excel, save_outputs
    from iptc_entity_pipeline.pipeline import EvalResult
    from iptc_entity_pipeline.evaluation.reporting import (
        build_test_scalar_metrics,
        conf_logging,
        report_test_eval_scalars,
        report_test_eval_tables,
    )
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
        if upload_artifacts and task is not None:
            task.upload_artifact('evaluation_comparison_summary', artifact_object=result.summary_comparison)
            task.upload_artifact('evaluation_comparison_classes', artifact_object=result.classes_comparison)
            task.upload_artifact('evaluation_comparison_corpora', artifact_object=result.corpora_comparison)
            if result.excel_path is not None:
                task.upload_artifact('evaluation_comparison_xlsx', artifact_object=str(result.excel_path))
        return result

    def upload_eval_artifacts():
        """Upload eval-step artifacts to ClearML (test outputs only; CV in ``run_cv``)."""
        if task is None:
            return
        objective_row_name = f'All-{eval_cfg.averaging_type}'
        task.upload_artifact('test_corpora_dataframe', artifact_object=df_corpora_test)
        task.upload_artifact('test_classes_dataframe', artifact_object=df_classes_test)
        task.upload_artifact('final_evaluation_tables_xlsx', artifact_object=str(excel_path))
        task.upload_artifact('saved_model_paths', artifact_object=asdict(save_paths))
        task.upload_artifact(
            'evaluation_thresholds',
            artifact_object={
                'threshold_predict': eval_cfg.threshold_predict,
                'threshold_eval': eval_cfg.threshold_eval,
                'objective_row': objective_row,
                'objective_row_name': objective_row_name,
            },
        )
        # Compatibility aliases kept to match naming from baseline runs.
        task.upload_artifact('Corpora Dataframe', artifact_object=df_corpora_test)
        task.upload_artifact('Classes Dataframe', artifact_object=df_classes_test)

    conf_logging()
    logger = logging.getLogger(__name__)
    task = Task.current_task()
    clearml_logger = get_task_logger(task=task, logger=logger)

    eval_cfg = conf_from_dict(EvaluationCnf, eval_cnf)
    emb_cfg = conf_from_dict(EmbeddingCnf, emb_cnf)
    if task is not None:
        task.connect(eval_cnf, name='evaluationConfig')
        task.connect(emb_cnf, name='embeddingConfig')

    custom_thresholds = dict(tuned_thresholds) if tuned_thresholds else None
    logger.info(
        f'Evaluating final model on test: test_docs={len(test_data.corpus)}, '
        f'objective={objective_row}, custom_thresholds='
        f'{len(custom_thresholds) if custom_thresholds else 0} class(es)'
    )

    from iptc_entity_pipeline.legacy_reuse import wgh_labels_from_score_matrix

    test_data.load_temporary()
    try:
        df_corpora_test, df_classes_test, pred_score_matrix = evaluateModel(
            model=trained_model,
            evalData=test_data,
            evaluation_config=eval_cfg,
            customThresholds=custom_thresholds,
            returnPredictions=True,
        )

        scalar_metrics = build_test_scalar_metrics(
            df_corpora_test=df_corpora_test,
            df_classes_test=df_classes_test,
            objective_row=objective_row,
        )
        report_test_eval_scalars(
            clearml_logger=clearml_logger,
            df_corpora_test=df_corpora_test,
            df_classes_test=df_classes_test,
            objective_row=objective_row,
        )
        report_test_eval_tables(
            clearml_logger=clearml_logger,
            df_corpora_test=df_corpora_test,
            df_classes_test=df_classes_test,
        )

        pred_scores_legacy = list(
            wgh_labels_from_score_matrix(
                score_matrix=pred_score_matrix,
                cat_list=list(trained_model.catList),
            )
        )
        del pred_score_matrix
        save_paths = save_outputs(
            model=trained_model,
            test_data=test_data,
            pred_scores=pred_scores_legacy,
            eval_cnf=eval_cfg,
            emb_cnf=emb_cfg,
            config_mapping=config_mapping,
            config_name=config_name,
            feature_dim=feature_dim,
            tuned_thresholds=custom_thresholds,
            threshold_report_df=threshold_report_df,
            upload_artifacts=upload_artifacts,
        )

        comparison_result = run_comparison()

        output_dir = Path(save_paths.output_dir)
        model_name = sanitize_name(value=config_name)
        excel_path = output_dir / f'final_evaluation_tables_{model_name}.xlsx'
        export_eval_excel(
            excel_path=excel_path,
            df_corpora_test=df_corpora_test,
            df_classes_test=df_classes_test,
            comparison_result=comparison_result,
        )

        if upload_artifacts:
            upload_eval_artifacts()

        objective_metrics = df_corpora_test.loc[objective_row].to_dict()
        obj_suffix = objective_suffix(objective_row)
        logger.info(
            f'Evaluation complete: F1_{obj_suffix}={scalar_metrics[f"F1_{obj_suffix}"]:.4f}, '
            f'F1_macro_relevant={scalar_metrics["F1_macro_relevant"]:.4f}, config={config_name}'
        )
        return EvalResult(
            test_corpora_df=df_corpora_test,
            test_classes_df=df_classes_test,
            objective_metrics=objective_metrics,
            scalar_metrics=scalar_metrics,
        )
    finally:
        test_data.cleanup_temporary()


def _run_assembly_training_pipeline(
    *,
    cnf: Mapping[str, Any],
    assembly_cnf: Mapping[str, Any],
    eval_cnf: Mapping[str, Any],
    cv_cnf: Mapping[str, Any],
    objective_row: str,
    downsample_corpora: Mapping[str, float],
    config_name: str,
    print_logs: bool,
    upload_artifacts: bool,
    random_seed: int,
) -> None:
    """Run the dual-model assembly variant of the training pipeline.

    Step ordering (parallel where possible, serialized by data-flow only):

    1) ``load_data`` for each member — independent, in parallel.
    2) ``validate_member_catlists`` — gate that depends on every member's
       corpora; raises if their train ``catList`` differ.
    3) ``prepare_article_embeddings`` for each member — independent of the
       gate, in parallel with it.
    4) ``build_dataset`` for each member — in parallel; each call carries
       ``wait_for=cat_list_token`` so neither build starts until the
       cat-list check has passed.
    5) ``run_cv`` per member — full HPO + k-fold CV per member, with each
       member's per-fold evaluation pinned to that member's externally
       loaded per-class thresholds (``eval_thresholds=loaded``). Yields
       per-class CV F1 (mean+std) measured at production thresholds.
    6) ``run_assembly_step`` — pure aggregation: pick per-class winners
       and stitch loaded thresholds.
    7) ``train_best`` per member — uses each member's CV-selected best
       model and training configs.
    8) ``eval_final`` on the ``AssemblyModel`` wrapping both trained members.
    """
    from pathlib import Path

    from iptc_entity_pipeline.assembly import AssemblyModel, load_thresholds

    task = Task.current_task()
    members_raw = list(assembly_cnf.get('members') or ())
    if len(members_raw) != 2:
        raise ValueError(
            f'assembly.members must contain exactly 2 members; got {len(members_raw)}'
        )

    member_labels = [str(m.get('label') or f'member_{idx}') for idx, m in enumerate(members_raw)]
    member_configs = [m['config'] for m in members_raw]
    member_paths = [c['paths'] for c in member_configs]
    member_emb = [c['emb'] for c in member_configs]
    member_model_cnf_dicts = [c['model'] for c in member_configs]
    member_train_cnf_dicts = [c['train'] for c in member_configs]
    member_hparam_cnfs = [c.get('hparam', {}) for c in member_configs]
    member_cv_cnfs = [c['cv'] for c in member_configs]
    member_optuna_cnfs = [c.get('optuna', {}) for c in member_configs]
    member_tuning_cnfs = [c.get('tuning', {'enabled': False}) for c in member_configs]
    member_random_seeds = [int(c.get('random_seed', random_seed)) for c in member_configs]

    log_stage(
        task=task,
        message=f'Assembly mode: {len(members_raw)} members={member_labels}',
        print_logs=print_logs,
    )

    log_stage(
        task=task,
        message='Assembly stage 1: load_data per member (parallelizable)',
        print_logs=print_logs,
    )
    member_corpora = [
        load_data(
            paths_cnf=member_paths[idx],
            emb_cnf=member_emb[idx],
            downsample_corpora=downsample_corpora,
        )
        for idx in range(len(members_raw))
    ]

    log_stage(
        task=task,
        message='Assembly stage 2: validate catList consistency across members',
        print_logs=print_logs,
    )
    cat_list_token = validate_member_catlists(
        corpora_primary=member_corpora[0],
        corpora_secondary=member_corpora[1],
    )

    log_stage(
        task=task,
        message='Assembly stage 3: prepare_article_embeddings per member (parallelizable)',
        print_logs=print_logs,
    )
    member_article_stats = [
        prepare_article_embeddings(
            corpora=member_corpora[idx],
            paths_cnf=member_paths[idx],
            emb_cnf=member_emb[idx],
        )
        for idx in range(len(members_raw))
    ]
    if upload_artifacts and task is not None:
        for idx, label in enumerate(member_labels):
            task.upload_artifact(
                f'article_embedding_stats_{label}',
                artifact_object=dict(member_article_stats[idx]),
            )

    log_stage(
        task=task,
        message='Assembly stage 4: build_dataset per member (parallel, gated on catList check)',
        print_logs=print_logs,
    )
    member_train_data: list[Any] = []
    member_test_data: list[Any] = []
    member_feature_dims: list[int] = []
    for idx in range(len(members_raw)):
        m_train, m_test, m_dim = build_dataset(
            corpora=member_corpora[idx],
            paths_cnf=member_paths[idx],
            emb_cnf=member_emb[idx],
            article_embedding_stats=member_article_stats[idx],
            wait_for=cat_list_token,
        )
        member_train_data.append(m_train)
        member_test_data.append(m_test)
        member_feature_dims.append(m_dim)

    cat_list = list(member_train_data[0].corpus.catList)
    default_threshold = float(eval_cnf.get('threshold_eval', 0.5))
    member_loaded_thresholds: list[dict[str, float]] = []
    for member_idx, m in enumerate(members_raw):
        thr_path = str(m.get('thresholds_path', '') or '')
        if not thr_path:
            log_stage(
                task=task,
                message=(
                    f'Assembly member {member_labels[member_idx]} has no '
                    f'thresholds_path; using global threshold_eval={default_threshold}'
                ),
                print_logs=print_logs,
            )
            member_loaded_thresholds.append(
                {str(cid): default_threshold for cid in cat_list}
            )
            continue
        member_loaded_thresholds.append(
            load_thresholds(
                path=Path(thr_path),
                cat_list=cat_list,
                default_threshold=default_threshold,
            )
        )

    log_stage(
        task=task,
        message=(
            f'Assembly stage 5: run_cv per member with eval_thresholds=loaded '
            f'({cv_cnf.get("folds", 5)} folds, tuning forced off)'
        ),
        print_logs=print_logs,
    )
    member_cv_results: list[Any] = []
    for idx in range(len(members_raw)):
        cv_result = run_cv(
            train_data=member_train_data[idx],
            feature_dim=member_feature_dims[idx],
            model_cnf=member_model_cnf_dicts[idx],
            hparam_cnf=member_hparam_cnfs[idx],
            train_cnf=member_train_cnf_dicts[idx],
            eval_cnf=eval_cnf,
            cv_cnf=member_cv_cnfs[idx],
            optuna_cnf=member_optuna_cnfs[idx],
            tuning_cnf=member_tuning_cnfs[idx],
            objective_row=objective_row,
            random_seed=member_random_seeds[idx],
            print_logs=print_logs,
            upload_artifacts=upload_artifacts,
            eval_thresholds=member_loaded_thresholds[idx],
        )
        member_cv_results.append(cv_result)

    log_stage(
        task=task,
        message='Assembly stage 6: aggregate per-member CV into per-class assembly',
        print_logs=print_logs,
    )
    assembly_result = run_assembly_step(
        member_cv_results=member_cv_results,
        member_labels=member_labels,
        cat_list=cat_list,
        member_loaded_thresholds=member_loaded_thresholds,
        eval_cnf=eval_cnf,
        objective_row=objective_row,
        print_logs=print_logs,
        upload_artifacts=upload_artifacts,
        mapping_artifact_name=str(assembly_cnf.get('mapping_artifact_name', 'assembly_class_to_model')),
        sign_test=bool(assembly_cnf.get('sign_test', False)),
    )

    log_stage(
        task=task,
        message='Assembly stage 7: train_best per member (parallelizable)',
        print_logs=print_logs,
    )
    trained_members: list[Any] = []
    for idx in range(len(members_raw)):
        member_cv = member_cv_results[idx]
        trained = train_best(
            train_data=member_train_data[idx],
            test_data=member_test_data[idx],
            feature_dim=member_feature_dims[idx],
            best_model_cnf=asdict(member_cv.best_model_config),
            train_cnf=asdict(member_cv.best_training_config),
            random_seed=member_random_seeds[idx],
            print_logs=print_logs,
        )
        trained_members.append(trained)

    assembled_model = AssemblyModel(
        members=trained_members,
        cat_list=cat_list,
        class_to_model=assembly_result.class_to_model,
        member_eval_data={i: member_test_data[i] for i in range(len(members_raw))},
        member_feature_dims=member_feature_dims,
    )

    log_stage(
        task=task,
        message='Assembly stage 8: evaluating assembled model on test',
        print_logs=print_logs,
    )
    # eval_final's `test_data` is used only as the gold-label / corpus carrier
    # (for evaluate_predictions, save_outputs, etc). The actual per-member
    # prediction dispatch happens inside AssemblyModel.classifyDataset using
    # the per-member datasets registered above. Member 0's test corpus is
    # row-aligned with every other member's by validate_member_catlists.
    eval_result = eval_final(
        trained_model=assembled_model,
        test_data=member_test_data[0],
        eval_cnf=eval_cnf,
        emb_cnf=member_emb[0],
        objective_row=objective_row,
        config_name=config_name,
        config_mapping=cnf,
        feature_dim=member_feature_dims[0],
        tuned_thresholds=assembly_result.tuned_thresholds,
        threshold_report_df=assembly_result.threshold_report_df,
        upload_artifacts=upload_artifacts,
    )
    from iptc_entity_pipeline.evaluation.reporting import report_test_eval_scalars

    report_test_eval_scalars(
        clearml_logger=get_task_logger(task=task, logger=logging.getLogger(__name__)),
        df_corpora_test=eval_result.test_corpora_df,
        df_classes_test=eval_result.test_classes_df,
        objective_row=objective_row,
    )

    if upload_artifacts and task is not None:
        task.upload_artifact('pipeline_config', artifact_object=dict(cnf))
        task.upload_artifact('objective_metrics', artifact_object=dict(eval_result.objective_metrics))
        task.upload_artifact('test_scalar_metrics', artifact_object=dict(eval_result.scalar_metrics))
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
    if task is not None:
        task.connect(cnf, name='pipelineConfig')
    config_name = str(cnf.get('config_name', 'wpentities'))
    if task is not None:
        task.add_tags([config_name])

    from iptc_entity_pipeline.seeding import set_global_seed
    from iptc_entity_pipeline.evaluation.reporting import report_cv_std, report_eval, report_test_eval_scalars

    paths_cnf = cnf['paths']
    emb_cnf = cnf['emb']
    model_cnf = cnf['model']
    train_cnf = cnf['train']
    eval_cnf = cnf['eval']
    cv_cnf = cnf['cv']
    optuna_cnf = cnf.get('optuna', {})
    tuning_cnf = cnf.get('tuning', {'enabled': False})
    assembly_cnf = cnf.get('assembly') or {'enabled': False}
    hparam_cnf = cnf['hparam']
    obj_row = cnf['objective_row']
    down_smpl = cnf.get('downsample_corpora', {})
    use_art_emb = bool(emb_cnf.get('use_article_embeddings', True))
    use_ent_emb = bool(emb_cnf.get('use_entity_embeddings', True))
    print_logs = bool(cnf.get('print_logs', True))
    upload_artifacts = bool(cnf.get('upload_artifacts', False))
    random_seed = int(cnf.get('random_seed', 43))

    set_global_seed(seed=random_seed)

    if assembly_cnf.get('enabled'):
        _run_assembly_training_pipeline(
            cnf=cnf,
            assembly_cnf=assembly_cnf,
            eval_cnf=eval_cnf,
            cv_cnf=cv_cnf,
            objective_row=obj_row,
            downsample_corpora=down_smpl,
            config_name=config_name,
            print_logs=print_logs,
            upload_artifacts=upload_artifacts,
            random_seed=random_seed,
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
    if upload_artifacts and task is not None:
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
        
    model_path = str(cnf.get('model_path') or '')

    if model_path:
        from geneea.catlib.model.nnet import NeuralCategModel
        from iptc_entity_pipeline.evaluation.comparison import load_custom_thresholds

        log_stage(
            task=task,
            message=f'Stages 4-5 skipped: loading pre-trained model from {model_path}',
            print_logs=print_logs,
        )
        model_dir = Path(model_path)
        model_file = model_dir / 'model.nn.bin' if model_dir.is_dir() else model_dir
        if not model_file.is_file():
            raise FileNotFoundError(f'Model file does not exist: {model_file}')
        trained_model = NeuralCategModel.load(str(model_file))
        run_dir = model_dir if model_dir.is_dir() else model_dir.parent
        tuned_thresholds = load_custom_thresholds(run_dir=run_dir) or None
        threshold_report_df = None
    else:
        log_stage(
            task=task,
            message=f'Stage 4/6: Running mandatory {cv_cnf.get("folds", 5)}-fold cross-validation on train',
            print_logs=print_logs,
        )
        cv = run_cv(
            train_data=train_data,
            feature_dim=feature_dim,
            model_cnf=model_cnf,
            hparam_cnf=hparam_cnf,
            train_cnf=train_cnf,
            eval_cnf=eval_cnf,
            cv_cnf=cv_cnf,
            optuna_cnf=optuna_cnf,
            tuning_cnf=tuning_cnf,
            objective_row=obj_row,
            random_seed=random_seed,
            print_logs=print_logs,
            upload_artifacts=upload_artifacts,
        )
        if upload_artifacts and task is not None:
            task.upload_artifact('cv_objective_metrics', artifact_object=dict(cv.best_trial_stats))
        logger = get_task_logger(task=task, logger=logging.getLogger(__name__))
        report_eval(
            logger=logger,
            title='Cross Validation Results',
            row=cv.best_trial_stats,
            iteration=0,
        )
        report_cv_std(
            logger=logger,
            row=cv.best_trial_stats,
            title='Cross Validation Results',
            iteration=0,
        )

        log_stage(
            task=task,
            message='Stage 5/6: Training final model on full train with fixed CV-derived epochs',
            print_logs=print_logs,
        )
        trained_model = train_best(
            train_data=train_data,
            feature_dim=feature_dim,
            best_model_cnf=asdict(cv.best_model_config),
            train_cnf=asdict(cv.best_training_config),
            random_seed=random_seed,
            print_logs=print_logs,
        )
        tuned_thresholds = cv.tuned_thresholds
        threshold_report_df = cv.threshold_report

    log_stage(
        task=task,
        message='Stage 6/6: Evaluating final model on test',
        print_logs=print_logs,
    )
    eval_result = eval_final(
        trained_model=trained_model,
        test_data=test_data,
        eval_cnf=eval_cnf,
        emb_cnf=emb_cnf,
        objective_row=obj_row,
        config_name=config_name,
        config_mapping=cnf,
        feature_dim=feature_dim,
        tuned_thresholds=tuned_thresholds,
        threshold_report_df=threshold_report_df,
        upload_artifacts=upload_artifacts,
    )
    # Log on the pipeline controller from scalar_metrics (pickle-safe; includes F1_macro_relevant).
    report_test_eval_scalars(
        clearml_logger=get_task_logger(task=task, logger=logging.getLogger(__name__)),
        df_corpora_test=eval_result.test_corpora_df,
        df_classes_test=eval_result.test_classes_df,
        objective_row=obj_row,
    )

    if upload_artifacts and task is not None:
        task.upload_artifact('pipeline_config', artifact_object=dict(cnf))
        task.upload_artifact('objective_metrics', artifact_object=dict(eval_result.objective_metrics))
        task.upload_artifact('test_scalar_metrics', artifact_object=dict(eval_result.scalar_metrics))
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
    if not is_local and not is_clearml_available():
        raise RuntimeError(
            'ClearML is not installed. Install package "clearml" to run non-local pipeline mode, '
            'or rerun with --local.'
        )
    if is_local:
        set_local_clearml_bypass(enabled=True)
    try:
        if is_local:
            PipelineDecorator.run_locally()
        run_training_pipeline(cnf=config_mapping)
    finally:
        if is_local:
            set_local_clearml_bypass(enabled=False)
    return resolved_config, config_mapping
