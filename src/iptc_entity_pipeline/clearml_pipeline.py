"""ClearML pipeline orchestration for entity-enhanced IPTC training."""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import numpy as np
from clearml import Task, TaskTypes
from clearml.automation.controller import PipelineDecorator

from iptc_entity_pipeline.article_embeddings import ArticleEmbeddingProvider, EmbeddingCacheStats
from iptc_entity_pipeline.config import (
    BaseConfig,
    CvConfig,
    EmbeddingConfig,
    EvaluationConfig,
    HyperparamSpace,
    ModelConfig,
    PathsConfig,
    TrainingConfig,
    config_from_dict,
    resolve_paths,
)
from iptc_entity_pipeline.data_loading import (
    attach_entities_to_corpus,
    get_article_text,
    get_doc_wdids,
    load_and_normalize_corpora,
    load_wdid_mapping,
)
from iptc_entity_pipeline.dataset_builder import (
    build_embedding_dataset,
    build_multilabel_targets,
    slice_dataset,
    to_numpy_array,
)
from iptc_entity_pipeline.entity_embeddings import EntityEmbeddingStore
from iptc_entity_pipeline.evaluation_comparison import build_output_path, compare_runs
from iptc_entity_pipeline.feature_builder import FeatureBuilder
from iptc_entity_pipeline.legacy_reuse import evaluateModel
from iptc_entity_pipeline.model_io import save_final_model_outputs
from iptc_entity_pipeline.pooling import SumEntityPooling
from iptc_entity_pipeline.reporting import (
    log_stage,
    report_cv_fold_curve_charts,
    report_cv_outputs,
    report_cv_std_scalars,
    report_eval_scalars,
    report_train_test_curve_charts,
)
from iptc_entity_pipeline.training import (
    CvFoldCurves,
    combo_params_json,
    get_obj_row,
    train_model,
)


@dataclass(frozen=True)
class EntityEmbeddingStats:
    """Coverage statistics for entity embeddings linked to a corpus."""

    use_entity_embeddings: bool = True
    entity_dim: int = 0
    linked_unique_wdids: int = 0
    found_embeddings: int = 0
    missing_embeddings: int = 0


@dataclass(frozen=True)
class DatasetBundle:
    """Outputs of the dataset-building pipeline step."""

    train_data: Any
    test_data: Any
    feature_dim: int
    entity_embedding_stats: EntityEmbeddingStats


@dataclass(frozen=True)
class CvResult:
    """Outputs of the cross-validation pipeline step."""

    cv_dev_df: Any
    best_model_config: dict[str, Any]
    best_training_config: dict[str, Any]
    objective_metrics: dict[str, Any]


@dataclass(frozen=True)
class EvalResult:
    """Outputs of the final evaluation pipeline step."""

    dev_corpora_df: Any
    test_corpora_df: Any
    test_classes_df: Any
    objective_metrics: dict[str, Any]


@dataclass
class FoldScores:
    """Per-fold metric accumulator for cross-validation."""

    loss: list[float] = field(default_factory=list)
    epochs: list[float] = field(default_factory=list)
    precision: list[float] = field(default_factory=list)
    recall: list[float] = field(default_factory=list)
    f1: list[float] = field(default_factory=list)



@PipelineDecorator.component(
    return_values=['corpora'],
    execution_queue='iptc_entity_tasks',
    task_type=TaskTypes.data_processing,
)
def load_data(
    paths_config: Mapping[str, Any],
    downsample_corpora: Mapping[str, float] | None = None,
):
    """Load normalized corpora with linked entities attached to each document."""
    from iptc_entity_pipeline.config import PathsConfig, config_from_dict

    paths = config_from_dict(PathsConfig, paths_config)
    corpora = load_and_normalize_corpora(
        train_csv=paths.train_csv,
        test_csv=paths.test_csv,
        removed_cat_ids=paths.removed_cat_ids,
        downsample_corpora=downsample_corpora or {},
        downsampling_order_cache_json=paths.downsampling_order_cache_json,
    )
    wdid_mapping = load_wdid_mapping(wdid_mapping_tsv=paths.wdid_mapping_tsv)
    attach_entities_to_corpus(corpus=corpora.train, csv_path=paths.train_csv, wdid_mapping=wdid_mapping)
    attach_entities_to_corpus(corpus=corpora.test, csv_path=paths.test_csv, wdid_mapping=wdid_mapping)
    return corpora


@PipelineDecorator.component(
    return_values=['articleEmbeddingStats'],
    execution_queue='iptc_entity_tasks',
    task_type=TaskTypes.data_processing,
)
def prepare_article_embeddings(
    corpora,
    paths_config: Mapping[str, Any],
    embedding_config: Mapping[str, Any],
):
    """Precompute and cache missing article embeddings for all corpora."""
    from dataclasses import asdict

    from iptc_entity_pipeline.config import EmbeddingConfig, PathsConfig, config_from_dict

    logger = logging.getLogger(__name__)
    paths = config_from_dict(PathsConfig, paths_config)
    emb = config_from_dict(EmbeddingConfig, embedding_config)
    article_provider = ArticleEmbeddingProvider(
        embeddings_dir=paths.article_embeddings_dir,
        model_name=emb.article_model_name,
        backend=emb.article_embedding_backend,
        embed_svc_url=emb.embed_svc_url,
        embedding_dim=emb.article_embedding_dim,
    )

    train_stats = article_provider.recompute_embeddings(corpus=corpora.train)
    test_stats = article_provider.recompute_embeddings(corpus=corpora.test)
    total_docs = train_stats.total_docs + test_stats.total_docs
    total_computed = train_stats.computed_docs + test_stats.computed_docs
    logger.info(
        'Article embedding preparation progress: %s/%s articles (computed_missing=%s, cached_or_present=%s)',
        total_docs,
        total_docs,
        total_computed,
        total_docs - total_computed,
    )
    return {
        'train': asdict(train_stats),
        'test': asdict(test_stats),
        'total_docs': total_docs,
        'total_computed': total_computed,
        'total_cached_or_present': total_docs - total_computed,
    }


@PipelineDecorator.component(
    return_values=['datasetBundle'],
    execution_queue='iptc_entity_tasks',
    task_type=TaskTypes.data_processing,
)
def link_embeddings_and_build_datasets(
    corpora,
    paths_config: Mapping[str, Any],
    embedding_config: Mapping[str, Any],
):
    """Prepare entity coverage, link embeddings, and build EmbeddingDataset objects."""
    from iptc_entity_pipeline.clearml_pipeline import DatasetBundle, EntityEmbeddingStats
    from iptc_entity_pipeline.config import EmbeddingConfig, PathsConfig, config_from_dict

    logger = logging.getLogger(__name__)
    logger.info('Initializing providers for merged entity preparation + linking step')
    paths = config_from_dict(PathsConfig, paths_config)
    emb = config_from_dict(EmbeddingConfig, embedding_config)
    article_provider = ArticleEmbeddingProvider(
        embeddings_dir=paths.article_embeddings_dir,
        model_name=emb.article_model_name,
        backend=emb.article_embedding_backend,
        embed_svc_url=emb.embed_svc_url,
        embedding_dim=emb.article_embedding_dim,
    )

    if not emb.use_entity_embeddings:
        def build_article_only_matrix(*, split_corpus, split_name: str) -> np.ndarray:
            rows: list[np.ndarray] = []
            total_docs = len(split_corpus)
            logger.info('Building article-only features for %s corpus (%s articles)', split_name, total_docs)
            for idx, doc in enumerate(split_corpus, start=1):
                article_embedding = article_provider.get_embedding(
                    article_id=doc.id,
                    article_text=get_article_text(doc),
                    article_doc=doc,
                )
                rows.append(np.asarray(article_embedding, dtype=np.float32))
                if idx % 1000 == 0 or idx == total_docs:
                    logger.info(
                        'Built article-only features for %s/%s articles in %s corpus',
                        idx,
                        total_docs,
                        split_name,
                    )
            return np.vstack(rows)

        logger.info('Entity embeddings disabled by config; running article-only feature pipeline')
        x_train = build_article_only_matrix(split_corpus=corpora.train, split_name='train')
        x_test = build_article_only_matrix(split_corpus=corpora.test, split_name='test')
        train_data = build_embedding_dataset(corpus=corpora.train, x_matrix=x_train)
        test_data = build_embedding_dataset(corpus=corpora.test, x_matrix=x_test)
        feature_dim = int(x_train.shape[1])
        entity_embedding_stats = EntityEmbeddingStats(use_entity_embeddings=False)
        return DatasetBundle(
            train_data=train_data,
            test_data=test_data,
            feature_dim=feature_dim,
            entity_embedding_stats=entity_embedding_stats,
        )

    entity_store = EntityEmbeddingStore(
        root_dir=paths.entity_embeddings_dir,
        lang=emb.entity_lang,
    )
    pooling = SumEntityPooling()
    builder = FeatureBuilder(
        article_embedding_provider=article_provider,
        entity_embedding_store=entity_store,
        pooling_strategy=pooling,
        combine_method=emb.combine_method,
    )

    unique_wdids: set[str] = set()
    for corpus in [corpora.train, corpora.test]:
        for doc in corpus:
            unique_wdids.update(get_doc_wdids(doc))

    found_cnt = 0
    missing_cnt = 0
    total_wdids = len(unique_wdids)
    logger.info('Collected %s unique linked entities for coverage check', total_wdids)
    entity_progress_interval = max(1, total_wdids // 20) if total_wdids else 1
    for idx, wdid in enumerate(sorted(unique_wdids), start=1):
        if entity_store.get_entity_embedding(wdid=wdid) is None:
            missing_cnt += 1
        else:
            found_cnt += 1
        if idx % entity_progress_interval == 0 or idx == total_wdids:
            logger.info('Prepared entity embeddings for %s/%s linked entities', idx, total_wdids)

    entity_dim = int(entity_store.infer_embedding_dim())
    entity_embedding_stats = EntityEmbeddingStats(
        entity_dim=entity_dim,
        linked_unique_wdids=int(total_wdids),
        found_embeddings=int(found_cnt),
        missing_embeddings=int(missing_cnt),
    )

    logger.info('Building linked features for train corpus (%s articles)', len(corpora.train))
    x_train = builder.build_features_for_corpus(
        corpus=corpora.train,
        ensure_article_embeddings=False,
    )
    logger.info('Building linked features for test corpus (%s articles)', len(corpora.test))
    x_test = builder.build_features_for_corpus(
        corpus=corpora.test,
        ensure_article_embeddings=False,
    )

    train_data = build_embedding_dataset(corpus=corpora.train, x_matrix=x_train)
    test_data = build_embedding_dataset(corpus=corpora.test, x_matrix=x_test)
    feature_dim = int(x_train.shape[1])
    actual_entity_dim = int(entity_store.infer_embedding_dim())
    if entity_dim != actual_entity_dim:
        raise ValueError(f'Entity embedding dimension mismatch: expected={entity_dim}, actual={actual_entity_dim}')
    return DatasetBundle(
        train_data=train_data,
        test_data=test_data,
        feature_dim=feature_dim,
        entity_embedding_stats=entity_embedding_stats,
    )


@PipelineDecorator.component(
    return_values=['cvResult'],
    execution_queue='iptc_entity_tasks',
    task_type=TaskTypes.training,
)
def run_cv(
    trainData,
    featureDim: int,
    hyperparam_space_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    evaluation_config: Mapping[str, Any],
    cv_config: Mapping[str, Any],
    objective_corpora: str,
    print_logs: bool = True,
    debug: bool = False,
):
    """Run mandatory CV over train and select best hyperparameter combination."""
    from dataclasses import asdict

    import pandas as pd
    from iptc_entity_pipeline.clearml_pipeline import CvResult, FoldScores
    from iptc_entity_pipeline.config import (
        CvConfig,
        EvaluationConfig,
        HyperparamSpace,
        ModelConfig,
        TrainingConfig,
        config_from_dict,
    )
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

    task = Task.current_task()
    logger = task.get_logger()

    space = config_from_dict(HyperparamSpace, hyperparam_space_config)
    base_training = config_from_dict(TrainingConfig, training_config)
    eval_cfg = config_from_dict(EvaluationConfig, evaluation_config)
    cv_cfg = config_from_dict(CvConfig, cv_config)

    x_full = to_numpy_array(matrix_like=trainData.X)
    y_full = (
        to_numpy_array(matrix_like=trainData.Y)
        if hasattr(trainData, 'Y')
        else build_multilabel_targets(corpus=trainData.corpus)
    )

    combinations = space.iter_combinations(base_training=base_training)
    fold_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    best_trial: dict[str, Any] | None = None
    best_combo: tuple[ModelConfig, TrainingConfig] | None = None
    best_fold_curves: tuple[CvFoldCurves, ...] = ()

    for combo_idx, (combo_model_cfg, combo_train_cfg) in enumerate(combinations, start=1):
        params_json = combo_params_json(
            model_config=combo_model_cfg,
            training_config=combo_train_cfg,
        )
        cv_splitter = MultilabelStratifiedKFold(
            n_splits=cv_cfg.folds,
            shuffle=True,
            random_state=cv_cfg.random_seed,
        )
        fold_scores = FoldScores()
        combo_fold_curves: list[CvFoldCurves] = []
        for fold_idx, (fit_indices, val_indices) in enumerate(cv_splitter.split(x_full, y_full), start=1):
            fit_data = slice_dataset(dataset=trainData, indices=fit_indices.tolist())
            val_data = slice_dataset(dataset=trainData, indices=val_indices.tolist())
            train_result = train_model(
                train_data=fit_data,
                dev_data=val_data,
                feature_dim=featureDim,
                model_config=combo_model_cfg,
                training_config=combo_train_cfg,
                print_logs=print_logs,
            )
            model = train_result.model
            dev_loss = train_result.final_dev_loss
            df_corpora_fold, _ = evaluateModel(
                model=model,
                evalData=val_data,
                evaluation_config=eval_cfg,
                customThresholds=None,
            )
            micro_row = (
                df_corpora_fold.loc['All-micro'].to_dict()
                if 'All-micro' in df_corpora_fold.index
                else get_obj_row(
                    df_corpora=df_corpora_fold,
                    objective_corpora=objective_corpora,
                    averaging_type=eval_cfg.averaging_type,
                )
            )
            fold_scores.loss.append(dev_loss)
            fold_scores.epochs.append(float(train_result.epochs_run))
            fold_scores.precision.append(float(micro_row['Precision']))
            fold_scores.recall.append(float(micro_row['Recall']))
            fold_scores.f1.append(float(micro_row['F1']))
            combo_fold_curves.append(
                CvFoldCurves(
                    fold_id=fold_idx,
                    train_loss_per_epoch=train_result.train_loss_per_epoch,
                    dev_loss_per_epoch=train_result.dev_loss_per_epoch,
                    train_f1_per_epoch=train_result.train_f1_per_epoch,
                    dev_f1_per_epoch=train_result.dev_f1_per_epoch,
                )
            )
            fold_rows.append(
                {
                    'trial_id': combo_idx,
                    'fold_id': fold_idx,
                    'params': params_json,
                    'epochs': float(train_result.epochs_run),
                    'Loss': dev_loss,
                    'Precision': float(micro_row['Precision']),
                    'Recall': float(micro_row['Recall']),
                    'F1': float(micro_row['F1']),
                }
            )
            if debug:
                break

        trial_row = {
            'trial_id': combo_idx,
            'params': params_json,
            'epochs': float(np.mean(fold_scores.epochs)),
            'Loss_mean': float(np.mean(fold_scores.loss)),
            'Loss_std': float(np.std(fold_scores.loss)),
            'F1_mean': float(np.mean(fold_scores.f1)),
            'F1_std': float(np.std(fold_scores.f1)),
            'Precision_mean': float(np.mean(fold_scores.precision)),
            'Precision_std': float(np.std(fold_scores.precision)),
            'Recall_mean': float(np.mean(fold_scores.recall)),
            'Recall_std': float(np.std(fold_scores.recall)),
        }
        trial_rows.append(trial_row)
        if best_trial is None or trial_row['F1_mean'] > best_trial['F1_mean']:
            best_trial = trial_row
            best_combo = (combo_model_cfg, combo_train_cfg)
            best_fold_curves = tuple(combo_fold_curves)

    if best_trial is None or best_combo is None:
        raise ValueError('No CV trial results were produced.')
    trials_df = pd.DataFrame(trial_rows).sort_values(by='F1_mean', ascending=False).reset_index(drop=True)
    folds_df = pd.DataFrame(fold_rows)
    cv_dev_df = pd.DataFrame(
        [
            {
                'params': best_trial['params'],
                'epochs': best_trial['epochs'],
                'Precision': best_trial['Precision_mean'],
                'Recall': best_trial['Recall_mean'],
                'F1': best_trial['F1_mean'],
                'Loss': best_trial['Loss_mean'],
                'Precision_std': best_trial['Precision_std'],
                'Recall_std': best_trial['Recall_std'],
                'F1_std': best_trial['F1_std'],
                'Loss_std': best_trial['Loss_std'],
            }
        ],
    )

    report_cv_outputs(task=task, logger=logger, trials_df=trials_df, folds_df=folds_df, cv_dev_df=cv_dev_df)
    report_cv_fold_curve_charts(logger=logger, fold_curves=best_fold_curves)

    best_model_cfg, best_train_cfg = best_combo
    objective_metrics = {
        'Loss_mean': float(best_trial['Loss_mean']),
        'Loss_std': float(best_trial['Loss_std']),
        'Precision_mean': float(best_trial['Precision_mean']),
        'Precision_std': float(best_trial['Precision_std']),
        'Recall_mean': float(best_trial['Recall_mean']),
        'Recall_std': float(best_trial['Recall_std']),
        'F1_mean': float(best_trial['F1_mean']),
        'F1_std': float(best_trial['F1_std']),
        'epochs': float(best_trial['epochs']),
    }
    return CvResult(
        cv_dev_df=cv_dev_df,
        best_model_config=asdict(best_model_cfg),
        best_training_config=asdict(best_train_cfg),
        objective_metrics=objective_metrics,
    )

@PipelineDecorator.component(
    return_values=['trainedModel'],
    execution_queue='iptc_entity_tasks',
    task_type=TaskTypes.training,
)
def train_best(
    train_data,
    test_data,
    feature_dim: int,
    best_model_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    print_logs: bool = True,
):
    """Train final model on full train set with best hyperparams from CV."""
    from iptc_entity_pipeline.config import ModelConfig, TrainingConfig, config_from_dict

    model_cfg = config_from_dict(ModelConfig, best_model_config)
    train_cfg = config_from_dict(TrainingConfig, training_config)
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
    report_train_test_curve_charts(logger=logger, result=result)
    return result.model

@PipelineDecorator.component(
    return_values=['evalResult'],
    execution_queue='iptc_entity_tasks',
    task_type=TaskTypes.testing,
)
def eval_final(
    trainedModel,
    cvDevDf,
    testData,
    evaluation_config: Mapping[str, Any],
    embedding_config: Mapping[str, Any],
    objective_corpora: str,
    config_name: str,
    config_mapping: Mapping[str, Any],
    feature_dim: int,
):
    """Evaluate final model on test and persist CV dev summary with mean/std."""
    from dataclasses import asdict

    import pandas as pd
    from geneea.catlib.model.model import filterLabels
    from iptc_entity_pipeline.clearml_pipeline import EvalResult
    from iptc_entity_pipeline.config import EmbeddingConfig, EvaluationConfig, config_from_dict

    task = Task.current_task()
    logger = task.get_logger()
    eval_cfg = config_from_dict(EvaluationConfig, evaluation_config)
    emb_cfg = config_from_dict(EmbeddingConfig, embedding_config)
    df_corpora_test, df_classes_test, pred_scores = evaluateModel(
        model=trainedModel,
        evalData=testData,
        evaluation_config=eval_cfg,
        customThresholds=None,
        returnPredictions=True,
    )

    objective_row_name = f'All-{eval_cfg.averaging_type}'
    if objective_corpora in cvDevDf.index:
        row_cv = cvDevDf.loc[objective_corpora].to_dict()
    else:
        row_cv = cvDevDf.iloc[0].to_dict()
    report_eval_scalars(logger=logger, title='Dev Cross Validation Mean Results', row=row_cv, iteration=0)
    if 'Precision_std' in row_cv:
        report_cv_std_scalars(
            logger=logger,
            row=row_cv,
            title='Dev Cross Validation Mean Results',
            iteration=0,
        )
    row_all_test = df_corpora_test.loc[objective_row_name].to_dict()
    report_eval_scalars(logger=logger, title='Test Evaluation Results', row=row_all_test, iteration=0)
    if 'All-micro' in df_corpora_test.index:
        row_micro_test = df_corpora_test.loc['All-micro'].to_dict()
        report_eval_scalars(logger=logger, title='Test Evaluation Results (micro)', row=row_micro_test, iteration=0)
    if objective_corpora in df_corpora_test.index:
        row_objective_test = df_corpora_test.loc[objective_corpora].to_dict()
        report_eval_scalars(logger=logger, title='Objective Test Evaluation Results', row=row_objective_test, iteration=0)
    else:
        logging.getLogger(__name__).warning(
            'Requested objective_corpora "%s" not found in test corpus index; available=%s',
            objective_corpora,
            list(df_corpora_test.index),
        )

    def build_predictions_dataframe(*, dataset) -> pd.DataFrame:
        pred_labels = [
            filterLabels(dc, thr=eval_cfg.threshold_eval, thrByLabel=None, keepWgh=False)
            for dc in pred_scores
        ]
        rows = []
        for doc, pred in zip(dataset.corpus, pred_labels):
            rows.append(
                {
                    'article_id': doc.id,
                    'corpus_name': doc.metadata.get('corpusName', ''),
                    'predicted_categories': '|'.join(sorted(pred)),
                    'gold_categories': '|'.join(sorted(doc.cats)),
                }
            )
        return pd.DataFrame(rows)

    test_predictions_df = build_predictions_dataframe(dataset=testData)

    logger.report_table(title='Cross Validation Summary', series='Mean+Std', iteration=0, table_plot=cvDevDf)
    logger.report_table(title='Test Evaluation', series='Corpora Dataframe', iteration=0, table_plot=df_corpora_test)
    logger.report_table(title='Test Evaluation', series='Classes Dataframe', iteration=0, table_plot=df_classes_test)

    sanitized_config_name = ''.join(ch if ch.isalnum() or ch in ['-', '_'] else '_' for ch in config_name)
    results_dir = Path('results')
    results_dir.mkdir(parents=True, exist_ok=True)
    excel_path = results_dir / f'final_evaluation_tables_{sanitized_config_name}.xlsx'
    save_paths = save_final_model_outputs(
        model=trainedModel,
        test_data=testData,
        pred_scores=pred_scores,
        evaluation_config=eval_cfg,
        embedding_config=emb_cfg,
        config_mapping=config_mapping,
        config_name=config_name,
        feature_dim=feature_dim,
    )

    task.upload_artifact('dev_corpora_dataframe', artifact_object=cvDevDf)
    task.upload_artifact('test_corpora_dataframe', artifact_object=df_corpora_test)
    task.upload_artifact('test_classes_dataframe', artifact_object=df_classes_test)
    task.upload_artifact('final_evaluation_tables_xlsx', artifact_object=str(excel_path))
    task.upload_artifact('test_article_predictions', artifact_object=test_predictions_df)
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
    base_probabilities_csv = str(
        comparison_cfg.get('base_probabilities_csv', '') or eval_cfg.base_probabilities_csv
    ).strip()
    comparison_result = None
    if base_probabilities_csv:
        comparison_result = compare_runs(
            current_probabilities=save_paths.probabilities_csv_path,
            base_probabilities=base_probabilities_csv,
            gold_data=testData,
            threshold_eval=eval_cfg.threshold_eval,
            averaging_type=eval_cfg.averaging_type,
            top_n=int(comparison_cfg.get('top_n', 20)),
            only_diff=bool(comparison_cfg.get('only_diff', False)),
            output_path=build_output_path(
                output_root=str(comparison_cfg.get('output_root', 'results/comparisons')),
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
        task.upload_artifact('evaluation_comparison_summary', artifact_object=comparison_result.summary_comparison)
        task.upload_artifact('evaluation_comparison_classes', artifact_object=comparison_result.classes_comparison)
        task.upload_artifact('evaluation_comparison_corpora', artifact_object=comparison_result.corpora_comparison)
        if comparison_result.excel_path is not None:
            task.upload_artifact('evaluation_comparison_xlsx', artifact_object=str(comparison_result.excel_path))

    with pd.ExcelWriter(excel_path) as writer:
        cvDevDf.to_excel(excel_writer=writer, sheet_name='dev_cv_summary')
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
        dev_corpora_df=cvDevDf,
        test_corpora_df=df_corpora_test,
        test_classes_df=df_classes_test,
        objective_metrics=objective_metrics,
    )


@PipelineDecorator.pipeline(
    name='iptc-entity-enhanced-v1',
    project='iptc/EntityEnhanced',
    version='0.1',
    pipeline_execution_queue='iptc_entity_tasks',
)
def run_training_pipeline(config_mapping: Mapping[str, Any]) -> None:
    """Execute full v1 training and evaluation pipeline."""
    task = Task.current_task()
    task.connect(config_mapping, name='pipelineConfig')
    config_name = str(config_mapping.get('config_name', 'wpentities'))
    task.add_tags([config_name])

    paths_config = config_mapping['paths']
    embedding_config = config_mapping['embeddings']
    training_config = config_mapping['training']
    evaluation_config = config_mapping['evaluation']
    cv_config = config_mapping.get('cv', {'folds': 5, 'random_seed': 43})
    hyperparam_space_config = config_mapping['hyperparam_space']
    objective_corpora = config_mapping['objective_corpora']
    downsample_corpora = config_mapping.get('downsample_corpora', {})
    use_entity_embeddings = bool(embedding_config.get('use_entity_embeddings', True))
    print_logs = bool(config_mapping.get('print_logs', True))
    debug = bool(config_mapping.get('debug', False))

    log_stage(
        task=task,
        message='Stage 1/6: Loading corpora and article-to-entity mapping',
        print_logs=print_logs,
    )
    corpora = load_data(
        paths_config=paths_config,
        downsample_corpora=downsample_corpora,
    )
    log_stage(
        task=task,
        message='Stage 2/6: Preparing article embeddings (train/test)',
        print_logs=print_logs,
    )
    article_embedding_stats = prepare_article_embeddings(
        corpora=corpora,
        paths_config=paths_config,
        embedding_config=embedding_config,
    )
    task.upload_artifact('article_embedding_stats', artifact_object=dict(article_embedding_stats))

    log_stage(
        task=task,
        message=(
            'Stage 3/6: Preparing entity embeddings, linking, and building datasets'
            if use_entity_embeddings
            else 'Stage 3/6: Building article-only datasets (entity embeddings disabled)'
        ),
        print_logs=print_logs,
    )
    dataset_bundle = link_embeddings_and_build_datasets(
        corpora=corpora,
        paths_config=paths_config,
        embedding_config=embedding_config,
    )
    task.upload_artifact('entity_embedding_stats', artifact_object=asdict(dataset_bundle.entity_embedding_stats))

    log_stage(
        task=task,
        message=f'Stage 4/6: Running mandatory {cv_config.get("folds", 5)}-fold cross-validation on train',
        print_logs=print_logs,
    )
    cv_result = run_cv(
        trainData=dataset_bundle.train_data,
        featureDim=dataset_bundle.feature_dim,
        hyperparam_space_config=hyperparam_space_config,
        training_config=training_config,
        evaluation_config=evaluation_config,
        cv_config=cv_config,
        objective_corpora=objective_corpora,
        print_logs=print_logs,
        debug=debug,
    )
    task.upload_artifact('cv_objective_metrics', artifact_object=dict(cv_result.objective_metrics))

    log_stage(
        task=task,
        message='Stage 5/6: Training final model on full train (validation=test)',
        print_logs=print_logs,
    )
    trained_model = train_best(
        train_data=dataset_bundle.train_data,
        test_data=dataset_bundle.test_data,
        feature_dim=dataset_bundle.feature_dim,
        best_model_config=cv_result.best_model_config,
        training_config=cv_result.best_training_config,
        print_logs=print_logs,
    )

    log_stage(
        task=task,
        message='Stage 6/6: Evaluating final model on test',
        print_logs=print_logs,
    )
    eval_result = eval_final(
        trainedModel=trained_model,
        cvDevDf=cv_result.cv_dev_df,
        testData=dataset_bundle.test_data,
        evaluation_config=evaluation_config,
        embedding_config=embedding_config,
        objective_corpora=objective_corpora,
        config_name=config_name,
        config_mapping=config_mapping,
        feature_dim=dataset_bundle.feature_dim,
    )

    logger = task.get_logger()
    objective_row_name = f'All-{evaluation_config["averaging_type"]}'
    if objective_row_name in eval_result.dev_corpora_df.index:
        row_cv = eval_result.dev_corpora_df.loc[objective_row_name].to_dict()
    else:
        row_cv = eval_result.dev_corpora_df.iloc[0].to_dict()

    report_eval_scalars(
        logger=logger,
        title='Cross Validation Results',
        row=row_cv,
        iteration=0,
    )
    if 'Precision_std' in row_cv:
        report_cv_std_scalars(
            logger=logger,
            row=row_cv,
            title='Cross Validation Results',
            iteration=0,
        )

    report_eval_scalars(
        logger=logger,
        title='Test Evaluation Results',
        row=eval_result.objective_metrics,
        iteration=0,
    )
    if 'All-micro' in eval_result.test_corpora_df.index:
        report_eval_scalars(
            logger=logger,
            title='Test Evaluation Results (micro)',
            row=eval_result.test_corpora_df.loc['All-micro'].to_dict(),
            iteration=0,
        )

    task.upload_artifact('pipeline_config', artifact_object=dict(config_mapping))
    task.upload_artifact('objective_metrics', artifact_object=dict(eval_result.objective_metrics))
    task.upload_artifact('dev_corpora_dataframe', artifact_object=eval_result.dev_corpora_df)
    task.upload_artifact('test_corpora_dataframe', artifact_object=eval_result.test_corpora_df)
    task.upload_artifact('test_classes_dataframe', artifact_object=eval_result.test_classes_df)
    log_stage(
        task=task,
        message='Pipeline finished: metrics, tables, and artifacts uploaded',
        print_logs=print_logs,
    )


def run_pipeline(
    config: Optional[BaseConfig] = None,
    config_name: str = 'wpentities',
    is_local: bool = False,
) -> Tuple[BaseConfig, Mapping[str, Any]]:
    """
    Resolve config and execute the ClearML pipeline locally.

    :param config: Optional custom pipeline config.
    :param config_name: Name of selected config variant.
    :param is_local: Run pipeline steps locally instead of dispatching to a queue.
    :return: Tuple of resolved config and mapping passed to ClearML.
    """
    pipeline_config = config or BaseConfig()
    resolved_config = resolve_paths(config=pipeline_config, root_dir=Path.cwd())
    config_mapping = asdict(resolved_config)
    config_mapping['config_name'] = config_name
    if is_local:
        PipelineDecorator.run_locally()
    run_training_pipeline(config_mapping=config_mapping)
    return resolved_config, config_mapping
