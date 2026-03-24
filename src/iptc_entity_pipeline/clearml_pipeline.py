"""ClearML pipeline orchestration for entity-enhanced IPTC training."""

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import numpy as np
from clearml import Task, TaskTypes
from clearml.automation.controller import PipelineDecorator

from iptc_entity_pipeline.article_embeddings import ArticleEmbeddingProvider
from iptc_entity_pipeline.config import BaseConfig, resolve_paths
from iptc_entity_pipeline.data_loading import get_article_text, load_and_normalize_corpora, load_article_wdids
from iptc_entity_pipeline.dataset_builder import build_embedding_dataset
from iptc_entity_pipeline.entity_embeddings import EntityEmbeddingStore
from iptc_entity_pipeline.feature_builder import FeatureBuilder
from iptc_entity_pipeline.legacy_reuse import (
    createClassificationModel,
    evaluateModel,
    trainClassificationModel,
)
from iptc_entity_pipeline.pooling import SumEntityPooling

LOGGER = logging.getLogger(__name__)


def _report_eval_scalars(*, logger: Any, title: str, row: Mapping[str, Any], iteration: int = 0) -> None:
    logger.report_scalar(title=title, series='Precision', value=row['Precision'], iteration=iteration)
    logger.report_scalar(title=title, series='Recall', value=row['Recall'], iteration=iteration)
    logger.report_scalar(title=title, series='F04', value=row['F04'], iteration=iteration)
    logger.report_scalar(title=title, series='F1', value=row['F1'], iteration=iteration)


def _report_stage_message(*, task: Task, message: str, logging_config: Mapping[str, Any]) -> None:
    LOGGER.info(message)
    task.get_logger().report_text(message, print_console=bool(logging_config['print_logs']))


@PipelineDecorator.component(
    return_values=['corpora', 'articleToWdids'],
    execution_queue='iptc_entity_tasks',
    task_type=TaskTypes.data_processing,
)
def load_data(
    paths_config: Mapping[str, Any],
    downsample_corpora: Mapping[str, float] | None = None,
):
    """Load normalized corpora and article-to-wdIds mapping."""
    corpora = load_and_normalize_corpora(
        train_csv=paths_config['train_csv'],
        dev_csv=paths_config['dev_csv'],
        test_csv=paths_config['test_csv'],
        removed_cat_ids=paths_config['removed_cat_ids'],
        downsample_corpora=downsample_corpora or {},
        downsampling_order_cache_json=paths_config.get('downsampling_order_cache_json'),
    )
    article_to_wdids = load_article_wdids(article_entities_tsv=paths_config['article_entities_tsv'])
    return corpora, article_to_wdids


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
    logger = logging.getLogger(__name__)
    article_provider = ArticleEmbeddingProvider(
        embeddings_dir=paths_config['article_embeddings_dir'],
        model_name=embedding_config['article_model_name'],
        backend=embedding_config['article_embedding_backend'],
        embed_svc_url=embedding_config['embed_svc_url'],
        embedding_dim=embedding_config['article_embedding_dim'],
    )

    # Keep explicit per-corpus calls for visibility in logs and progress reporting.
    train_stats = article_provider.recompute_embeddings(corpus=corpora.train)
    dev_stats = article_provider.recompute_embeddings(corpus=corpora.dev)
    test_stats = article_provider.recompute_embeddings(corpus=corpora.test)
    total_docs = int(train_stats['total_docs'] + dev_stats['total_docs'] + test_stats['total_docs'])
    total_computed = int(
        train_stats['computed_docs'] + dev_stats['computed_docs'] + test_stats['computed_docs']
    )
    logger.info(
        'Article embedding preparation progress: %s/%s articles (computed_missing=%s, cached_or_present=%s)',
        total_docs,
        total_docs,
        total_computed,
        total_docs - total_computed,
    )
    return {
        'train': dict(train_stats),
        'dev': dict(dev_stats),
        'test': dict(test_stats),
        'total_docs': total_docs,
        'total_computed': total_computed,
        'total_cached_or_present': int(total_docs - total_computed),
    }


@PipelineDecorator.component(
    return_values=['trainData', 'devData', 'testData', 'featureDim', 'entityEmbeddingStats'],
    execution_queue='iptc_entity_tasks',
    task_type=TaskTypes.data_processing,
)
def link_embeddings_and_build_datasets(
    corpora,
    articleToWdids,
    paths_config: Mapping[str, Any],
    embedding_config: Mapping[str, Any],
):
    """Prepare entity coverage, link embeddings, and build EmbeddingDataset objects."""
    logger = logging.getLogger(__name__)
    logger.info('Initializing providers for merged entity preparation + linking step')
    article_provider = ArticleEmbeddingProvider(
        embeddings_dir=paths_config['article_embeddings_dir'],
        model_name=embedding_config['article_model_name'],
        backend=embedding_config['article_embedding_backend'],
        embed_svc_url=embedding_config['embed_svc_url'],
        embedding_dim=embedding_config['article_embedding_dim'],
    )
    entity_store = EntityEmbeddingStore(
        root_dir=paths_config['entity_embeddings_dir'],
        lang=embedding_config['entity_lang'],
    )
    pooling = SumEntityPooling()
    builder = FeatureBuilder(
        article_embedding_provider=article_provider,
        entity_embedding_store=entity_store,
        pooling_strategy=pooling,
        combine_method=embedding_config['combine_method'],
    )
    use_entity_embeddings = bool(embedding_config.get('use_entity_embeddings', True))

    if not use_entity_embeddings:
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
        x_dev = build_article_only_matrix(split_corpus=corpora.dev, split_name='dev')
        x_test = build_article_only_matrix(split_corpus=corpora.test, split_name='test')
        train_data = build_embedding_dataset(corpus=corpora.train, x_matrix=x_train)
        dev_data = build_embedding_dataset(corpus=corpora.dev, x_matrix=x_dev)
        test_data = build_embedding_dataset(corpus=corpora.test, x_matrix=x_test)
        feature_dim = int(x_train.shape[1])
        entity_embedding_stats = {
            'use_entity_embeddings': False,
            'entity_dim': 0,
            'linked_unique_wdids': 0,
            'found_embeddings': 0,
            'missing_embeddings': 0,
        }
        return train_data, dev_data, test_data, feature_dim, entity_embedding_stats

    unique_wdids: set[str] = set()
    for corpus in [corpora.train, corpora.dev, corpora.test]:
        for doc in corpus:
            unique_wdids.update(articleToWdids.get(doc.id, []))

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
    entity_embedding_stats = {
        'entity_dim': entity_dim,
        'linked_unique_wdids': int(total_wdids),
        'found_embeddings': int(found_cnt),
        'missing_embeddings': int(missing_cnt),
    }

    logger.info('Building linked features for train corpus (%s articles)', len(corpora.train))
    x_train = builder.build_features_for_corpus(
        corpus=corpora.train,
        article_to_wdids=articleToWdids,
        ensure_article_embeddings=False,
    )
    logger.info('Building linked features for dev corpus (%s articles)', len(corpora.dev))
    x_dev = builder.build_features_for_corpus(
        corpus=corpora.dev,
        article_to_wdids=articleToWdids,
        ensure_article_embeddings=False,
    )
    logger.info('Building linked features for test corpus (%s articles)', len(corpora.test))
    x_test = builder.build_features_for_corpus(
        corpus=corpora.test,
        article_to_wdids=articleToWdids,
        ensure_article_embeddings=False,
    )

    train_data = build_embedding_dataset(corpus=corpora.train, x_matrix=x_train)
    dev_data = build_embedding_dataset(corpus=corpora.dev, x_matrix=x_dev)
    test_data = build_embedding_dataset(corpus=corpora.test, x_matrix=x_test)
    feature_dim = int(x_train.shape[1])
    actual_entity_dim = int(entity_store.infer_embedding_dim())
    if entity_dim != actual_entity_dim:
        raise ValueError(f'Entity embedding dimension mismatch: expected={entity_dim}, actual={actual_entity_dim}')
    return train_data, dev_data, test_data, feature_dim, entity_embedding_stats


@PipelineDecorator.component(
    return_values=['trainedModel'],
    execution_queue='iptc_entity_tasks',
    task_type=TaskTypes.training,
)
def train_classification_model(
    trainData,
    devData,
    featureDim: int,
    model_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    logging_config: Mapping[str, Any],
):
    """Create and train classification model as a dedicated pipeline step."""
    out_dim = int(trainData.corpus.catCnt)
    model = createClassificationModel(
        embDim=int(featureDim),
        outDim=out_dim,
        modelConfig={
            'hiddenDim': int(model_config['hidden_dim']),
            'dropouts1': float(model_config['dropouts1']),
            'dropouts2': float(model_config['dropouts2']),
        },
        textVectorizer='not None',
        logConfig={'PRINT_LOGS': bool(logging_config['print_logs'])},
    )
    return trainClassificationModel(
        model=model,
        trainData=trainData,
        devData=devData,
        trainingConfig={
            'epochs': int(training_config['epochs']),
            'batchSize': int(training_config['batch_size']),
            'earlyStoppingPatience': int(training_config.get('early_stopping_patience', 0)),
            'earlyStoppingMinDelta': float(training_config.get('early_stopping_min_delta', 0.0)),
            'optimizerConfig': {
                'name': training_config['optimizer_name'],
                'adamConfig': {'lr': float(training_config['learning_rate'])},
            },
            'lrSchedulerConfig': {
                'name': training_config['lr_scheduler_name'],
                'stepLRConfig': {
                    'stepSize': int(training_config['step_size']),
                    'gamma': float(training_config['gamma']),
                },
                'cosineAnnealingLRConfig': {'T_max': max(int(training_config['epochs']), 1)},
            },
            'lossConfig': {
                'name': training_config['loss_name'],
                'focalLossConfig': {'alpha': 0.25, 'gamma': 2.0},
            },
        },
        logConfig={'PRINT_LOGS': bool(logging_config['print_logs'])},
    )


@PipelineDecorator.component(
    return_values=['devCorporaDf', 'testCorporaDf', 'testClassesDf', 'objectiveMetrics'],
    execution_queue='iptc_entity_tasks',
    task_type=TaskTypes.testing,
)
def evaluate_classification_model(
    trainedModel,
    devData,
    testData,
    evaluation_config: Mapping[str, Any],
    objective_corpora: str,
    config_name: str,
):
    """Evaluate trained model and save per-article predictions plus metric artifacts."""
    import pandas as pd
    from geneea.catlib.model.model import filterLabels

    task = Task.current_task()
    logger = task.get_logger()
    eval_cfg = {
        'thresholdPredict': float(evaluation_config['threshold_predict']),
        'thresholdEval': float(evaluation_config['threshold_eval']),
        'perCorpus': bool(evaluation_config['per_corpus']),
        'perClass': bool(evaluation_config['per_class']),
        'averagingType': str(evaluation_config['averaging_type']),
    }

    df_corpora_dev, df_classes_dev = evaluateModel(
        model=trainedModel,
        evalData=devData,
        evaluationConfig=eval_cfg,
        customThresholds=None,
    )
    df_corpora_test, df_classes_test = evaluateModel(
        model=trainedModel,
        evalData=testData,
        evaluationConfig=eval_cfg,
        customThresholds=None,
    )

    def report_eval_scalars(*, title: str, row: Mapping[str, Any]) -> None:
        logger.report_scalar(title=title, series='Precision', value=row['Precision'], iteration=0)
        logger.report_scalar(title=title, series='Recall', value=row['Recall'], iteration=0)
        logger.report_scalar(title=title, series='F04', value=row['F04'], iteration=0)
        logger.report_scalar(title=title, series='F1', value=row['F1'], iteration=0)

    objective_row_name = f'All-{eval_cfg["averagingType"]}'
    row_all = df_corpora_dev.loc[objective_row_name].to_dict()
    report_eval_scalars(title='Dev Evaluation Results', row=row_all)
    row_all_test = df_corpora_test.loc[objective_row_name].to_dict()
    report_eval_scalars(title='Test Evaluation Results', row=row_all_test)
    if objective_corpora in df_corpora_dev.index:
        row_objective = df_corpora_dev.loc[objective_corpora].to_dict()
        report_eval_scalars(title='Objective Evaluation Results', row=row_objective)
    else:
        logging.getLogger(__name__).warning(
            'Requested objective_corpora "%s" not found in dev corpus index; available=%s',
            objective_corpora,
            list(df_corpora_dev.index),
        )
    if objective_corpora in df_corpora_test.index:
        row_objective_test = df_corpora_test.loc[objective_corpora].to_dict()
        report_eval_scalars(title='Objective Test Evaluation Results', row=row_objective_test)
    else:
        logging.getLogger(__name__).warning(
            'Requested objective_corpora "%s" not found in test corpus index; available=%s',
            objective_corpora,
            list(df_corpora_test.index),
        )

    def build_predictions_dataframe(*, dataset) -> pd.DataFrame:
        pred_wgh = trainedModel.classifyDataset(
            dataset,
            thr=eval_cfg['thresholdPredict'],
            returnScores=True,
        )
        pred_labels = [filterLabels(dc, thr=eval_cfg['thresholdEval'], thrByLabel=None, keepWgh=False) for dc in pred_wgh]
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

    dev_predictions_df = build_predictions_dataframe(dataset=devData)
    test_predictions_df = build_predictions_dataframe(dataset=testData)

    logger.report_table(title='Dev Evaluation', series='Corpora Dataframe', iteration=0, table_plot=df_corpora_dev)
    logger.report_table(title='Dev Evaluation', series='Classes Dataframe', iteration=0, table_plot=df_classes_dev)
    logger.report_table(title='Test Evaluation', series='Corpora Dataframe', iteration=0, table_plot=df_corpora_test)
    logger.report_table(title='Test Evaluation', series='Classes Dataframe', iteration=0, table_plot=df_classes_test)

    sanitized_config_name = ''.join(ch if ch.isalnum() or ch in ['-', '_'] else '_' for ch in config_name)
    results_dir = Path('results')
    results_dir.mkdir(parents=True, exist_ok=True)
    excel_path = results_dir / f'final_evaluation_tables_{sanitized_config_name}.xlsx'
    with pd.ExcelWriter(excel_path) as writer:
        df_corpora_dev.to_excel(excel_writer=writer, sheet_name='dev_corpora')
        df_classes_dev.to_excel(excel_writer=writer, sheet_name='dev_classes')
        df_corpora_test.to_excel(excel_writer=writer, sheet_name='test_corpora')
        df_classes_test.to_excel(excel_writer=writer, sheet_name='test_classes')
    logger.report_text(f'Saved final evaluation tables to {excel_path}')

    task.upload_artifact('dev_corpora_dataframe', artifact_object=df_corpora_dev)
    task.upload_artifact('dev_classes_dataframe', artifact_object=df_classes_dev)
    task.upload_artifact('test_corpora_dataframe', artifact_object=df_corpora_test)
    task.upload_artifact('test_classes_dataframe', artifact_object=df_classes_test)
    task.upload_artifact('final_evaluation_tables_xlsx', artifact_object=str(excel_path))
    task.upload_artifact('dev_article_predictions', artifact_object=dev_predictions_df)
    task.upload_artifact('test_article_predictions', artifact_object=test_predictions_df)
    task.upload_artifact(
        'evaluation_thresholds',
        artifact_object={
            'threshold_predict': eval_cfg['thresholdPredict'],
            'threshold_eval': eval_cfg['thresholdEval'],
            'objective_corpora': objective_corpora,
            'objective_row_name': objective_row_name,
        },
    )
    # Compatibility aliases kept to match naming from baseline runs.
    task.upload_artifact('Corpora Dataframe', artifact_object=df_corpora_test)
    task.upload_artifact('Classes Dataframe', artifact_object=df_classes_test)

    objective_metrics = (
        df_corpora_dev.loc[objective_corpora].to_dict()
        if objective_corpora in df_corpora_dev.index
        else row_all
    )
    return df_corpora_dev, df_corpora_test, df_classes_test, objective_metrics


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
    config_name = str(config_mapping.get('config_name', 'base'))
    task.add_tags([config_name])

    paths_config = config_mapping['paths']
    embedding_config = config_mapping['embeddings']
    model_config = config_mapping['model']
    training_config = config_mapping['training']
    evaluation_config = config_mapping['evaluation']
    logging_config = config_mapping['logging']
    objective_corpora = config_mapping['objective_corpora']
    downsample_corpora = config_mapping.get('downsample_corpora', {})
    use_entity_embeddings = bool(embedding_config.get('use_entity_embeddings', True))

    _report_stage_message(
        task=task,
        message='Stage 1/5: Loading corpora and article-to-entity mapping',
        logging_config=logging_config,
    )
    corpora, article_to_wdids = load_data(
        paths_config=paths_config,
        downsample_corpora=downsample_corpora,
    )
    _report_stage_message(
        task=task,
        message='Stage 2/5: Preparing article embeddings (train/dev/test)',
        logging_config=logging_config,
    )
    article_embedding_stats = prepare_article_embeddings(
        corpora=corpora,
        paths_config=paths_config,
        embedding_config=embedding_config,
    )
    task.upload_artifact('article_embedding_stats', artifact_object=dict(article_embedding_stats))

    _report_stage_message(
        task=task,
        message=(
            'Stage 3/5: Preparing entity embeddings, linking, and building datasets'
            if use_entity_embeddings
            else 'Stage 3/5: Building article-only datasets (entity embeddings disabled)'
        ),
        logging_config=logging_config,
    )
    train_data, dev_data, test_data, feature_dim, entity_embedding_stats = link_embeddings_and_build_datasets(
        corpora=corpora,
        articleToWdids=article_to_wdids,
        paths_config=paths_config,
        embedding_config=embedding_config,
    )
    task.upload_artifact('entity_embedding_stats', artifact_object=dict(entity_embedding_stats))

    _report_stage_message(
        task=task,
        message=f'Stage 4/5: Training classification model (feature_dim={feature_dim})',
        logging_config=logging_config,
    )
    trained_model = train_classification_model(
        trainData=train_data,
        devData=dev_data,
        featureDim=feature_dim,
        model_config=model_config,
        training_config=training_config,
        logging_config=logging_config,
    )

    _report_stage_message(
        task=task,
        message='Stage 5/5: Evaluating model on dev and test',
        logging_config=logging_config,
    )
    df_corpora_dev, df_corpora_test, df_classes_test, objective_metrics = evaluate_classification_model(
        trainedModel=trained_model,
        devData=dev_data,
        testData=test_data,
        evaluation_config=evaluation_config,
        objective_corpora=objective_corpora,
        config_name=config_name,
    )

    logger = task.get_logger()
    objective_row_name = f'All-{evaluation_config["averaging_type"]}'
    if objective_row_name in df_corpora_dev.index:
        _report_eval_scalars(
            logger=logger,
            title='Dev Evaluation Results',
            row=df_corpora_dev.loc[objective_row_name].to_dict(),
            iteration=0,
        )
    if objective_row_name in df_corpora_test.index:
        _report_eval_scalars(
            logger=logger,
            title='Test Evaluation Results',
            row=df_corpora_test.loc[objective_row_name].to_dict(),
            iteration=0,
        )

    task.upload_artifact('pipeline_config', artifact_object=dict(config_mapping))
    task.upload_artifact('objective_metrics', artifact_object=dict(objective_metrics))
    task.upload_artifact('dev_corpora_dataframe', artifact_object=df_corpora_dev)
    task.upload_artifact('test_corpora_dataframe', artifact_object=df_corpora_test)
    task.upload_artifact('test_classes_dataframe', artifact_object=df_classes_test)
    _report_stage_message(
        task=task,
        message='Pipeline finished: metrics, tables, and artifacts uploaded',
        logging_config=logging_config,
    )


def run_local_pipeline(
    config: Optional[BaseConfig] = None,
    config_name: str = 'base',
) -> Tuple[BaseConfig, Mapping[str, Any]]:
    """
    Resolve config and execute the ClearML pipeline locally.

    :param config: Optional custom pipeline config.
    :param config_name: Name of selected config variant.
    :return: Tuple of resolved config and mapping passed to ClearML.
    """
    pipeline_config = config or BaseConfig()
    resolved_config = resolve_paths(config=pipeline_config, root_dir=Path.cwd())
    config_mapping = asdict(resolved_config)
    config_mapping['config_name'] = config_name
    run_training_pipeline(config_mapping=config_mapping)
    return resolved_config, config_mapping

