"""ClearML pipeline orchestration for entity-enhanced IPTC training."""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from clearml import Task, TaskTypes
from clearml.automation.controller import PipelineDecorator

from iptc_entity_pipeline.article_embeddings import ArticleEmbeddingProvider
from iptc_entity_pipeline.config import PipelineConfig, resolve_paths
from iptc_entity_pipeline.data_loading import load_and_normalize_corpora, load_article_wdids
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


def _log_cfg(logging_config: Mapping[str, Any]) -> dict[str, Any]:
    return {'PRINT_LOGS': bool(logging_config['print_logs'])}


def _legacy_model_cfg(model_config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        'hiddenDim': int(model_config['hidden_dim']),
        'dropouts1': float(model_config['dropouts1']),
        'dropouts2': float(model_config['dropouts2']),
    }


def _legacy_training_cfg(training_config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        'epochs': int(training_config['epochs']),
        'batchSize': int(training_config['batch_size']),
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
    }


def _legacy_eval_cfg(evaluation_config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        'thresholdPredict': float(evaluation_config['threshold_predict']),
        'thresholdEval': float(evaluation_config['threshold_eval']),
        'perCorpus': bool(evaluation_config['per_corpus']),
        'perClass': bool(evaluation_config['per_class']),
        'averagingType': str(evaluation_config['averaging_type']),
    }


@PipelineDecorator.component(
    return_values=['corpora', 'articleToWdids'],
    execution_queue='iptc_entity_tasks',
    task_type=TaskTypes.data_processing,
)
def load_data(paths_config: Mapping[str, Any]):
    """Load normalized corpora and article-to-wdIds mapping."""
    corpora = load_and_normalize_corpora(
        train_csv=paths_config['train_csv'],
        dev_csv=paths_config['dev_csv'],
        test_csv=paths_config['test_csv'],
        removed_cat_ids=paths_config['removed_cat_ids'],
    )
    article_to_wdids = load_article_wdids(article_entities_tsv=paths_config['article_entities_tsv'])
    return corpora, article_to_wdids


@PipelineDecorator.component(
    return_values=['trainData', 'devData', 'testData', 'featureDim'],
    execution_queue='iptc_entity_tasks',
    task_type=TaskTypes.data_processing,
)
def build_datasets(
    corpora,
    articleToWdids,
    paths_config: Mapping[str, Any],
    embedding_config: Mapping[str, Any],
):
    """Build entity-enhanced EmbeddingDataset objects for train/dev/test."""
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

    x_train = builder.build_features_for_corpus(corpus=corpora.train, article_to_wdids=articleToWdids)
    x_dev = builder.build_features_for_corpus(corpus=corpora.dev, article_to_wdids=articleToWdids)
    x_test = builder.build_features_for_corpus(corpus=corpora.test, article_to_wdids=articleToWdids)

    train_data = build_embedding_dataset(corpus=corpora.train, x_matrix=x_train)
    dev_data = build_embedding_dataset(corpus=corpora.dev, x_matrix=x_dev)
    test_data = build_embedding_dataset(corpus=corpora.test, x_matrix=x_test)
    feature_dim = int(x_train.shape[1])
    return train_data, dev_data, test_data, feature_dim


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

    paths_config = config_mapping['paths']
    embedding_config = config_mapping['embeddings']
    model_config = config_mapping['model']
    training_config = config_mapping['training']
    evaluation_config = config_mapping['evaluation']
    logging_config = config_mapping['logging']
    objective_corpora = config_mapping['objective_corpora']

    corpora, article_to_wdids = load_data(paths_config=paths_config)
    train_data, dev_data, test_data, feature_dim = build_datasets(
        corpora=corpora,
        articleToWdids=article_to_wdids,
        paths_config=paths_config,
        embedding_config=embedding_config,
    )

    model = createClassificationModel(
        embDim=feature_dim,
        outDim=corpora.train.catCnt,
        modelConfig=_legacy_model_cfg(model_config=model_config),
        textVectorizer='not None',
        logConfig=_log_cfg(logging_config=logging_config),
    )
    model = trainClassificationModel(
        model=model,
        trainData=train_data,
        devData=dev_data,
        trainingConfig=_legacy_training_cfg(training_config=training_config),
        logConfig=_log_cfg(logging_config=logging_config),
    )

    df_corpora_dev, _ = evaluateModel(
        model=model,
        evalData=dev_data,
        evaluationConfig=_legacy_eval_cfg(evaluation_config=evaluation_config),
        customThresholds=None,
    )
    df_corpora_test, df_classes_test = evaluateModel(
        model=model,
        evalData=test_data,
        evaluationConfig=_legacy_eval_cfg(evaluation_config=evaluation_config),
        customThresholds=None,
    )

    logger = task.get_logger()
    row_all = df_corpora_dev.loc[f"All-{evaluation_config['averaging_type']}"].to_dict()
    logger.report_scalar(title='Dev Evaluation Results', series='Precision', value=row_all['Precision'], iteration=0)
    logger.report_scalar(title='Dev Evaluation Results', series='Recall', value=row_all['Recall'], iteration=0)
    logger.report_scalar(title='Dev Evaluation Results', series='F04', value=row_all['F04'], iteration=0)
    logger.report_scalar(title='Dev Evaluation Results', series='F1', value=row_all['F1'], iteration=0)

    if objective_corpora in df_corpora_dev.index:
        row_objective = df_corpora_dev.loc[objective_corpora].to_dict()
        logger.report_scalar(
            title='Objective Evaluation Results',
            series='Precision',
            value=row_objective['Precision'],
            iteration=0,
        )
        logger.report_scalar(
            title='Objective Evaluation Results',
            series='Recall',
            value=row_objective['Recall'],
            iteration=0,
        )
        logger.report_scalar(
            title='Objective Evaluation Results',
            series='F04',
            value=row_objective['F04'],
            iteration=0,
        )
        logger.report_scalar(
            title='Objective Evaluation Results',
            series='F1',
            value=row_objective['F1'],
            iteration=0,
        )

    task.upload_artifact('Corpora Dataframe', artifact_object=df_corpora_test)
    task.upload_artifact('Classes Dataframe', artifact_object=df_classes_test)
    logger.report_table(title='Test Evaluation', series='Corpora Dataframe', iteration=0, table_plot=df_corpora_test)
    logger.report_table(title='Test Evaluation', series='Classes Dataframe', iteration=0, table_plot=df_classes_test)


def run_local_pipeline(config: Optional[PipelineConfig] = None) -> Tuple[PipelineConfig, Mapping[str, Any]]:
    """
    Resolve config and execute the ClearML pipeline locally.

    :param config: Optional custom pipeline config.
    :return: Tuple of resolved config and mapping passed to ClearML.
    """
    pipeline_config = config or PipelineConfig()
    resolved_config = resolve_paths(config=pipeline_config, root_dir=Path.cwd())
    config_mapping = asdict(resolved_config)
    run_training_pipeline(config_mapping=config_mapping)
    return resolved_config, config_mapping

