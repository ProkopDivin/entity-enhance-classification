"""Dataset-building pipeline step with article/entity embedding linkage."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from clearml import Task, TaskTypes
from clearml.automation.controller import PipelineDecorator

from iptc_entity_pipeline.article_embeddings import ArticleEmbeddingProvider
from iptc_entity_pipeline.config import EmbeddingConfig, PathsConfig, config_from_dict
from iptc_entity_pipeline.dataset_builder import build_embedding_dataset
from iptc_entity_pipeline.entity_embeddings import EntityEmbeddingStore
from iptc_entity_pipeline.feature_builder import FeatureBuildStats, FeatureBuilder
from iptc_entity_pipeline.pooling import (
    EntityPoolingStrategy,
    MeanEntityPooling,
    MentionWeightedSumEntityPooling,
    SumEntityPooling,
    WeightedMeanEntityPooling,
    WeightedSumEntityPooling,
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


def build_article_only_matrix(
    *,
    split_corpus: Any,
    split_name: str,
    article_provider: ArticleEmbeddingProvider,
    logger: logging.Logger,
) -> np.ndarray:
    """Build article-only feature matrix for one corpus split."""
    rows: list[np.ndarray] = []
    total_docs = len(split_corpus)
    logger.info('Building article-only features for %s corpus (%s articles)', split_name, total_docs)
    for idx, doc in enumerate(split_corpus, start=1):
        article_embedding = article_provider.get_embedding(article_id=doc.id)
        rows.append(np.asarray(article_embedding, dtype=np.float32))
        if idx % 1000 == 0 or idx == total_docs:
            logger.info(
                'Built article-only features for %s/%s articles in %s corpus',
                idx,
                total_docs,
                split_name,
            )
    return np.vstack(rows)


def no_entities(
    *,
    corpora: Any,
    article_provider: ArticleEmbeddingProvider,
    logger: logging.Logger,
) -> DatasetBundle:
    """Build train/test datasets for article-only feature mode."""
    logger.info('Entity embeddings disabled by config; running article-only feature pipeline')
    x_train = build_article_only_matrix(
        split_corpus=corpora.train,
        split_name='train',
        article_provider=article_provider,
        logger=logger,
    )
    x_test = build_article_only_matrix(
        split_corpus=corpora.test,
        split_name='test',
        article_provider=article_provider,
        logger=logger,
    )
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


def get_pooling(
    *,
    emb_cfg: EmbeddingConfig,
    logger: logging.Logger,
) -> EntityPoolingStrategy:
    """Select pooling strategy and entity weighting mode from config."""
    if emb_cfg.entity_pooling == 'weighted_mean':
        pooling = WeightedMeanEntityPooling()
        logger.info('Using relevance-weighted entity pooling (normalized weighted mean)')
    elif emb_cfg.entity_pooling == 'weighted_sum':
        pooling = MentionWeightedSumEntityPooling()
        logger.info('Using mention-weighted entity pooling (weighted sum)')
    elif emb_cfg.entity_pooling == 'weighted_sum_relevance':
        pooling = WeightedSumEntityPooling()
        logger.info('Using relevance-weighted entity pooling (weighted sum)')
    elif emb_cfg.entity_pooling == 'mean':
        pooling = MeanEntityPooling()
        logger.info('Using unweighted entity pooling (mean)')
    elif emb_cfg.entity_pooling == 'sum':
        pooling = SumEntityPooling()
        logger.info('Using unweighted entity pooling (sum)')
    else:
        raise ValueError(f'Unsupported entity_pooling: {emb_cfg.entity_pooling}')
    return pooling


def report_entity_stats(
    *,
    stats: FeatureBuildStats,
    clearml_task: Task | None,
    logger: logging.Logger,
) -> EntityEmbeddingStats:
    """Aggregate per-corpus coverage stats into one entity summary."""

    total_wdids = len(stats.unique_requested_wdids)
    missing_cnt = len(stats.unique_missing_wdids)
    found_cnt = total_wdids - missing_cnt
    entity_dim = stats.entity_dim
    linked_ratio = (found_cnt / total_wdids) if total_wdids else 0.0
    summary_message = (
        'Entity linking summary: '
        f'linked_unique={total_wdids} '
        f'linked_with_embedding={found_cnt} '
        f'unlinked_missing_embedding={missing_cnt} '
        f'coverage={linked_ratio:.4f} '
        f'entity_dim={entity_dim}'
    )
    logger.info(summary_message)
    if clearml_task is not None:
        clearml_task.get_logger().report_text(summary_message, print_console=True)
