"""Build entity-enhanced model input features."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from clearml import Task

from iptc_entity_pipeline.article_embeddings import ArticleEmbeddingProvider
from iptc_entity_pipeline.entity_embeddings import EntityEmbeddingStore
from iptc_entity_pipeline.pooling import EntityPoolingStrategy

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureBuildStats:
    """Entity embedding coverage stats collected while building corpus features."""

    unique_requested_wdids: frozenset[str]
    unique_missing_wdids: frozenset[str]
    total_found_embeddings: int
    total_missing_embeddings: int
    entity_dim: int


class FeatureBuilder:
    """Create concatenated article + pooled-entity embeddings for one corpus."""

    def __init__(
        self,
        *,
        article_embedding_provider: ArticleEmbeddingProvider,
        entity_embedding_store: EntityEmbeddingStore,
        pooling_strategy: EntityPoolingStrategy,
        combine_method: str = 'concat',
    ) -> None:
        if combine_method != 'concat':
            raise ValueError(f'Unsupported v1 combine method: {combine_method}')
        self._article_embedding_provider = article_embedding_provider
        self._entity_embedding_store = entity_embedding_store
        self._pooling_strategy = pooling_strategy
        self._combine_method = combine_method

    def build_features_for_corpus(
        self,
        *,
        corpus: Any,
        ensure_article_embeddings: bool = False,
        clearml_logger: Any | None = None,
        return_stats: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, FeatureBuildStats]:
        """
        Build feature matrix for all docs in a corpus.

        Entity wdIds are read from ``doc.entities`` (attached :class:`LinkedEntity` objects).

        :param corpus: ``geneea.catlib.data.Corpus`` object with entities attached to each doc.
        :param ensure_article_embeddings: If ``True``, precompute missing article embeddings for the corpus.
        :param clearml_logger: Optional ClearML task logger used to report summary text.
        :param return_stats: If ``True``, return matrix together with coverage stats.
        :return: Feature matrix of shape ``[docs, article_dim + entity_dim]``.
        """
        if ensure_article_embeddings:
            self._article_embedding_provider.prepare_embeddings(corpus=corpus)
        entity_dim = self._entity_embedding_store.infer_embedding_dim()
        rows: list[np.ndarray] = []
        total_docs = len(corpus)
        progress_interval = max(1, total_docs // 20) if total_docs else 1
        unique_requested_wdids: set[str] = set()
        unique_missing_wdids_all: set[str] = set()
        total_found_embeddings = 0
        total_missing_embeddings = 0

        for idx, doc in enumerate(corpus):
            article_embedding = self._article_embedding_provider.get_embedding(article_id=doc.id)
            pooling_result = self._pooling_strategy.pool(
                doc=doc,
                entity_embedding_store=self._entity_embedding_store,
                embedding_dim=entity_dim,
            )
            wdids = list(pooling_result.requested_wdids)
            if not wdids:
                LOGGER.warning('No linked entities for article_id=%s', doc.id)
            unique_requested_wdids.update(wdids)
            total_found_embeddings += pooling_result.found_embeddings
            total_missing_embeddings += pooling_result.missing_embeddings
            unique_missing_wdids_all.update(pooling_result.missing_wdids)
            if pooling_result.missing_wdids:
                unique_missing_wdids = sorted(set(pooling_result.missing_wdids))
                LOGGER.warning(
                    'Missing entity embeddings for article_id=%s: %s',
                    doc.id,
                    ', '.join(unique_missing_wdids),
                )
            row = np.concatenate([article_embedding, pooling_result.pooled_embedding]).astype(np.float32)
            rows.append(row)

            processed_docs = idx + 1
            if processed_docs % progress_interval == 0 or processed_docs == total_docs:
                LOGGER.info('Linked article/entity embeddings for %s/%s articles', processed_docs, total_docs)

        unique_total_entities = len(unique_requested_wdids)
        unique_missing_entities = len(unique_missing_wdids_all)
        indexed_embedding_files = self._entity_embedding_store.indexed_file_count()
        avg_missing_per_article = (total_missing_embeddings / total_docs) if total_docs else 0.0
        avg_found_per_article = (total_found_embeddings / total_docs) if total_docs else 0.0
        final_stats_message = (
            'Entity embedding final stats: '
            f'unique_missing={unique_missing_entities} '
            f'unique_total={unique_total_entities} '
            f'indexed_embedding_files={indexed_embedding_files} '
            f'missing_ratio='
            f'{((unique_missing_entities / unique_total_entities) if unique_total_entities else 0.0):.4f} '
            f'avg_missing_per_article={avg_missing_per_article:.4f} '
            f'avg_found_per_article={avg_found_per_article:.4f}'
        )
        LOGGER.info(
            final_stats_message,
        )
        if clearml_logger is not None:
            clearml_logger.report_text(final_stats_message, print_console=True)
        else:
            task = Task.current_task()
            if task is not None:
                task.get_logger().report_text(final_stats_message, print_console=True)

        matrix = np.vstack(rows)
        if not return_stats:
            return matrix

        stats = FeatureBuildStats(
            unique_requested_wdids=frozenset(unique_requested_wdids),
            unique_missing_wdids=frozenset(unique_missing_wdids_all),
            total_found_embeddings=total_found_embeddings,
            total_missing_embeddings=total_missing_embeddings,
            entity_dim=int(entity_dim),
        )
        return matrix, stats

