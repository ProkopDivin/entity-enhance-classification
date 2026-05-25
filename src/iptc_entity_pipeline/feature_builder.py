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


@dataclass(frozen=True)
class RaggedFeatureData:
    """Ragged entity features for no-pooling mode."""

    article_matrix: np.ndarray
    entity_matrices: tuple[np.ndarray, ...]
    stats: FeatureBuildStats


@dataclass
class _BuildTracking:
    """Mutable counters/state used while building one corpus matrix."""

    unique_requested_wdids: set[str]
    unique_missing_wdids: set[str]
    total_found_embeddings: int
    total_missing_embeddings: int


class FeatureBuilder:
    """Create concatenated article + pooled-entity embeddings for one corpus."""

    def __init__(
        self,
        *,
        article_embedding_provider: ArticleEmbeddingProvider | None,
        entity_embedding_store: EntityEmbeddingStore,
        pooling_strategy: EntityPoolingStrategy,
        use_article_embeddings: bool = True,
        combine_method: str = 'concat',
    ) -> None:
        if combine_method != 'concat':
            raise ValueError(f'Unsupported v1 combine method: {combine_method}')
        if use_article_embeddings and article_embedding_provider is None:
            raise ValueError('article_embedding_provider is required when use_article_embeddings=True')
        self._article_embedding_provider = article_embedding_provider
        self._entity_embedding_store = entity_embedding_store
        self._pooling_strategy = pooling_strategy
        self._use_article_embeddings = use_article_embeddings
        self._combine_method = combine_method

    @staticmethod
    def _init_tracking() -> _BuildTracking:
        """Initialize mutable tracking state for one corpus pass."""
        return _BuildTracking(
            unique_requested_wdids=set(),
            unique_missing_wdids=set(),
            total_found_embeddings=0,
            total_missing_embeddings=0,
        )

    @staticmethod
    def _update_log(*, doc_id: str, pooling_result: Any, tracking: _BuildTracking) -> None:
        """Update per-corpus counters and emit per-document diagnostics."""
        wdids = list(pooling_result.requested_wdids)
        if not wdids:
            LOGGER.warning('No linked entities for article_id=%s', doc_id)

        tracking.unique_requested_wdids.update(wdids)
        tracking.total_found_embeddings += pooling_result.found_embeddings
        tracking.total_missing_embeddings += pooling_result.missing_embeddings
        tracking.unique_missing_wdids.update(pooling_result.missing_wdids)

        if pooling_result.missing_wdids:
            unique_missing_wdids = sorted(set(pooling_result.missing_wdids))
            LOGGER.warning(
                'Missing entity embeddings for article_id=%s: %s',
                doc_id,
                ', '.join(unique_missing_wdids),
            )

    def _final_message(self, *, tracking: _BuildTracking, total_docs: int) -> str:
        """Compose final entity-linking summary for logs and ClearML."""
        unique_total_entities = len(tracking.unique_requested_wdids)
        unique_missing_entities = len(tracking.unique_missing_wdids)
        indexed_embedding_files = self._entity_embedding_store.indexed_file_count()
        avg_missing_per_article = (tracking.total_missing_embeddings / total_docs) if total_docs else 0.0
        avg_found_per_article = (tracking.total_found_embeddings / total_docs) if total_docs else 0.0
        return (
            'Entity embedding final stats: '
            f'unique_missing={unique_missing_entities} '
            f'unique_total={unique_total_entities} '
            f'indexed_embedding_files={indexed_embedding_files} '
            f'missing_ratio='
            f'{((unique_missing_entities / unique_total_entities) if unique_total_entities else 0.0):.4f} '
            f'avg_missing_per_article={avg_missing_per_article:.4f} '
            f'avg_found_per_article={avg_found_per_article:.4f}'
        )

    @staticmethod
    def _report_final_stats(*, final_stats_message: str, clearml_logger: Any | None) -> None:
        """Log final summary locally and report it to ClearML logger when available."""
        LOGGER.info(final_stats_message)
        if clearml_logger is not None:
            clearml_logger.report_text(final_stats_message, print_console=True)
            return

        task = Task.current_task()
        if task is not None:
            task.get_logger().report_text(final_stats_message, print_console=True)

    def build_features(
        self,
        *,
        corpus: Any,
        clearml_logger: Any | None = None,
        return_stats: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, FeatureBuildStats]:
        """
        Build feature matrix for all docs in a corpus.

        Entity wdIds are read from ``doc.entities`` (attached :class:`LinkedEntity` objects).

        :param corpus: ``geneea.catlib.data.Corpus`` object with entities attached to each doc.
        :param clearml_logger: Optional ClearML task logger used to report summary text.
        :param return_stats: If ``True``, return matrix together with coverage stats.
        :return: Feature matrix of shape ``[docs, article_dim + entity_dim]`` or ``[docs, entity_dim]``.
        """
        entity_dim = self._entity_embedding_store.infer_embedding_dim()
        rows: list[np.ndarray] = []
        total_docs = len(corpus)
        tracking = self._init_tracking()

        for _, doc in enumerate(corpus):
            pooling_result = self._pooling_strategy.pool(
                doc=doc,
                entity_embedding_store=self._entity_embedding_store,
                embedding_dim=entity_dim,
            )
            self._update_log(
                doc_id=doc.id,
                pooling_result=pooling_result,
                tracking=tracking,
            )
            if self._use_article_embeddings:
                assert self._article_embedding_provider is not None
                article_embedding = self._article_embedding_provider.get_embedding(article_id=doc.id)
                row = np.concatenate([article_embedding, pooling_result.pooled_embedding]).astype(np.float32)
            else:
                row = np.asarray(pooling_result.pooled_embedding, dtype=np.float32)
            rows.append(row)

        final_stats_message = self._final_message(tracking=tracking, total_docs=total_docs)
        self._report_final_stats(final_stats_message=final_stats_message, clearml_logger=clearml_logger)

        matrix = np.vstack(rows)
        if not return_stats:
            return matrix

        stats = FeatureBuildStats(
            unique_requested_wdids=frozenset(tracking.unique_requested_wdids),
            unique_missing_wdids=frozenset(tracking.unique_missing_wdids),
            total_found_embeddings=tracking.total_found_embeddings,
            total_missing_embeddings=tracking.total_missing_embeddings,
            entity_dim=int(entity_dim),
        )
        return matrix, stats

    def build_ragged_features(
        self,
        *,
        corpus: Any,
        clearml_logger: Any | None = None,
    ) -> RaggedFeatureData:
        """Build article vectors and per-document entity matrices for no-pooling mode."""
        if not self._use_article_embeddings:
            raise ValueError('no_pooling mode currently requires use_article_embeddings=True')

        entity_dim = self._entity_embedding_store.infer_embedding_dim()
        tracking = self._init_tracking()
        article_rows: list[np.ndarray] = []
        entity_rows: list[np.ndarray] = []
        total_docs = len(corpus)
        assert self._article_embedding_provider is not None

        for doc in corpus:
            pooling_result = self._pooling_strategy.pool(
                doc=doc,
                entity_embedding_store=self._entity_embedding_store,
                embedding_dim=entity_dim,
            )
            self._update_log(
                doc_id=doc.id,
                pooling_result=pooling_result,
                tracking=tracking,
            )
            article_embedding = self._article_embedding_provider.get_embedding(article_id=doc.id)
            article_rows.append(np.asarray(article_embedding, dtype=np.float32))
            if pooling_result.pooled_embedding.ndim == 1:
                entity_rows.append(
                    np.asarray(pooling_result.pooled_embedding, dtype=np.float32).reshape(1, -1)
                )
            else:
                entity_rows.append(np.asarray(pooling_result.pooled_embedding, dtype=np.float32))

        final_stats_message = self._final_message(tracking=tracking, total_docs=total_docs)
        self._report_final_stats(final_stats_message=final_stats_message, clearml_logger=clearml_logger)
        stats = FeatureBuildStats(
            unique_requested_wdids=frozenset(tracking.unique_requested_wdids),
            unique_missing_wdids=frozenset(tracking.unique_missing_wdids),
            total_found_embeddings=tracking.total_found_embeddings,
            total_missing_embeddings=tracking.total_missing_embeddings,
            entity_dim=int(entity_dim),
        )
        return RaggedFeatureData(
            article_matrix=np.vstack(article_rows),
            entity_matrices=tuple(entity_rows),
            stats=stats,
        )

