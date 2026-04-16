"""Build entity-enhanced model input features."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from clearml import Task

from iptc_entity_pipeline.article_embeddings import ArticleEmbeddingProvider
from iptc_entity_pipeline.data_loading import get_article_text, get_doc_weighted_wdids
from iptc_entity_pipeline.entity_embeddings import EntityEmbeddingStore
from iptc_entity_pipeline.pooling import EntityPoolingStrategy

LOGGER = logging.getLogger(__name__)


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
    ) -> np.ndarray:
        """
        Build feature matrix for all docs in a corpus.

        Entity wdIds are read from ``doc.entities`` (attached :class:`LinkedEntity` objects).

        :param corpus: ``geneea.catlib.data.Corpus`` object with entities attached to each doc.
        :param ensure_article_embeddings: If ``True``, precompute missing article embeddings for the corpus.
        :return: Feature matrix of shape ``[docs, article_dim + entity_dim]``.
        """
        if ensure_article_embeddings:
            self._article_embedding_provider.recompute_embeddings(corpus=corpus)
        entity_dim = self._entity_embedding_store.infer_embedding_dim()
        rows: list[np.ndarray] = []
        total_docs = len(corpus)
        progress_interval = max(1, total_docs // 20) if total_docs else 1
        unique_requested_wdids: set[str] = set()
        unique_missing_wdids_all: set[str] = set()
        total_found_embeddings = 0
        total_missing_embeddings = 0

        for idx, doc in enumerate(corpus):
            article_embedding = self._article_embedding_provider.get_embedding(
                article_id=doc.id,
                article_text=get_article_text(doc),
                article_doc=doc,
            )
            weighted_wdids = get_doc_weighted_wdids(doc)
            wdids = [wd_id for wd_id, _ in weighted_wdids]
            if not wdids:
                LOGGER.warning('No linked entities for article_id=%s', doc.id)
            unique_requested_wdids.update(wdids)
            entity_embeddings: list[np.ndarray] = []
            entity_weights: list[float] = []
            missing_wdids: list[str] = []
            for wdid, weight in weighted_wdids:
                entity_embedding = self._entity_embedding_store.get_entity_embedding(wdid=wdid)
                if entity_embedding is not None:
                    entity_embeddings.append(entity_embedding)
                    entity_weights.append(weight)
                else:
                    missing_wdids.append(wdid)

            total_found_embeddings += len(entity_embeddings)
            total_missing_embeddings += len(missing_wdids)
            unique_missing_wdids_all.update(missing_wdids)
            if missing_wdids:
                unique_missing_wdids = sorted(set(missing_wdids))
                LOGGER.warning(
                    'Missing entity embeddings for article_id=%s: %s',
                    doc.id,
                    ', '.join(unique_missing_wdids),
                )
            pooled_entity = self._pooling_strategy.pool(
                entity_embeddings=entity_embeddings,
                embedding_dim=entity_dim,
                weights=entity_weights,
            )
            row = np.concatenate([article_embedding, pooled_entity]).astype(np.float32)
            rows.append(row)

            processed_docs = idx + 1
            if processed_docs % progress_interval == 0 or processed_docs == total_docs:
                LOGGER.info('Linked article/entity embeddings for %s/%s articles', processed_docs, total_docs)

        unique_total_entities = len(unique_requested_wdids)
        unique_missing_entities = len(unique_missing_wdids_all)
        avg_missing_per_article = (total_missing_embeddings / total_docs) if total_docs else 0.0
        avg_found_per_article = (total_found_embeddings / total_docs) if total_docs else 0.0
        final_stats_message = (
            'Entity embedding final stats: '
            f'unique_missing={unique_missing_entities} '
            f'unique_total={unique_total_entities} '
            f'missing_ratio='
            f'{((unique_missing_entities / unique_total_entities) if unique_total_entities else 0.0):.4f} '
            f'avg_missing_per_article={avg_missing_per_article:.4f} '
            f'avg_found_per_article={avg_found_per_article:.4f}'
        )
        LOGGER.info(
            final_stats_message,
        )
        task = Task.current_task()
        if task is not None:
            task.get_logger().report_text(final_stats_message, print_console=True)

        return np.vstack(rows)

