"""Build entity-enhanced model input features."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from iptc_entity_pipeline.article_embeddings import ArticleEmbeddingProvider
from iptc_entity_pipeline.data_loading import get_article_text, get_doc_wdids
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
            wdids = get_doc_wdids(doc)
            if not wdids:
                LOGGER.warning('No linked entities for article_id=%s', doc.id)
            unique_requested_wdids.update(wdids)
            entity_embeddings = []
            missing_wdids: list[str] = []
            for wdid in wdids:
                entity_embedding = self._entity_embedding_store.get_entity_embedding(wdid=wdid)
                if entity_embedding is not None:
                    entity_embeddings.append(entity_embedding)
                else:
                    missing_wdids.append(wdid)
                    continue
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
            pooled_entity = self._pooling_strategy.pool(entity_embeddings=entity_embeddings, embedding_dim=entity_dim)
            row = np.concatenate([article_embedding, pooled_entity]).astype(np.float32)
            rows.append(row)

            processed_docs = idx + 1
            if processed_docs % progress_interval == 0 or processed_docs == total_docs:
                LOGGER.info('Linked article/entity embeddings for %s/%s articles', processed_docs, total_docs)

        unique_total_entities = len(unique_requested_wdids)
        unique_missing_entities = len(unique_missing_wdids_all)
        avg_missing_per_article = (total_missing_embeddings / total_docs) if total_docs else 0.0
        avg_found_per_article = (total_found_embeddings / total_docs) if total_docs else 0.0
        LOGGER.info(
            (
                'Entity embedding final stats: unique_missing=%s unique_total=%s missing_ratio=%.4f '
                'avg_missing_per_article=%.4f avg_found_per_article=%.4f'
            ),
            unique_missing_entities,
            unique_total_entities,
            (unique_missing_entities / unique_total_entities) if unique_total_entities else 0.0,
            avg_missing_per_article,
            avg_found_per_article,
        )

        return np.vstack(rows)

