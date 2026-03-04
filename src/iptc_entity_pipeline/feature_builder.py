"""Build entity-enhanced model input features."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np

from iptc_entity_pipeline.article_embeddings import ArticleEmbeddingProvider
from iptc_entity_pipeline.data_loading import get_article_text
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
        article_to_wdids: Mapping[str, list[str]],
        ensure_article_embeddings: bool = False,
    ) -> np.ndarray:
        """
        Build feature matrix for all docs in a corpus.

        :param corpus: ``geneea.catlib.data.Corpus`` object.
        :param article_to_wdids: Mapping ``article_id -> wdIds``.
        :param ensure_article_embeddings: If ``True``, precompute missing article embeddings for the corpus.
        :return: Feature matrix of shape ``[docs, article_dim + entity_dim]``.
        """
        if ensure_article_embeddings:
            self._article_embedding_provider.recompute_embeddings(corpus=corpus)
        entity_dim = self._entity_embedding_store.infer_embedding_dim()
        rows: list[np.ndarray] = []
        total_docs = len(corpus)

        for idx, doc in enumerate(corpus):
            article_embedding = self._article_embedding_provider.get_embedding(
                article_id=doc.id,
                article_text=get_article_text(doc),
                article_doc=doc,
            )
            wdids = article_to_wdids.get(doc.id, [])
            if not wdids:
                LOGGER.warning('No linked entities for article_id=%s (article missing in article_to_wdids mapping)', doc.id)
            entity_embeddings = []
            missing_wdids: list[str] = []
            for wdid in wdids:
                entity_embedding = self._entity_embedding_store.get_entity_embedding(wdid=wdid)
                if entity_embedding is not None:
                    entity_embeddings.append(entity_embedding)
                else:
                    missing_wdids.append(wdid)
                    continue
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
            if processed_docs % 1000 == 0 or processed_docs == total_docs:
                LOGGER.info('Linked article/entity embeddings for %s/%s articles', processed_docs, total_docs)

        return np.vstack(rows)

