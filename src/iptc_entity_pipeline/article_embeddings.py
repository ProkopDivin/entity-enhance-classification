"""Article embedding computation, caching, and cache-only loading."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingCacheStats:
    """Summary of article embedding cache state after a recompute pass."""

    total_docs: int
    cached_docs: int
    missing_docs: int
    computed_docs: int


class ArticleEmbeddingProvider:
    """
    Provide article embeddings from disk and compute missing ones.

    Embedding files follow ``<article_id>.npy`` naming.
    """

    def __init__(
        self,
        *,
        embeddings_dir: str,
        model_name: str,
        embed_svc_url: str = 'http://tau.g:5533',
        embedding_dim: int = 384,
    ) -> None:
        self._embeddings_dir = Path(embeddings_dir)
        self._embeddings_dir.mkdir(parents=True, exist_ok=True)
        self._model_name = model_name
        self._embed_svc_url = embed_svc_url
        self._embedding_dim = embedding_dim
        self._doc_vectorizer = None

    def _get_doc_vectorizer(self):
        if self._doc_vectorizer is None:
            from geneea.catlib.vec.vectorizer import DocVectorizer, SvcTextVectorizer

            text_vectorizer = SvcTextVectorizer(
                embedSvcUrl=self._embed_svc_url,
                modelId=self._model_name,
                embedDim=self._embedding_dim,
            )
            self._doc_vectorizer = DocVectorizer.fromTextVectorizer(text_vectorizer)
            LOGGER.info(
                'Using origin-compatible document vectorizer: model=%s, embedSvcUrl=%s',
                self._model_name,
                self._embed_svc_url,
            )
        return self._doc_vectorizer

    def _path_for_article(self, *, article_id: str) -> Path:
        return self._embeddings_dir / f'{article_id}.npy'

    def _to_numpy(self, *, matrix: Any) -> np.ndarray:
        if hasattr(matrix, 'detach') and hasattr(matrix, 'cpu'):
            return matrix.detach().cpu().numpy()
        return np.asarray(matrix)

    def _compute_cache_embeddings(self, *, docs: Sequence[Any]) -> None:
        if not docs:
            return
        from geneea.catlib.data import Corpus
        from geneea.catlib.vec.dataset import EmbeddingDataset

        corpus = Corpus(doc for doc in docs)
        LOGGER.info('Computing and caching %s missing article embeddings', len(docs))
        embedding_dataset = EmbeddingDataset.fromCorpus(corpus, vectorizer=self._get_doc_vectorizer())
        vectors = self._to_numpy(matrix=embedding_dataset.X)

        total_docs = len(docs)
        for idx, (doc, vector) in enumerate(zip(corpus, vectors), start=1):
            emb_path = self._path_for_article(article_id=doc.id)
            np.save(emb_path, np.asarray(vector, dtype=np.float32))
            if idx % 1000 == 0:
                LOGGER.info('Computed and cached article embeddings for %s/%s missing articles', idx, total_docs)

        LOGGER.info('Computed and cached article embeddings for %s/%s missing articles', total_docs, total_docs)

    def prepare_embeddings(self, *, corpus: Any) -> EmbeddingCacheStats:
        """
        Compute and cache missing article embeddings for the whole corpus.

        After this call every article in the corpus has a cached ``.npy`` file,
        so subsequent :meth:`get_embedding` calls will always hit the cache.

        :param corpus: Corpus of documents with ``id``.
        :return: Summary with total, cached, missing and computed article counts.
        """
        total = len(corpus)
        LOGGER.info('Checking article embedding cache for corpus_size=%s', total)
        missing_docs = [doc for doc in corpus if not self._path_for_article(article_id=doc.id).is_file()]
        cached = total - len(missing_docs)
        computed = len(missing_docs)

        if missing_docs:
            LOGGER.info('Found %s missing article embeddings', computed)
            self._compute_cache_embeddings(docs=missing_docs)

        LOGGER.info(
            'Article embeddings prepared for %s/%s articles (computed=%s, cached=%s)',
            total, total, computed, cached,
        )
        return EmbeddingCacheStats(total_docs=total, cached_docs=cached, missing_docs=computed, computed_docs=computed)

    def get_embedding(self, *, article_id: str) -> np.ndarray:
        """
        Load a cached article embedding from disk.

        Raises :exc:`FileNotFoundError` if the embedding is not in cache.
        Call :meth:`prepare_embeddings` first to ensure all embeddings are present.

        :param article_id: Article identifier.
        :return: Article embedding vector.
        """
        emb_path = self._path_for_article(article_id=article_id)
        if not emb_path.is_file():
            raise FileNotFoundError(
                f'Embedding not cached for article_id={article_id!r}. '
                'Run prepare_embeddings() first.'
            )
        return np.load(emb_path)


