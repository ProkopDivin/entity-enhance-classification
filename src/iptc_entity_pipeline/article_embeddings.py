"""Article embedding loading with fallback computation and persistent cache."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)


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
        backend: str = 'origin_service',
        embed_svc_url: str = 'http://tau.g:5533',
        embedding_dim: int = 384,
    ) -> None:
        self._embeddings_dir = Path(embeddings_dir)
        self._embeddings_dir.mkdir(parents=True, exist_ok=True)
        self._model_name = model_name
        self._backend = backend
        self._embed_svc_url = embed_svc_url
        self._embedding_dim = embedding_dim
        self._model = None
        self._doc_vectorizer = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            LOGGER.info('Loading article embedding model: %s', self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

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

        for idx, (doc, vector) in enumerate(zip(corpus, vectors), start=1):
            emb_path = self._path_for_article(article_id=doc.id)
            np.save(emb_path, np.asarray(vector, dtype=np.float32))
            if idx % 1000 == 0:
                LOGGER.info('Computed and cached %s missing article embeddings', idx)

        LOGGER.info('Computed and cached %s missing article embeddings', len(docs))

    def recompute_embeddings(self, *, corpus: Any) -> None:
        """
        Precompute and cache missing article embeddings for the whole corpus.

        :param corpus: Corpus of documents with ``id``.
        :return: None
        """
        LOGGER.info('Checking article embedding cache for corpus_size=%s', len(corpus))
        missing_docs = [doc for doc in corpus if not self._path_for_article(article_id=doc.id).is_file()]
        if not missing_docs:
            LOGGER.info('All article embeddings present in cache, nothing to compute')
            return

        LOGGER.info('Found %s missing article embeddings', len(missing_docs))
        if self._backend == 'origin_service':
            self._compute_cache_embeddings(docs=missing_docs)
            return

        if self._backend == 'local_sentence_transformers':
            from iptc_entity_pipeline.data_loading import get_article_text

            for idx, doc in enumerate(missing_docs, start=1):
                _ = self.get_embedding(article_id=doc.id, article_text=get_article_text(doc), article_doc=doc)
                if idx % 1000 == 0:
                    LOGGER.info('Computed and cached %s missing article embeddings', idx)
            return

        raise ValueError(f'Unsupported article embedding backend: {self._backend}')

    def get_embedding(self, *, article_id: str, article_text: str, article_doc: Any | None = None) -> np.ndarray:
        """
        Load embedding from cache or compute and save it.

        :param article_id: Article identifier.
        :param article_text: Article text used when embedding is missing.
        :param article_doc: Original document object required for origin-compatible backend.
        :return: Article embedding vector.
        """
        emb_path = self._path_for_article(article_id=article_id)
        if emb_path.is_file():
            return np.load(emb_path)

        if self._backend == 'origin_service':
            if article_doc is None:
                raise ValueError('article_doc must be provided for origin_service backend.')
            self._compute_cache_embeddings(docs=[article_doc])
            return np.load(emb_path)

        if self._backend == 'local_sentence_transformers':
            model = self._get_model()
            embedding = model.encode(sentences=[article_text], show_progress_bar=False, normalize_embeddings=False)[0]
            embedding = np.asarray(embedding, dtype=np.float32)
            np.save(emb_path, embedding)
            LOGGER.debug('Computed and cached missing article embedding: %s', emb_path.name)
            return embedding

        raise ValueError(f'Unsupported article embedding backend: {self._backend}')

