"""Entity embedding loading and per-entity chunk averaging."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np

from iptc_entity_pipeline.data_loading import get_doc_wdids

LOGGER = logging.getLogger(__name__)


class EntityEmbeddingStore:
    """
    Load entity embeddings from ``.npy`` files with caching.

    File pattern (v1): ``{wdid}_{lang}_{chunk}.npy``.
    """

    def __init__(
        self,
        *,
        root_dir: str,
        langs: tuple[str, ...] | str = ('en',),
        lang_mode: Literal['average', 'fallback'] = 'average',
    ) -> None:
        self._root_dir = Path(root_dir)
        raw_langs = (langs,) if isinstance(langs, str) else langs
        normalized_langs = tuple(dict.fromkeys(lang.strip() for lang in raw_langs if lang and lang.strip()))
        self._langs = normalized_langs if normalized_langs else ('en',)
        self._lang_mode = lang_mode
        self._cache: dict[str, np.ndarray | None] = {}
        self._wdid_lang_to_paths: dict[str, dict[str, list[Path]]] = {}
        self._sample_path: Path | None = None
        self._indexed_file_count = 0
        self._index_built = False
        self._train_mean: np.ndarray | None = None

    def _chunk_paths(self, *, wdid: str, lang: str) -> list[Path]:
        self._ensure_index()
        by_lang = self._wdid_lang_to_paths.get(wdid, {})
        return by_lang.get(lang, [])

    @staticmethod
    def _parse_stem(*, stem: str) -> tuple[str, str] | None:
        parts = stem.rsplit('_', maxsplit=2)
        if len(parts) != 3:
            return None
        wdid, lang, _chunk = parts
        if not wdid or not lang:
            return None
        return wdid, lang

    def _ensure_index(self) -> None:
        if self._index_built:
            return

        LOGGER.info('Scanning entity embedding files in %s for languages=%s', self._root_dir, self._langs)
        wdid_lang_to_paths: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
        selected_langs = set(self._langs)
        sample_path: Path | None = None
        file_count = 0
        for path in self._root_dir.glob('*.npy'):
            file_count += 1
            if file_count % 10000 == 0:
                LOGGER.info('Indexed %s entity embedding files so far', file_count)
            parsed = self._parse_stem(stem=path.stem)
            if parsed is None:
                continue
            wdid, lang = parsed
            if lang in selected_langs:
                wdid_lang_to_paths[wdid][lang].append(path)
                if sample_path is None:
                    sample_path = path

        for by_lang in wdid_lang_to_paths.values():
            for paths in by_lang.values():
                paths.sort()

        self._wdid_lang_to_paths = {
            wdid: dict(by_lang)
            for wdid, by_lang in wdid_lang_to_paths.items()
        }
        self._sample_path = sample_path
        self._index_built = True
        indexed_file_count = sum(
            len(paths)
            for by_lang in self._wdid_lang_to_paths.values()
            for paths in by_lang.values()
        )
        self._indexed_file_count = indexed_file_count
        LOGGER.info(
            'Indexed entity embedding files once: entities=%s files=%s',
            len(self._wdid_lang_to_paths),
            indexed_file_count,
        )

    def get_entity_embedding(self, *, wdid: str) -> np.ndarray | None:
        """
        Return averaged chunk embedding for one entity.

        :param wdid: Wikidata ID.
        :return: Averaged entity embedding or ``None`` when unavailable.
        """
        if wdid in self._cache:
            return self._cache[wdid]

        if self._lang_mode == 'fallback':
            for lang in self._langs:
                chunk_paths = self._chunk_paths(wdid=wdid, lang=lang)
                if not chunk_paths:
                    continue
                chunks = [np.asarray(np.load(path), dtype=np.float32) for path in chunk_paths]
                embedding = np.mean(np.vstack(chunks), axis=0, dtype=np.float32)
                self._cache[wdid] = embedding
                return embedding
            self._cache[wdid] = None
            return None

        chunks_all_langs: list[np.ndarray] = []
        for lang in self._langs:
            chunk_paths = self._chunk_paths(wdid=wdid, lang=lang)
            if not chunk_paths:
                continue
            chunks_all_langs.extend(np.asarray(np.load(path), dtype=np.float32) for path in chunk_paths)
        
        # so we do not have to search in memory for it 
        if not chunks_all_langs:
            self._cache[wdid] = None
            return None
        # there can be multiple chunks for the same wdid when the entity is too known
        embedding = np.mean(np.vstack(chunks_all_langs), axis=0, dtype=np.float32)
        self._cache[wdid] = embedding
        return embedding

    def indexed_file_count(self) -> int:
        """Return number of indexed entity embedding files for configured languages."""
        self._ensure_index()
        return int(self._indexed_file_count)

    def infer_embedding_dim(self) -> int:
        """
        Infer embedding dimensionality from first available chunk file.

        :return: Entity embedding dimension.
        :raises FileNotFoundError: if no matching ``.npy`` embeddings are found.
        """
        self._ensure_index()
        if self._sample_path is None:
            raise FileNotFoundError(
                f'No entity embeddings found in {self._root_dir} for languages={self._langs}. '
                'Expected files matching patterns "*_{lang}_*.npy".'
            )
        sample = np.asarray(np.load(self._sample_path), dtype=np.float32)
        LOGGER.info('Entity embedding dimension inferred as %s from %s', sample.shape[0], self._sample_path.name)
        return int(sample.shape[0])

    def compute_train_mean_from_corpus(self, *, corpus: Any) -> None:
        """
        Compute mean entity embedding over unique train-corpus wdIds with available vectors.

        :param corpus: Train corpus with entities attached to each document.
        :raises ValueError: when no resolvable entity embeddings exist in the train corpus.
        """
        unique_wdids: set[str] = set()
        for doc in corpus:
            unique_wdids.update(get_doc_wdids(doc))

        vectors: list[np.ndarray] = []
        resolved_wdids = 0
        for wdid in unique_wdids:
            embedding = self.get_entity_embedding(wdid=wdid)
            if embedding is None:
                continue
            resolved_wdids += 1
            vectors.append(embedding)

        if not vectors:
            raise ValueError(
                'No resolvable entity embeddings in train corpus: '
                f'unique_wdids={len(unique_wdids)} resolved_wdids=0 '
                f'root_dir={self._root_dir} langs={self._langs}'
            )

        self._train_mean = np.mean(np.vstack(vectors), axis=0, dtype=np.float32)
        LOGGER.info(
            'Train corpus entity mean computed: unique_wdids=%s resolved_wdids=%s dim=%s',
            len(unique_wdids),
            resolved_wdids,
            self._train_mean.shape[0],
        )

    def get_train_mean_embedding(self) -> np.ndarray:
        """
        Return train-corpus mean entity embedding used when an article has no entity hits.

        :return: Copy of cached train mean vector.
        :raises RuntimeError: when :meth:`compute_train_mean_from_corpus` was not called.
        """
        if self._train_mean is None:
            raise RuntimeError(
                'Train corpus entity mean is not computed; call compute_train_mean_from_corpus first.'
            )
        return np.asarray(self._train_mean, dtype=np.float32)

    def clear_cache(self) -> None:
        """
        Drop per-entity vector cache and path index to free memory.

        Once the dense feature matrices have been built, the in-memory
        caches accumulated by :meth:`get_entity_embedding` and
        :meth:`_ensure_index` are dead weight for any downstream step
        running in the same process (e.g. ``run_cv`` under
        ``PipelineDecorator.run_locally()``). The store can still be
        re-used afterwards; the next request will rebuild the index.

        :return: ``None``. ``_train_mean`` is preserved.
        """
        cached_vectors = len(self._cache)
        indexed_entities = len(self._wdid_lang_to_paths)
        self._cache = {}
        self._wdid_lang_to_paths = {}
        self._sample_path = None
        self._indexed_file_count = 0
        self._index_built = False
        LOGGER.info(
            'Cleared EntityEmbeddingStore cache: dropped_vectors=%s indexed_entities=%s',
            cached_vectors,
            indexed_entities,
        )

