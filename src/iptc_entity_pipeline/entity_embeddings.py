"""Entity embedding loading and per-entity chunk averaging."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)


class EntityEmbeddingStore:
    """
    Load entity embeddings from ``.npy`` files with caching.

    File pattern (v1): ``{wdid}_{lang}_{chunk}.npy``.
    """

    def __init__(self, *, root_dir: str, langs: tuple[str, ...] = ('en',)) -> None:
        self._root_dir = Path(root_dir)
        normalized_langs = tuple(dict.fromkeys(lang.strip() for lang in langs if lang and lang.strip()))
        self._langs = normalized_langs if normalized_langs else ('en',)
        self._cache: dict[str, np.ndarray | None] = {}
        self._wdid_lang_to_paths: dict[str, dict[str, list[Path]]] = {}
        self._sample_path: Path | None = None
        self._index_built = False

    def _chunk_paths(self, *, wdid: str, lang: str) -> list[Path]:
        self._ensure_index()
        by_lang = self._wdid_lang_to_paths.get(wdid, {})
        return by_lang.get(lang, [])

    def _ensure_index(self) -> None:
        if self._index_built:
            return

        LOGGER.info('Scanning entity embedding files in %s for languages=%s', self._root_dir, self._langs)
        wdid_lang_to_paths: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
        sample_path: Path | None = None
        file_count = 0
        for lang in self._langs:
            pattern = f'*_{lang}_*.npy'
            for path in self._root_dir.glob(pattern):
                file_count += 1
                if file_count % 10000 == 0:
                    LOGGER.info('Indexed %s entity embedding files so far', file_count)
                if sample_path is None:
                    sample_path = path
                stem = path.stem
                split_suffix = f'_{lang}_'
                if split_suffix not in stem:
                    continue
                wdid = stem.split(split_suffix, maxsplit=1)[0]
                if not wdid:
                    continue
                wdid_lang_to_paths[wdid][lang].append(path)

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

        per_lang_embeddings: list[np.ndarray] = []
        for lang in self._langs:
            chunk_paths = self._chunk_paths(wdid=wdid, lang=lang)
            if not chunk_paths:
                continue
            chunks = [np.asarray(np.load(path), dtype=np.float32) for path in chunk_paths]
            per_lang_embeddings.append(np.mean(np.vstack(chunks), axis=0, dtype=np.float32))

        if not per_lang_embeddings:
            self._cache[wdid] = None
            return None

        embedding = np.mean(np.vstack(per_lang_embeddings), axis=0, dtype=np.float32)
        self._cache[wdid] = embedding
        return embedding

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

