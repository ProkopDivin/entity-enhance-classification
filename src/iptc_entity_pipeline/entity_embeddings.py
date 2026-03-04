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

    def __init__(self, *, root_dir: str, lang: str = 'en') -> None:
        self._root_dir = Path(root_dir)
        self._lang = lang
        self._cache: dict[str, np.ndarray | None] = {}
        self._wdid_to_paths: dict[str, list[Path]] = {}
        self._sample_path: Path | None = None
        self._index_built = False

    def _chunk_paths(self, *, wdid: str) -> list[Path]:
        self._ensure_index()
        return self._wdid_to_paths.get(wdid, [])

    def _ensure_index(self) -> None:
        if self._index_built:
            return

        pattern = f'*_{self._lang}_*.npy'
        wdid_to_paths: dict[str, list[Path]] = defaultdict(list)
        sample_path: Path | None = None
        for path in sorted(self._root_dir.glob(pattern)):
            if sample_path is None:
                sample_path = path
            stem = path.stem
            split_suffix = f'_{self._lang}_'
            if split_suffix not in stem:
                continue
            wdid = stem.split(split_suffix, maxsplit=1)[0]
            if not wdid:
                continue
            wdid_to_paths[wdid].append(path)

        self._wdid_to_paths = dict(wdid_to_paths)
        self._sample_path = sample_path
        self._index_built = True
        LOGGER.info(
            'Indexed entity embedding files once: entities=%s files=%s',
            len(self._wdid_to_paths),
            sum(len(paths) for paths in self._wdid_to_paths.values()),
        )

    def get_entity_embedding(self, *, wdid: str) -> np.ndarray | None:
        """
        Return averaged chunk embedding for one entity.

        :param wdid: Wikidata ID.
        :return: Averaged entity embedding or ``None`` when unavailable.
        """
        if wdid in self._cache:
            return self._cache[wdid]

        chunk_paths = self._chunk_paths(wdid=wdid)
        if not chunk_paths:
            self._cache[wdid] = None
            return None

        chunks = [np.asarray(np.load(path), dtype=np.float32) for path in chunk_paths]
        embedding = np.mean(np.vstack(chunks), axis=0, dtype=np.float32)
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
                f'No entity embeddings found in {self._root_dir} for language "{self._lang}". '
                'Expected files matching pattern "*_{lang}_*.npy".'
            )
        sample = np.asarray(np.load(self._sample_path), dtype=np.float32)
        LOGGER.info('Entity embedding dimension inferred as %s from %s', sample.shape[0], self._sample_path.name)
        return int(sample.shape[0])

