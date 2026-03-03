"""Entity embedding loading and per-entity chunk averaging."""

from __future__ import annotations

import logging
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

    def _chunk_paths(self, *, wdid: str) -> list[Path]:
        pattern = f'{wdid}_{self._lang}_*.npy'
        return sorted(self._root_dir.glob(pattern))

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
        sample_paths = sorted(self._root_dir.glob(f'*_{self._lang}_*.npy'))
        if not sample_paths:
            raise FileNotFoundError(
                f'No entity embeddings found in {self._root_dir} for language "{self._lang}". '
                'Expected files matching pattern "*_{lang}_*.npy".'
            )
        sample = np.asarray(np.load(sample_paths[0]), dtype=np.float32)
        LOGGER.info('Entity embedding dimension inferred as %s from %s', sample.shape[0], sample_paths[0].name)
        return int(sample.shape[0])

