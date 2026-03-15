"""Entity pooling strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class EntityPoolingStrategy(ABC):
    """Abstract entity pooling interface."""

    @abstractmethod
    def pool(self, *, entity_embeddings: Sequence[np.ndarray], embedding_dim: int) -> np.ndarray:
        """
        Pool entity embeddings to one article-level vector.

        :param entity_embeddings: Entity vectors for one article.
        :param embedding_dim: Expected output dimensionality.
        :return: Pooled vector.
        """


class SumEntityPooling(EntityPoolingStrategy):
    """v1 pooling: sum all entity vectors."""

    def pool(self, *, entity_embeddings: Sequence[np.ndarray], embedding_dim: int) -> np.ndarray:
        if not entity_embeddings:
            return np.zeros(embedding_dim, dtype=np.float32)
        return np.sum(np.vstack(entity_embeddings), axis=0, dtype=np.float32)


class NoEntityPooling(EntityPoolingStrategy):
    """
    No pooling: return np.array with all entity embeddings. Shape: [len(entity_embeddings), embedding_dim]
    Can be used when attention is used for pooling.
    """

    def pool(self, *, entity_embeddings: Sequence[np.ndarray], embedding_dim: int) -> np.ndarray:
        if not entity_embeddings:
            return np.zeros(embedding_dim, dtype=np.float32)
        return np.vstack(entity_embeddings)
