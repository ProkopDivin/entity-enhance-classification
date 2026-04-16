"""Entity pooling strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class EntityPoolingStrategy(ABC):
    """Abstract entity pooling interface."""

    @abstractmethod
    def pool(
        self,
        *,
        entity_embeddings: Sequence[np.ndarray],
        embedding_dim: int,
        weights: Sequence[float] | None = None,
    ) -> np.ndarray:
        """
        Pool entity embeddings to one article-level vector.

        :param entity_embeddings: Entity vectors for one article.
        :param embedding_dim: Expected output dimensionality.
        :param weights: Optional per-entity weights aligned with ``entity_embeddings``.
        :return: Pooled vector.
        """


class SumEntityPooling(EntityPoolingStrategy):
    """v1 pooling: sum all entity vectors."""

    def pool(
        self,
        *,
        entity_embeddings: Sequence[np.ndarray],
        embedding_dim: int,
        weights: Sequence[float] | None = None,
    ) -> np.ndarray:
        if not entity_embeddings:
            return np.zeros(embedding_dim, dtype=np.float32)
        return np.sum(np.vstack(entity_embeddings), axis=0, dtype=np.float32)


class MeanEntityPooling(EntityPoolingStrategy):
    """Pool entity vectors using unweighted arithmetic mean."""

    def pool(
        self,
        *,
        entity_embeddings: Sequence[np.ndarray],
        embedding_dim: int,
        weights: Sequence[float] | None = None,
    ) -> np.ndarray:
        if not entity_embeddings:
            return np.zeros(embedding_dim, dtype=np.float32)
        return np.mean(np.vstack(entity_embeddings), axis=0, dtype=np.float32)


class WeightedMeanEntityPooling(EntityPoolingStrategy):
    """Pool entity vectors using a normalized weighted mean."""

    def pool(
        self,
        *,
        entity_embeddings: Sequence[np.ndarray],
        embedding_dim: int,
        weights: Sequence[float] | None = None,
    ) -> np.ndarray:
        if not entity_embeddings:
            return np.zeros(embedding_dim, dtype=np.float32)
        if weights is None:
            raise ValueError('WeightedMeanEntityPooling requires weights')
        if len(entity_embeddings) != len(weights):
            raise ValueError('entity_embeddings and weights must have the same length')

        weights_arr = np.asarray(weights, dtype=np.float32)
        total_weight = float(np.sum(weights_arr, dtype=np.float32))
        if total_weight <= 0.0:
            return np.zeros(embedding_dim, dtype=np.float32)

        embeddings_matrix = np.vstack(entity_embeddings).astype(np.float32)
        weighted_embeddings = embeddings_matrix * weights_arr[:, np.newaxis]
        return (np.sum(weighted_embeddings, axis=0, dtype=np.float32) / total_weight).astype(np.float32)

class WeightedSumEntityPooling(EntityPoolingStrategy):
    """Pool entity vectors using a weighted sum."""

    def pool(
        self,
        *,
        entity_embeddings: Sequence[np.ndarray],
        embedding_dim: int,
        weights: Sequence[float] | None = None,
    ) -> np.ndarray:
        if not entity_embeddings:
            return np.zeros(embedding_dim, dtype=np.float32)
        if weights is None:
            raise ValueError('WeightedSumEntityPooling requires weights')
        if len(entity_embeddings) != len(weights):
            raise ValueError('entity_embeddings and weights must have the same length')
        
        weights_arr = np.asarray(weights, dtype=np.float32)
        return np.sum(np.vstack(entity_embeddings) * weights_arr[:, np.newaxis], axis=0, dtype=np.float32)
    
    
class NoEntityPooling(EntityPoolingStrategy):
    """
    No pooling: return np.array with all entity embeddings. Shape: [len(entity_embeddings), embedding_dim]
    Can be used when attention is used for pooling.
    """

    def pool(
        self,
        *,
        entity_embeddings: Sequence[np.ndarray],
        embedding_dim: int,
        weights: Sequence[float] | None = None,
    ) -> np.ndarray:
        if not entity_embeddings:
            return np.zeros(embedding_dim, dtype=np.float32)
        return np.vstack(entity_embeddings)
