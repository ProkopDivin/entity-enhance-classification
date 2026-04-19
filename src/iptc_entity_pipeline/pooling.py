"""Entity pooling strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from iptc_entity_pipeline.entity_embeddings import EntityEmbeddingStore
from iptc_entity_pipeline.data_loading import DocWithEntities
import numpy as np

from iptc_entity_pipeline.data_loading import get_doc_wdid_mention_counts, get_doc_wdids, get_doc_weighted_wdids


def _unit_weighted_wdids(*, doc: DocWithEntities) -> list[tuple[str, float]]:
    """Return wdIds with unit weights."""
    return [(wd_id, 1.0) for wd_id in get_doc_wdids(doc)]


def _validate_weight_alignment(
    *,
    class_name: str,
    entity_embeddings: Sequence[np.ndarray],
    weights: Sequence[float] | None,
) -> np.ndarray:
    """Validate weights for weighted pooling and return NumPy view."""
    if weights is None:
        raise ValueError(f'{class_name} requires weights')
    if len(entity_embeddings) != len(weights):
        raise ValueError('entity_embeddings and weights must have the same length')
    return np.asarray(weights, dtype=np.float32)


def _weighted_sum(
    *,
    entity_embeddings: Sequence[np.ndarray],
    weights_arr: np.ndarray,
) -> np.ndarray:
    """Compute weighted sum of entity embeddings."""
    return np.sum(np.vstack(entity_embeddings) * weights_arr[:, np.newaxis], axis=0, dtype=np.float32)


@dataclass(frozen=True)
class EntityPoolingResult:
    """Pooled vector with per-document entity-linking diagnostics."""

    pooled_embedding: np.ndarray
    requested_wdids: tuple[str, ...]
    missing_wdids: tuple[str, ...]
    found_embeddings: int
    missing_embeddings: int


class EntityPoolingStrategy(ABC):
    """Abstract entity pooling interface."""

    @abstractmethod
    def _get_weighted_wdids(self, *, doc: DocWithEntities) -> list[tuple[str, float]]:
        """Return wdIds and their weights for one document."""

    @abstractmethod
    def _pool_embeddings(
        self,
        *,
        entity_embeddings: Sequence[np.ndarray],
        embedding_dim: int,
        weights: Sequence[float] | None = None,
    ) -> np.ndarray:
        """Pool vectors already loaded for one document."""

    def pool(
        self,
        *,
        doc: DocWithEntities,
        entity_embedding_store: EntityEmbeddingStore,
        embedding_dim: int,
    ) -> EntityPoolingResult:
        """
        Build pooled embedding from one document.

        :param doc: Corpus document with attached linked entities.
        :param entity_embedding_store: Entity embedding store used to resolve wdId vectors.
        :param embedding_dim: Expected pooled vector dimensionality.
        :return: Pooled embedding and linking diagnostics.
        """
        weighted_wdids = self._get_weighted_wdids(doc=doc)
        requested_wdids = tuple(wd_id for wd_id, _ in weighted_wdids)

        entity_embeddings: list[np.ndarray] = []
        entity_weights: list[float] = []
        missing_wdids: list[str] = []
        for wdid, weight in weighted_wdids:
            entity_embedding = entity_embedding_store.get_entity_embedding(wdid=wdid)
            if entity_embedding is None:
                missing_wdids.append(wdid)
                continue
            entity_embeddings.append(entity_embedding)
            entity_weights.append(weight)

        pooled_embedding = self._pool_embeddings(
            entity_embeddings=entity_embeddings,
            embedding_dim=embedding_dim,
            weights=entity_weights,
        )
        return EntityPoolingResult(
            pooled_embedding=pooled_embedding,
            requested_wdids=requested_wdids,
            missing_wdids=tuple(missing_wdids),
            found_embeddings=len(entity_embeddings),
            missing_embeddings=len(missing_wdids),
        )


class SumEntityPooling(EntityPoolingStrategy):
    """Sum all entity vectors."""

    def _get_weighted_wdids(self, *, doc: DocWithEntities) -> list[tuple[str, float]]:
        return _unit_weighted_wdids(doc=doc)

    def _pool_embeddings(
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

    def _get_weighted_wdids(self, *, doc: DocWithEntities) -> list[tuple[str, float]]:
        return _unit_weighted_wdids(doc=doc)

    def _pool_embeddings(
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
    """Pool entity vectors using relevance-weighted normalized mean."""

    def _get_weighted_wdids(self, *, doc: DocWithEntities) -> list[tuple[str, float]]:
        return get_doc_weighted_wdids(doc)

    def _pool_embeddings(
        self,
        *,
        entity_embeddings: Sequence[np.ndarray],
        embedding_dim: int,
        weights: Sequence[float] | None = None,
    ) -> np.ndarray:
        if not entity_embeddings:
            return np.zeros(embedding_dim, dtype=np.float32)
        weights_arr = _validate_weight_alignment(
            class_name=self.__class__.__name__,
            entity_embeddings=entity_embeddings,
            weights=weights,
        )

        total_weight = float(np.sum(weights_arr, dtype=np.float32))
        if total_weight <= 0.0:
            return np.zeros(embedding_dim, dtype=np.float32)

        embeddings_matrix = np.vstack(entity_embeddings).astype(np.float32)
        weighted_embeddings = embeddings_matrix * weights_arr[:, np.newaxis]
        return (np.sum(weighted_embeddings, axis=0, dtype=np.float32) / total_weight).astype(np.float32)


class WeightedSumEntityPooling(EntityPoolingStrategy):
    """Pool entity vectors using relevance-weighted sum."""

    def _get_weighted_wdids(self, *, doc: DocWithEntities) -> list[tuple[str, float]]:
        return get_doc_weighted_wdids(doc)

    def _pool_embeddings(
        self,
        *,
        entity_embeddings: Sequence[np.ndarray],
        embedding_dim: int,
        weights: Sequence[float] | None = None,
    ) -> np.ndarray:
        if not entity_embeddings:
            return np.zeros(embedding_dim, dtype=np.float32)
        weights_arr = _validate_weight_alignment(
            class_name=self.__class__.__name__,
            entity_embeddings=entity_embeddings,
            weights=weights,
        )
        return _weighted_sum(entity_embeddings=entity_embeddings, weights_arr=weights_arr)


class MentionWeightedSumEntityPooling(EntityPoolingStrategy):
    """Pool entity vectors using mention-count weighted sum."""

    def _get_weighted_wdids(self, *, doc: DocWithEntities) -> list[tuple[str, float]]:
        return get_doc_wdid_mention_counts(doc)

    def _pool_embeddings(
        self,
        *,
        entity_embeddings: Sequence[np.ndarray],
        embedding_dim: int,
        weights: Sequence[float] | None = None,
    ) -> np.ndarray:
        if not entity_embeddings:
            return np.zeros(embedding_dim, dtype=np.float32)
        weights_arr = _validate_weight_alignment(
            class_name=self.__class__.__name__,
            entity_embeddings=entity_embeddings,
            weights=weights,
        )
        return _weighted_sum(entity_embeddings=entity_embeddings, weights_arr=weights_arr)


class NoEntityPooling(EntityPoolingStrategy):
    """
    No pooling: return np.array with all entity embeddings. Shape: [len(entity_embeddings), embedding_dim]
    Can be used when attention is used for pooling.
    """

    def _get_weighted_wdids(self, *, doc: DocWithEntities) -> list[tuple[str, float]]:
        return _unit_weighted_wdids(doc=doc)

    def _pool_embeddings(
        self,
        *,
        entity_embeddings: Sequence[np.ndarray],
        embedding_dim: int,
        weights: Sequence[float] | None = None,
    ) -> np.ndarray:
        if not entity_embeddings:
            return np.zeros(embedding_dim, dtype=np.float32)
        return np.vstack(entity_embeddings)
