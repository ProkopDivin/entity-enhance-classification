"""Tests for entity pooling strategies."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from iptc_entity_pipeline.pooling import (
    MeanEntityPooling,
    MentionWeightedMeanEntityPooling,
    MentionWeightedSumEntityPooling,
    NoEntityPooling,
    SumEntityPooling,
    WeightedMeanEntityPooling,
    WeightedSumEntityPooling,
)

DIM = 3


class FakeEntityEmbeddingStore:
    def __init__(
        self,
        *,
        embeddings: dict[str, np.ndarray],
        train_mean: np.ndarray | None = None,
    ) -> None:
        self._embeddings = embeddings
        self._train_mean = train_mean

    def get_entity_embedding(self, *, wdid: str) -> np.ndarray | None:
        return self._embeddings.get(wdid)

    def get_train_mean_embedding(self) -> np.ndarray:
        if self._train_mean is None:
            raise RuntimeError('Train corpus entity mean is not computed')
        return np.asarray(self._train_mean, dtype=np.float32)


def test_sum_pooling() -> None:
    doc = SimpleNamespace(entities=[
        SimpleNamespace(wd_ids=('Q1',)),
        SimpleNamespace(wd_ids=('Q2',)),
    ])
    store = FakeEntityEmbeddingStore(embeddings={
        'Q1': np.array([1.0, 0.0, 0.0], dtype=np.float32),
        'Q2': np.array([0.0, 2.0, 0.0], dtype=np.float32),
    })
    result = SumEntityPooling().pool(doc=doc, entity_embedding_store=store, embedding_dim=DIM)
    np.testing.assert_array_almost_equal(result.pooled_embedding, np.array([1.0, 2.0, 0.0], dtype=np.float32))
    assert result.found_embeddings == 2
    assert result.missing_embeddings == 0


def test_mean_pooling() -> None:
    doc = SimpleNamespace(entities=[
        SimpleNamespace(wd_ids=('Q1',)),
        SimpleNamespace(wd_ids=('Q2',)),
    ])
    store = FakeEntityEmbeddingStore(embeddings={
        'Q1': np.array([1.0, 0.0, 0.0], dtype=np.float32),
        'Q2': np.array([0.0, 2.0, 0.0], dtype=np.float32),
    })
    result = MeanEntityPooling().pool(doc=doc, entity_embedding_store=store, embedding_dim=DIM)
    np.testing.assert_array_almost_equal(result.pooled_embedding, np.array([0.5, 1.0, 0.0], dtype=np.float32))


def test_no_pooling() -> None:
    """NoEntityPooling must vstack entity vectors in requested order without aggregation."""
    q1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    q2 = np.array([0.0, 2.0, 0.0], dtype=np.float32)
    doc = SimpleNamespace(entities=[
        SimpleNamespace(wd_ids=('Q1',)),
        SimpleNamespace(wd_ids=('Q2',)),
    ])
    store = FakeEntityEmbeddingStore(embeddings={'Q1': q1, 'Q2': q2})

    result = NoEntityPooling().pool(doc=doc, entity_embedding_store=store, embedding_dim=DIM)

    assert result.pooled_embedding.shape == (2, DIM)
    assert result.pooled_embedding.dtype == np.float32
    np.testing.assert_array_almost_equal(result.pooled_embedding[0], q1)
    np.testing.assert_array_almost_equal(result.pooled_embedding[1], q2)
    assert result.found_embeddings == 2
    assert result.missing_embeddings == 0
    assert result.requested_wdids == ('Q1', 'Q2')
    assert result.missing_wdids == ()


def test_weighted_mean_relevance_pooling() -> None:
    doc = SimpleNamespace(entities=[
        SimpleNamespace(wd_ids=('Q1', 'Q2'), relevance=0.8),
        SimpleNamespace(wd_ids=('Q3',), relevance=0.2),
    ])
    store = FakeEntityEmbeddingStore(embeddings={
        'Q1': np.array([1.0, 0.0, 0.0], dtype=np.float32),
        'Q2': np.array([0.0, 2.0, 0.0], dtype=np.float32),
        'Q3': np.array([0.0, 0.0, 4.0], dtype=np.float32),
    })
    result = WeightedMeanEntityPooling().pool(doc=doc, entity_embedding_store=store, embedding_dim=DIM)
    np.testing.assert_array_almost_equal(result.pooled_embedding, np.array([0.4, 0.8, 0.8], dtype=np.float32))


def test_mention_weighted_mean_pooling_uses_doc_mentions() -> None:
    doc = SimpleNamespace(entities=[
        SimpleNamespace(wd_ids=('Q1',), mention_count=3),
        SimpleNamespace(wd_ids=('Q1', 'Q2'), mention_count=2),
    ])
    store = FakeEntityEmbeddingStore(embeddings={
        'Q1': np.array([1.0, 0.0, 0.0], dtype=np.float32),
        'Q2': np.array([0.0, 2.0, 0.0], dtype=np.float32),
    })
    result = MentionWeightedMeanEntityPooling().pool(doc=doc, entity_embedding_store=store, embedding_dim=DIM)
    np.testing.assert_array_almost_equal(result.pooled_embedding, np.array([0.8, 0.4, 0.0], dtype=np.float32))


def test_weighted_sum_pooling_uses_relevance_split() -> None:
    doc = SimpleNamespace(entities=[
        SimpleNamespace(wd_ids=('Q1',), relevance=0.5),
        SimpleNamespace(wd_ids=('Q2',), relevance=1.5),
    ])
    store = FakeEntityEmbeddingStore(embeddings={
        'Q1': np.array([1.0, 0.0, 0.0], dtype=np.float32),
        'Q2': np.array([0.0, 2.0, 0.0], dtype=np.float32),
    })
    result = WeightedSumEntityPooling().pool(doc=doc, entity_embedding_store=store, embedding_dim=DIM)
    np.testing.assert_array_almost_equal(result.pooled_embedding, np.array([0.5, 3.0, 0.0], dtype=np.float32))


def test_mention_weighted_sum_pooling_uses_doc_mentions() -> None:
    doc = SimpleNamespace(entities=[
        SimpleNamespace(wd_ids=('Q1',), mention_count=3),
        SimpleNamespace(wd_ids=('Q1', 'Q2'), mention_count=2),
    ])
    store = FakeEntityEmbeddingStore(embeddings={
        'Q1': np.array([1.0, 0.0, 0.0], dtype=np.float32),
        'Q2': np.array([0.0, 2.0, 0.0], dtype=np.float32),
    })
    result = MentionWeightedSumEntityPooling().pool(doc=doc, entity_embedding_store=store, embedding_dim=DIM)
    np.testing.assert_array_almost_equal(result.pooled_embedding, np.array([4.0, 2.0, 0.0], dtype=np.float32))


def test_pooling_uses_train_mean_when_no_entity_hits() -> None:
    doc = SimpleNamespace(entities=[SimpleNamespace(wd_ids=('QX',), relevance=1.0, mention_count=1)])
    train_mean = np.array([0.2, 0.4, 0.6], dtype=np.float32)
    store = FakeEntityEmbeddingStore(embeddings={}, train_mean=train_mean)
    for pooling_cls in (
        MeanEntityPooling,
        SumEntityPooling,
        WeightedMeanEntityPooling,
        WeightedSumEntityPooling,
        MentionWeightedSumEntityPooling,
        MentionWeightedMeanEntityPooling,
        NoEntityPooling,
    ):
        result = pooling_cls().pool(doc=doc, entity_embedding_store=store, embedding_dim=DIM)
        np.testing.assert_array_almost_equal(result.pooled_embedding, train_mean)
        assert result.found_embeddings == 0
        assert result.missing_embeddings == 1


def test_pooling_reports_missing_wdids() -> None:
    doc = SimpleNamespace(entities=[
        SimpleNamespace(wd_ids=('Q1',), relevance=1.0),
        SimpleNamespace(wd_ids=('QX',), relevance=1.0),
    ])
    store = FakeEntityEmbeddingStore(embeddings={'Q1': np.array([1.0, 0.0, 0.0], dtype=np.float32)})
    result = WeightedSumEntityPooling().pool(doc=doc, entity_embedding_store=store, embedding_dim=DIM)
    assert result.requested_wdids == ('Q1', 'QX')
    assert result.missing_wdids == ('QX',)
    assert result.found_embeddings == 1
    assert result.missing_embeddings == 1
