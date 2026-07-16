"""Tests for entity-aware feature building."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from iptc_entity_pipeline.data_loading import LinkedEntity
from iptc_entity_pipeline.feature_builder import FeatureBuilder
from iptc_entity_pipeline.pooling import MentionWeightedSumEntityPooling, NoEntityPooling, WeightedMeanEntityPooling, WeightedSumEntityPooling


class FakeArticleEmbeddingProvider:
    def get_embedding(self, *, article_id: str) -> np.ndarray:
        del article_id
        return np.array([10.0, 20.0], dtype=np.float32)


class FakeEntityEmbeddingStore:
    def __init__(self, *, train_mean: np.ndarray | None = None) -> None:
        self._embeddings = {
            'Q1': np.array([1.0, 0.0], dtype=np.float32),
            'Q2': np.array([0.0, 1.0], dtype=np.float32),
            'Q3': np.array([1.0, 1.0], dtype=np.float32),
        }
        self._train_mean = (
            np.asarray(train_mean, dtype=np.float32)
            if train_mean is not None
            else np.array([0.5, 0.5], dtype=np.float32)
        )

    def infer_embedding_dim(self) -> int:
        return 2

    def get_entity_embedding(self, *, wdid: str) -> np.ndarray | None:
        return self._embeddings.get(wdid)

    def get_train_mean_embedding(self) -> np.ndarray:
        return np.asarray(self._train_mean, dtype=np.float32)

    def indexed_file_count(self) -> int:
        return len(self._embeddings)


def test_feature_builder_uses_split_relevance_weighted_mean() -> None:
    doc = SimpleNamespace(
        id='a1',
        title='Title',
        lead='Lead',
        text='Body',
        entities=[
            LinkedEntity(gkb_id='g1', wd_ids=('Q1', 'Q2'), relevance=0.8),
            LinkedEntity(gkb_id='g2', wd_ids=('Q3',), relevance=0.2),
        ],
    )
    builder = FeatureBuilder(
        article_embedding_provider=FakeArticleEmbeddingProvider(),
        entity_embedding_store=FakeEntityEmbeddingStore(),
        pooling_strategy=WeightedMeanEntityPooling(),
    )

    result = builder.build_features(corpus=[doc])

    expected = np.array([[10.0, 20.0, 0.6, 0.6]], dtype=np.float32)
    np.testing.assert_array_almost_equal(result, expected)


def test_feature_builder_sums_article_and_entity_embeddings() -> None:
    doc = SimpleNamespace(
        id='a_sum',
        title='Title',
        lead='Lead',
        text='Body',
        entities=[
            LinkedEntity(gkb_id='g1', wd_ids=('Q1', 'Q2'), relevance=0.8),
            LinkedEntity(gkb_id='g2', wd_ids=('Q3',), relevance=0.2),
        ],
    )
    builder = FeatureBuilder(
        article_embedding_provider=FakeArticleEmbeddingProvider(),
        entity_embedding_store=FakeEntityEmbeddingStore(),
        pooling_strategy=WeightedMeanEntityPooling(),
        combine_method='sum',
    )

    result = builder.build_features(corpus=[doc])

    expected = np.array([[10.6, 20.6]], dtype=np.float32)
    np.testing.assert_array_almost_equal(result, expected)


def test_feature_builder_uses_mention_counts_for_weighted_sum() -> None:
    doc = SimpleNamespace(
        id='a2',
        title='Title',
        lead='Lead',
        text='Body',
        entities=[
            LinkedEntity(gkb_id='g1', wd_ids=('Q1',), relevance=0.9),
            LinkedEntity(gkb_id='g2', wd_ids=('Q1',), relevance=0.1),
            LinkedEntity(gkb_id='g3', wd_ids=('Q2',), relevance=0.2),
        ],
    )
    builder = FeatureBuilder(
        article_embedding_provider=FakeArticleEmbeddingProvider(),
        entity_embedding_store=FakeEntityEmbeddingStore(),
        pooling_strategy=MentionWeightedSumEntityPooling(),
    )

    result = builder.build_features(corpus=[doc])

    expected = np.array([[10.0, 20.0, 2.0, 1.0]], dtype=np.float32)
    np.testing.assert_array_almost_equal(result, expected)


def test_feature_builder_uses_relevance_split_for_weighted_sum() -> None:
    doc = SimpleNamespace(
        id='a3',
        title='Title',
        lead='Lead',
        text='Body',
        entities=[
            LinkedEntity(gkb_id='g1', wd_ids=('Q1',), relevance=0.9),
            LinkedEntity(gkb_id='g2', wd_ids=('Q1',), relevance=0.1),
            LinkedEntity(gkb_id='g3', wd_ids=('Q2',), relevance=0.2),
        ],
    )
    builder = FeatureBuilder(
        article_embedding_provider=FakeArticleEmbeddingProvider(),
        entity_embedding_store=FakeEntityEmbeddingStore(),
        pooling_strategy=WeightedSumEntityPooling(),
    )

    result = builder.build_features(corpus=[doc])

    expected = np.array([[10.0, 20.0, 1.0, 0.2]], dtype=np.float32)
    np.testing.assert_array_almost_equal(result, expected)


def test_ragged_feature_builder_uses_train_mean_when_no_entity_hits() -> None:
    train_mean = np.array([0.3, 0.7], dtype=np.float32)
    doc = SimpleNamespace(
        id='a4',
        title='Title',
        lead='Lead',
        text='Body',
        entities=[],
    )
    builder = FeatureBuilder(
        article_embedding_provider=FakeArticleEmbeddingProvider(),
        entity_embedding_store=FakeEntityEmbeddingStore(train_mean=train_mean),
        pooling_strategy=NoEntityPooling(),
    )
    result = builder.build_ragged_features(corpus=[doc])
    np.testing.assert_array_almost_equal(result.entity_matrices[0], train_mean.reshape(1, -1))
    np.testing.assert_array_almost_equal(result.article_matrix, np.array([[10.0, 20.0]], dtype=np.float32))


def test_feature_builder_reports_max_and_p99_found_embeddings_per_article() -> None:
    docs = [
        SimpleNamespace(
            id='a10',
            title='Title',
            lead='Lead',
            text='Body',
            entities=[LinkedEntity(gkb_id='g1', wd_ids=('Q1',), relevance=1.0)],
        ),
        SimpleNamespace(
            id='a11',
            title='Title',
            lead='Lead',
            text='Body',
            entities=[
                LinkedEntity(gkb_id='g1', wd_ids=('Q1',), relevance=1.0),
                LinkedEntity(gkb_id='g2', wd_ids=('Q2',), relevance=1.0),
                LinkedEntity(gkb_id='g3', wd_ids=('Q3',), relevance=1.0),
            ],
        ),
        SimpleNamespace(
            id='a12',
            title='Title',
            lead='Lead',
            text='Body',
            entities=[],
        ),
    ]
    builder = FeatureBuilder(
        article_embedding_provider=FakeArticleEmbeddingProvider(),
        entity_embedding_store=FakeEntityEmbeddingStore(),
        pooling_strategy=WeightedMeanEntityPooling(),
    )

    _, stats = builder.build_features(corpus=docs, return_stats=True)

    assert stats.max_found_embeddings_per_article == 3
    assert stats.p99_found_embeddings_per_article == 3
