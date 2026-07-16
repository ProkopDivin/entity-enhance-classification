"""Tests for ArticleEmbeddingProvider cache behaviour."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from iptc_entity_pipeline.article_embeddings import ArticleEmbeddingProvider


def test_get_embedding_cache_hit(tmp_path: Path) -> None:
    expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    np.save(tmp_path / 'art1.npy', expected)

    provider = ArticleEmbeddingProvider(
        embeddings_dir=str(tmp_path),
        model_name='test-model',
    )
    result = provider.get_embedding(article_id='art1')
    np.testing.assert_array_almost_equal(result, expected)


def test_get_embedding_cache_miss_raises(tmp_path: Path) -> None:
    provider = ArticleEmbeddingProvider(
        embeddings_dir=str(tmp_path),
        model_name='test-model',
    )
    with pytest.raises(FileNotFoundError, match='Run prepare_embeddings'):
        provider.get_embedding(article_id='missing')
