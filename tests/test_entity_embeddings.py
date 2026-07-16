"""Tests for EntityEmbeddingStore."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from iptc_entity_pipeline.entity_embeddings import EntityEmbeddingStore


def test_entity_embedding_store(tmp_path: Path) -> None:
    np.save(tmp_path / 'Q123_en_0.npy', np.array([1.0, 0.0, 0.0], dtype=np.float32))
    np.save(tmp_path / 'Q123_en_1.npy', np.array([0.0, 1.0, 0.0], dtype=np.float32))
    np.save(tmp_path / 'Q456_en_0.npy', np.array([0.0, 0.0, 1.0], dtype=np.float32))

    store = EntityEmbeddingStore(root_dir=str(tmp_path), langs=('en',))

    emb = store.get_entity_embedding(wdid='Q123')
    np.testing.assert_array_almost_equal(emb, [0.5, 0.5, 0.0])

    emb2 = store.get_entity_embedding(wdid='Q456')
    np.testing.assert_array_almost_equal(emb2, [0.0, 0.0, 1.0])

    assert store.get_entity_embedding(wdid='Q999') is None
    assert store.get_entity_embedding(wdid='Q123') is emb
    assert store.infer_embedding_dim() == 3


def test_infer_dim_no_files(tmp_path: Path) -> None:
    store = EntityEmbeddingStore(root_dir=str(tmp_path), langs=('en',))
    with pytest.raises(FileNotFoundError, match='No entity embeddings'):
        store.infer_embedding_dim()


def test_entity_embedding_store_averages_across_languages(tmp_path: Path) -> None:
    np.save(tmp_path / 'Q100_en_0.npy', np.array([1.0, 0.0], dtype=np.float32))
    np.save(tmp_path / 'Q100_en_1.npy', np.array([3.0, 0.0], dtype=np.float32))
    np.save(tmp_path / 'Q100_de_0.npy', np.array([0.0, 4.0], dtype=np.float32))
    np.save(tmp_path / 'Q200_de_0.npy', np.array([2.0, 2.0], dtype=np.float32))

    store = EntityEmbeddingStore(root_dir=str(tmp_path), langs=('en', 'de'), lang_mode='average')

    emb_q100 = store.get_entity_embedding(wdid='Q100')
    np.testing.assert_array_almost_equal(emb_q100, [4.0 / 3.0, 4.0 / 3.0])

    emb_q200 = store.get_entity_embedding(wdid='Q200')
    np.testing.assert_array_almost_equal(emb_q200, [2.0, 2.0])


def test_entity_embedding_store_fallback_across_languages(tmp_path: Path) -> None:
    np.save(tmp_path / 'Q100_en_0.npy', np.array([1.0, 0.0], dtype=np.float32))
    np.save(tmp_path / 'Q100_en_1.npy', np.array([3.0, 0.0], dtype=np.float32))
    np.save(tmp_path / 'Q100_de_0.npy', np.array([0.0, 4.0], dtype=np.float32))
    np.save(tmp_path / 'Q200_de_0.npy', np.array([2.0, 2.0], dtype=np.float32))

    store = EntityEmbeddingStore(root_dir=str(tmp_path), langs=('en', 'de'), lang_mode='fallback')

    emb_q100 = store.get_entity_embedding(wdid='Q100')
    np.testing.assert_array_almost_equal(emb_q100, [2.0, 0.0])

    emb_q200 = store.get_entity_embedding(wdid='Q200')
    np.testing.assert_array_almost_equal(emb_q200, [2.0, 2.0])


def test_entity_embedding_store_fallback_respects_language_order(tmp_path: Path) -> None:
    np.save(tmp_path / 'Q100_en_0.npy', np.array([1.0, 0.0], dtype=np.float32))
    np.save(tmp_path / 'Q100_nl_0.npy', np.array([0.0, 2.0], dtype=np.float32))

    store_nl_en = EntityEmbeddingStore(root_dir=str(tmp_path), langs=('nl', 'en'), lang_mode='fallback')
    np.testing.assert_array_almost_equal(
        store_nl_en.get_entity_embedding(wdid='Q100'),
        [0.0, 2.0],
    )

    store_en_nl = EntityEmbeddingStore(root_dir=str(tmp_path), langs=('en', 'nl'), lang_mode='fallback')
    np.testing.assert_array_almost_equal(
        store_en_nl.get_entity_embedding(wdid='Q100'),
        [1.0, 0.0],
    )


def test_indexed_file_count_respects_language_filter(tmp_path: Path) -> None:
    np.save(tmp_path / 'Q1_en_0.npy', np.array([1.0], dtype=np.float32))
    np.save(tmp_path / 'Q1_en_1.npy', np.array([2.0], dtype=np.float32))
    np.save(tmp_path / 'Q1_nl_0.npy', np.array([3.0], dtype=np.float32))
    np.save(tmp_path / 'Q2_de_0.npy', np.array([4.0], dtype=np.float32))

    store_en = EntityEmbeddingStore(root_dir=str(tmp_path), langs=('en',))
    assert store_en.indexed_file_count() == 2

    store_en_nl = EntityEmbeddingStore(root_dir=str(tmp_path), langs=('en', 'nl'))
    assert store_en_nl.indexed_file_count() == 3


def test_compute_train_mean_from_corpus(tmp_path: Path) -> None:
    np.save(tmp_path / 'Q1_en_0.npy', np.array([1.0, 0.0], dtype=np.float32))
    np.save(tmp_path / 'Q2_en_0.npy', np.array([0.0, 2.0], dtype=np.float32))
    store = EntityEmbeddingStore(root_dir=str(tmp_path), langs=('en',))
    corpus = [
        SimpleNamespace(entities=[SimpleNamespace(wd_ids=('Q1',))]),
        SimpleNamespace(entities=[SimpleNamespace(wd_ids=('Q2',))]),
    ]
    store.compute_train_mean_from_corpus(corpus=corpus)
    np.testing.assert_array_almost_equal(store.get_train_mean_embedding(), [0.5, 1.0])


def test_compute_train_mean_from_corpus_fails_without_resolvable_embeddings(tmp_path: Path) -> None:
    store = EntityEmbeddingStore(root_dir=str(tmp_path), langs=('en',))
    corpus = [SimpleNamespace(entities=[SimpleNamespace(wd_ids=('Q999',))])]
    with pytest.raises(ValueError, match='No resolvable entity embeddings in train corpus'):
        store.compute_train_mean_from_corpus(corpus=corpus)


def test_clear_cache_drops_vectors_and_index_but_keeps_train_mean(tmp_path: Path) -> None:
    np.save(tmp_path / 'Q1_en_0.npy', np.array([1.0, 0.0], dtype=np.float32))
    np.save(tmp_path / 'Q2_en_0.npy', np.array([0.0, 2.0], dtype=np.float32))
    store = EntityEmbeddingStore(root_dir=str(tmp_path), langs=('en',))

    assert store.get_entity_embedding(wdid='Q1') is not None
    assert store.indexed_file_count() == 2
    corpus = [SimpleNamespace(entities=[SimpleNamespace(wd_ids=('Q1', 'Q2'))])]
    store.compute_train_mean_from_corpus(corpus=corpus)

    assert store._cache, 'precondition: cache populated before clear_cache'
    assert store._wdid_lang_to_paths, 'precondition: path index populated before clear_cache'

    store.clear_cache()

    assert store._cache == {}
    assert store._wdid_lang_to_paths == {}
    assert store._index_built is False
    np.testing.assert_array_almost_equal(store.get_train_mean_embedding(), [0.5, 1.0])
    assert store.get_entity_embedding(wdid='Q1') is not None  # rebuild on demand
    assert store.indexed_file_count() == 2
