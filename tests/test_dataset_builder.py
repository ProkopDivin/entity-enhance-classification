"""Tests for dataset_utils utilities."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from geneea.catlib.data import Corpus, Doc
from geneea.catlib.vec.dataset import EmbeddingDataset
from iptc_entity_pipeline.dataset_builder import (
    EmbeddingSubset,
    RaggedEmbeddingDataset,
    RaggedEmbeddingSubset,
    slice_dataset,
    to_numpy_array,
)


@pytest.mark.parametrize('input_val', [
    np.array([1.0, 2.0]),
    torch.tensor([1.0, 2.0]),
    [1.0, 2.0],
])
def test_to_numpy_array(input_val) -> None:
    result = to_numpy_array(matrix_like=input_val)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_almost_equal(result, [1.0, 2.0])


def _make_dataset(*, n_docs: int = 6, dim: int = 3, n_cats: int = 4) -> EmbeddingDataset:
    docs = [Doc.of(id=f'd{i}', text=f'text {i}', cats=[f'c{i % n_cats}']) for i in range(n_docs)]
    corpus = Corpus(docs)
    corpus.catList = [f'c{i}' for i in range(n_cats)]
    corpus.catToIdx = {c: i for i, c in enumerate(corpus.catList)}
    x = torch.arange(n_docs * dim, dtype=torch.float32).reshape(n_docs, dim)
    y = torch.zeros((n_docs, n_cats), dtype=torch.float32)
    for row, doc in enumerate(docs):
        y[row, row % n_cats] = 1.0
    return EmbeddingDataset(corpus, x, y)


def test_slice_dataset_shares_base_storage_no_copy() -> None:
    ds = _make_dataset(n_docs=6, dim=3, n_cats=4)
    indices = [0, 2, 4]

    subset = slice_dataset(dataset=ds, indices=indices)

    assert isinstance(subset, EmbeddingSubset)
    assert subset._base_x.data_ptr() == ds.X.data_ptr()
    assert subset._base_y.data_ptr() == ds.Y.data_ptr()
    assert len(subset) == len(indices)
    assert subset.catList == ds.corpus.catList
    assert subset.catCnt == ds.corpus.catCnt


def test_slice_dataset_getitem_matches_base_rows() -> None:
    ds = _make_dataset(n_docs=6, dim=3, n_cats=4)
    indices = [5, 1, 3]
    subset = slice_dataset(dataset=ds, indices=indices)

    for offset, base_row in enumerate(indices):
        x_view, y_view = subset[offset]
        assert torch.equal(x_view, ds.X[base_row])
        assert torch.equal(y_view, ds.Y[base_row])


def test_slice_dataset_lazy_x_y_materialization() -> None:
    ds = _make_dataset(n_docs=6, dim=3, n_cats=4)
    indices = [0, 2, 4]
    subset = slice_dataset(dataset=ds, indices=indices)

    assert subset._x_cache is None
    assert subset._y_cache is None
    expected_x = ds.X.index_select(0, torch.as_tensor(indices, dtype=torch.long))
    assert torch.equal(subset.X, expected_x)
    assert subset._x_cache is subset.X
    assert subset._y_cache is None
    expected_y = ds.Y.index_select(0, torch.as_tensor(indices, dtype=torch.long))
    assert torch.equal(subset.Y, expected_y)


def test_ragged_spill_survives_multiple_pipeline_step_loads() -> None:
    """Simulate ClearML steps each unpickling build_dataset output with the same tmp_dir."""
    docs = [Doc.of(id='d0', text='t0', cats=['c0']), Doc.of(id='d1', text='t1', cats=['c1'])]
    corpus = Corpus(docs)
    corpus.catList = ['c0', 'c1']
    corpus.catToIdx = {c: i for i, c in enumerate(corpus.catList)}
    article_x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    entity_x = (
        torch.tensor([[0.1]], dtype=torch.float32),
        torch.tensor([[0.2], [0.3]], dtype=torch.float32),
    )
    y = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    dataset = RaggedEmbeddingDataset(corpus=corpus, article_x=article_x, entity_x=entity_x, y=y)
    dataset.cache_temporary()
    spill_path = dataset.tmp_dir
    assert spill_path is not None
    assert (spill_path / 'X.pt').exists()
    assert (spill_path / 'entity_X.pt').exists()

    dataset.load_temporary()
    assert dataset.X is not None
    assert (spill_path / 'X.pt').exists()

    train_best_view = RaggedEmbeddingDataset(corpus=corpus, article_x=article_x, entity_x=entity_x, y=y)
    train_best_view.X = None
    train_best_view.entity_X = None
    train_best_view.tmp_dir = spill_path
    train_best_view.load_temporary()
    assert train_best_view.X is not None
    train_best_view.cleanup_temporary()
    assert not spill_path.exists()


def test_slice_ragged_dataset_returns_view_without_copy() -> None:
    docs = [Doc.of(id='d0', text='t0', cats=['c0']), Doc.of(id='d1', text='t1', cats=['c1']), Doc.of(id='d2', text='t2', cats=['c0'])]
    corpus = Corpus(docs)
    corpus.catList = ['c0', 'c1']
    corpus.catToIdx = {c: i for i, c in enumerate(corpus.catList)}
    article_x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [7.0, 8.0]], dtype=torch.float32)
    entity_x = (
        torch.tensor([[0.1]], dtype=torch.float32),
        torch.tensor([[0.2], [0.3]], dtype=torch.float32),
        torch.tensor([[0.4], [0.5], [0.6]], dtype=torch.float32),
    )
    y = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
    dataset = RaggedEmbeddingDataset(corpus=corpus, article_x=article_x, entity_x=entity_x, y=y)

    subset = slice_dataset(dataset=dataset, indices=[2, 0])

    assert isinstance(subset, RaggedEmbeddingSubset)
    assert subset._base_x.data_ptr() == dataset.X.data_ptr()
    assert subset._base_y.data_ptr() == dataset.Y.data_ptr()
    assert subset.entity_X[0] is dataset.entity_X[2]
    assert subset.entity_X[1] is dataset.entity_X[0]
    features, labels = subset[0]
    assert torch.equal(features['article_embeddings'], dataset.X[2])
    assert torch.equal(features['entity_embeddings'], dataset.entity_X[2])
    assert torch.equal(labels, dataset.Y[2])


def test_slice_dataset_resliced_subset_composes_against_base() -> None:
    ds = _make_dataset(n_docs=6, dim=3, n_cats=4)
    first_indices = [0, 2, 4, 5]
    second_indices = [1, 3]

    first = slice_dataset(dataset=ds, indices=first_indices)
    second = slice_dataset(dataset=first, indices=second_indices)

    assert isinstance(second, EmbeddingSubset)
    assert second._base_x.data_ptr() == ds.X.data_ptr()
    expected_rows = [first_indices[i] for i in second_indices]
    expected_x = ds.X.index_select(0, torch.as_tensor(expected_rows, dtype=torch.long))
    assert torch.equal(second.X, expected_x)
    for offset, base_row in enumerate(expected_rows):
        x_view, y_view = second[offset]
        assert torch.equal(x_view, ds.X[base_row])
        assert torch.equal(y_view, ds.Y[base_row])
