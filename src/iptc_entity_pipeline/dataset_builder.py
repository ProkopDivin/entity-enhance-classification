"""Construct, slice, and merge EmbeddingDataset objects."""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


def _require_embedding_dataset_cls():
    try:
        from geneea.catlib.vec.dataset import EmbeddingDataset  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise ImportError(
            'Missing geneea dependency "geneea.catlib". Install internal packages to build EmbeddingDataset.'
        ) from exc
    return EmbeddingDataset


def _require_corpus_cls() -> Any:
    from geneea.catlib.data import Corpus

    return Corpus


def to_numpy_array(*, matrix_like: Any) -> np.ndarray:
    """Convert ndarray-like object or tensor to numpy array."""
    if hasattr(matrix_like, 'detach') and hasattr(matrix_like, 'cpu'):
        return matrix_like.detach().cpu().numpy()
    return np.asarray(matrix_like)


def build_multilabel_targets(*, corpus: Any) -> np.ndarray:
    """
    Convert corpus labels to a multi-hot matrix.

    :param corpus: Corpus with ``catList`` and docs containing ``cats``.
    :return: ``float32`` matrix with shape ``[docs, num_categories]``.
    """
    cats = list(corpus.catList)
    cat_to_idx = {cat: idx for idx, cat in enumerate(cats)}
    y = np.zeros((len(corpus), len(cats)), dtype=np.float32)

    for row, doc in enumerate(corpus):
        for cat in doc.cats:
            idx = cat_to_idx.get(cat)
            if idx is not None:
                y[row, idx] = 1.0
    return y


def build_embedding_dataset(*, corpus: Any, x_matrix: np.ndarray) -> Any:
    """
    Wrap numpy features + derived targets into ``EmbeddingDataset``.

    :param corpus: Corpus with category labels.
    :param x_matrix: Feature matrix.
    :return: ``EmbeddingDataset`` instance.
    """
    EmbeddingDataset = _require_embedding_dataset_cls()
    y_matrix = build_multilabel_targets(corpus=corpus)
    x_tensor = torch.as_tensor(x_matrix, dtype=torch.float32)
    y_tensor = torch.as_tensor(y_matrix, dtype=torch.float32)
    return EmbeddingDataset(corpus, x_tensor, y_tensor)


def _set_corpus_cat_list(*, corpus: Any, cat_list: Sequence[str]) -> None:
    try:
        setattr(corpus, 'catList', list(cat_list))
    except Exception:
        LOGGER.warning('Unable to set corpus.catList explicitly, keeping corpus defaults')


def _build_dataset_with_targets(
    *,
    corpus: Any,
    x_matrix: np.ndarray,
    y_matrix: np.ndarray,
    cat_list: Sequence[str] | None = None,
) -> Any:
    embedding_dataset_cls = _require_embedding_dataset_cls()
    if cat_list is not None:
        _set_corpus_cat_list(corpus=corpus, cat_list=cat_list)
    x_tensor = torch.as_tensor(x_matrix, dtype=torch.float32)
    y_tensor = torch.as_tensor(y_matrix, dtype=torch.float32)
    return embedding_dataset_cls(corpus, x_tensor, y_tensor)


def merge_datasets(*, left_data: Any, right_data: Any) -> Any:
    """Merge two embedding datasets by concatenating corpus docs and feature matrices."""
    corpus_cls = _require_corpus_cls()
    merged_docs = list(left_data.corpus) + list(right_data.corpus)
    merged_corpus = corpus_cls(doc for doc in merged_docs)
    merged_x = np.vstack([to_numpy_array(matrix_like=left_data.X), to_numpy_array(matrix_like=right_data.X)])
    if hasattr(left_data, 'Y') and hasattr(right_data, 'Y'):
        merged_y = np.vstack([to_numpy_array(matrix_like=left_data.Y), to_numpy_array(matrix_like=right_data.Y)])
        merged_cat_list = list(left_data.corpus.catList) if hasattr(left_data.corpus, 'catList') else None
        return _build_dataset_with_targets(
            corpus=merged_corpus,
            x_matrix=merged_x,
            y_matrix=merged_y,
            cat_list=merged_cat_list,
        )
    return build_embedding_dataset(corpus=merged_corpus, x_matrix=merged_x)


def slice_dataset(*, dataset: Any, indices: Sequence[int]) -> Any:
    """Return dataset subset by explicit positional indices."""
    corpus_cls = _require_corpus_cls()
    docs = list(dataset.corpus)
    x_matrix = to_numpy_array(matrix_like=dataset.X)
    selected_docs = [docs[idx] for idx in indices]
    selected_x = x_matrix[np.asarray(indices, dtype=np.int64)]
    selected_corpus = corpus_cls(doc for doc in selected_docs)
    if hasattr(dataset, 'Y'):
        y_matrix = to_numpy_array(matrix_like=dataset.Y)
        selected_y = y_matrix[np.asarray(indices, dtype=np.int64)]
        cat_list = list(dataset.corpus.catList) if hasattr(dataset.corpus, 'catList') else None
        return _build_dataset_with_targets(
            corpus=selected_corpus,
            x_matrix=selected_x,
            y_matrix=selected_y,
            cat_list=cat_list,
        )
    return build_embedding_dataset(corpus=selected_corpus, x_matrix=selected_x)

