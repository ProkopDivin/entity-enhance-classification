"""Construct EmbeddingDataset objects from precomputed features."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _require_embedding_dataset_cls():
    try:
        from geneea.catlib.vec.dataset import EmbeddingDataset  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise ImportError(
            'Missing geneea dependency "geneea.catlib". Install internal packages to build EmbeddingDataset.'
        ) from exc
    return EmbeddingDataset


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

