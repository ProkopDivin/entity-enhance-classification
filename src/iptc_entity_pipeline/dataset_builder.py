"""Construct, slice, and merge EmbeddingDataset objects."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from geneea.catlib.data import Corpus
from geneea.catlib.vec.dataset import EmbeddingDataset
from torch.utils.data import Dataset

LOGGER = logging.getLogger(__name__)


class RaggedEmbeddingDataset(Dataset):
    """Dataset carrying article vectors and per-sample ragged entity vectors.

    Entity embeddings are stored as a single flat ``entity_flat`` buffer with
    a per-document row-pointer ``entity_offsets`` (``offsets[i]:offsets[i+1]``
    is doc ``i``'s slice). This packs the per-doc matrices into two
    contiguous tensors, so pickling/transport between pipeline steps is
    O(2 big blobs) instead of O(N small tensors).

    The same class also serves as a *view* over a base dataset: when
    ``doc_indices`` is provided, ``__len__``/``__getitem__`` and the lazy
    ``X``/``Y`` properties index through it while still sharing the four
    base buffers with the source dataset (mirrors :class:`EmbeddingSubset`).
    """

    def __init__(
        self,
        *,
        corpus: Any,
        article_x: torch.Tensor,
        entity_flat: torch.Tensor,
        entity_offsets: torch.Tensor,
        y: torch.Tensor,
        doc_indices: torch.Tensor | None = None,
        base_corpus: Any | None = None,
    ) -> None:
        if int(article_x.shape[0]) != int(y.shape[0]):
            raise ValueError('RaggedEmbeddingDataset requires equal lengths for article_x and y')
        if int(entity_offsets.shape[0]) != int(article_x.shape[0]) + 1:
            raise ValueError(
                'RaggedEmbeddingDataset requires entity_offsets of length article_x.shape[0] + 1'
            )
        if entity_flat.ndim != 2:
            raise ValueError('entity_flat must be a 2D tensor (total_entities, entity_dim)')
        if int(entity_offsets[-1].item()) != int(entity_flat.shape[0]):
            raise ValueError(
                'entity_offsets[-1] must equal entity_flat.shape[0] (total packed entities)'
            )
        self.corpus = corpus
        self._article_x_base = article_x
        self._entity_flat = entity_flat
        self._entity_offsets = entity_offsets
        self._y_base = y
        self._doc_indices = doc_indices
        self._base_corpus = base_corpus if base_corpus is not None else corpus
        self.entity_dim = int(entity_flat.shape[1])
        self.catList = list(corpus.catList) if hasattr(corpus, 'catList') else []
        self._x_cache: torch.Tensor | None = None
        self._y_cache: torch.Tensor | None = None

    def __len__(self) -> int:
        if self._doc_indices is not None:
            return int(self._doc_indices.shape[0])
        return int(self._article_x_base.shape[0])

    def _resolve_doc(self, idx: int) -> int:
        if self._doc_indices is not None:
            return int(self._doc_indices[idx])
        return int(idx)

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        doc = self._resolve_doc(idx)
        start = int(self._entity_offsets[doc])
        end = int(self._entity_offsets[doc + 1])
        return (
            {
                'article_embeddings': self._article_x_base[doc],
                'entity_embeddings': self._entity_flat[start:end],
            },
            self._y_base[doc],
        )

    @property
    def X(self) -> torch.Tensor:
        if self._doc_indices is None:
            return self._article_x_base
        if self._x_cache is None:
            self._x_cache = self._article_x_base.index_select(0, self._doc_indices)
        return self._x_cache

    @property
    def Y(self) -> torch.Tensor:
        if self._doc_indices is None:
            return self._y_base
        if self._y_cache is None:
            self._y_cache = self._y_base.index_select(0, self._doc_indices)
        return self._y_cache

    def saveEmbeds(self, file_name: str | Path) -> None:
        """
        Save article embeddings as TSV (docId, space-delimited vector elements).

        This preserves compatibility with artifact writers that expect the
        ``EmbeddingDataset.saveEmbeds`` interface.
        """
        with open(file_name, 'w', encoding='utf-8') as f:
            for i, doc in enumerate(self.corpus):
                doc_idx = self._resolve_doc(i)
                vector = self._article_x_base[doc_idx]
                vector_str = ' '.join(f'{v:.4g}' for v in vector)
                f.write(f'{doc.id}\t{vector_str}\n')

    @staticmethod
    def collate_fn(batch: Sequence[tuple[dict[str, torch.Tensor], torch.Tensor]]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Expose collate function for generic DataLoader call-sites."""
        return ragged_collate_fn(batch)


def ragged_collate_fn(batch: Sequence[tuple[dict[str, torch.Tensor], torch.Tensor]]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Collate ragged entity embeddings into a padded batch tensor."""
    features, labels = zip(*batch)
    article_batch = torch.stack([item['article_embeddings'] for item in features])
    entity_list = [item['entity_embeddings'] for item in features]
    max_entities = max((int(entities.shape[0]) for entities in entity_list), default=0)
    entity_dim = int(entity_list[0].shape[1]) if entity_list and entity_list[0].ndim == 2 else 0

    padded_entities = torch.zeros(
        (len(entity_list), max_entities, entity_dim),
        dtype=article_batch.dtype,
    )
    entity_mask = torch.zeros((len(entity_list), max_entities), dtype=torch.bool)
    for row_idx, entities in enumerate(entity_list):
        length = int(entities.shape[0])
        if length == 0:
            continue
        padded_entities[row_idx, :length, :] = entities
        entity_mask[row_idx, :length] = True

    labels_batch = torch.stack(list(labels))
    return {
        'article_embeddings': article_batch,
        'entity_embeddings': padded_entities,
        'entity_mask': entity_mask,
    }, labels_batch


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


def build_emb_data(*, corpus: Any, x_matrix: np.ndarray) -> Any:
    """
    Wrap numpy features + derived targets into ``EmbeddingDataset``.

    :param corpus: Corpus with category labels.
    :param x_matrix: Feature matrix.
    :return: ``EmbeddingDataset`` instance.
    """
    y_matrix = build_multilabel_targets(corpus=corpus)
    x_tensor = torch.as_tensor(x_matrix, dtype=torch.float32)
    y_tensor = torch.as_tensor(y_matrix, dtype=torch.float32)
    return EmbeddingDataset(corpus, x_tensor, y_tensor)


def build_ragged_emb_data(
    *,
    corpus: Any,
    article_matrix: np.ndarray,
    entity_flat: np.ndarray,
    entity_offsets: np.ndarray,
) -> RaggedEmbeddingDataset:
    """Build ragged embedding dataset for no-pooling mode from packed buffers."""
    y_matrix = build_multilabel_targets(corpus=corpus)
    return RaggedEmbeddingDataset(
        corpus=corpus,
        article_x=torch.from_numpy(np.ascontiguousarray(article_matrix, dtype=np.float32)),
        entity_flat=torch.from_numpy(np.ascontiguousarray(entity_flat, dtype=np.float32)),
        entity_offsets=torch.from_numpy(np.ascontiguousarray(entity_offsets, dtype=np.int64)),
        y=torch.from_numpy(np.ascontiguousarray(y_matrix, dtype=np.float32)),
    )

def _build_dataset_with_targets(
    *,
    corpus: Any,
    x_matrix: np.ndarray,
    y_matrix: np.ndarray,
    cat_list: Sequence[str] | None = None,
) -> Any:
    setattr(corpus, 'catList', list(cat_list)) # TODO: finf out if this is not stupid 
    x_tensor = torch.as_tensor(x_matrix, dtype=torch.float32)
    y_tensor = torch.as_tensor(y_matrix, dtype=torch.float32)
    return EmbeddingDataset(corpus, x_tensor, y_tensor)


def merge_datasets(*, left_data: Any, right_data: Any) -> Any:
    """Merge two embedding datasets by concatenating corpus docs and feature matrices."""
    if isinstance(left_data, RaggedEmbeddingDataset) or isinstance(right_data, RaggedEmbeddingDataset):
        raise NotImplementedError('merge_datasets is not implemented for RaggedEmbeddingDataset')

    merged_docs = list(left_data.corpus) + list(right_data.corpus)
    merged_corpus = Corpus(doc for doc in merged_docs)
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
    return build_emb_data(corpus=merged_corpus, x_matrix=merged_x)


class EmbeddingSubset(Dataset):
    """Lightweight view of an :class:`EmbeddingDataset` by positional indices.

    Stores a reference to the base ``X``/``Y`` tensors and an ``int64``
    index tensor, so per-fold CV splits don't pay the full ``X``/``Y``
    copy cost of the previous numpy round-trip implementation.

    Attributes ``X`` and ``Y`` are exposed as lazy cached properties: the
    materialized slice is only built on first attribute access (e.g. for
    a stratified OOF split), and never built at all when the dataset is
    only consumed via ``__getitem__``/``DataLoader`` and ``.corpus``.

    Re-slicing an :class:`EmbeddingSubset` composes indices against the
    original base tensors instead of chaining views.
    """

    def __init__(
        self,
        *,
        base_x: torch.Tensor,
        base_y: torch.Tensor,
        base_corpus: Any,
        indices: torch.Tensor,
        corpus: Any,
    ) -> None:
        self._base_x = base_x
        self._base_y = base_y
        self._base_corpus = base_corpus
        self._indices = indices
        self.corpus = corpus
        self._x_cache: torch.Tensor | None = None
        self._y_cache: torch.Tensor | None = None

    def __len__(self) -> int:
        return int(self._indices.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        base_idx = int(self._indices[idx])
        return self._base_x[base_idx], self._base_y[base_idx]

    @property
    def X(self) -> torch.Tensor:
        if self._x_cache is None:
            self._x_cache = self._base_x.index_select(0, self._indices)
        return self._x_cache

    @property
    def Y(self) -> torch.Tensor:
        if self._y_cache is None:
            self._y_cache = self._base_y.index_select(0, self._indices)
        return self._y_cache

    @property
    def catList(self) -> list[str]:
        return self.corpus.catList

    @property
    def catCnt(self) -> int:
        return self.corpus.catCnt


def _subset_corpus(*, base_corpus: Any, indices: Sequence[int]) -> Corpus:
    """Build a sub-corpus that preserves the parent ``catList``/``catToIdx``.

    The new ``Corpus`` instance only references the selected ``Doc``
    objects (no document copy); the category mapping is forced to the
    parent's so that ``catCnt`` matches the unchanged ``Y`` column count.
    """
    docs = list(base_corpus)
    selected_docs = [docs[int(idx)] for idx in indices]
    new_corpus = Corpus(doc for doc in selected_docs)
    if hasattr(base_corpus, 'catList'):
        new_corpus.catList = list(base_corpus.catList)
        new_corpus.catToIdx = {c: i for i, c in enumerate(new_corpus.catList)}
    return new_corpus


def slice_dataset(*, dataset: Any, indices: Sequence[int]) -> Any:
    """Return a dataset view selecting only rows at ``indices``.

    The ``X``/``Y`` tensors are shared with the base dataset; only a
    sub-corpus (referencing the same ``Doc`` objects) and an index
    tensor are allocated. ``index_select`` is invoked lazily and only
    when callers explicitly access ``.X`` or ``.Y``.

    :param dataset: Source dataset, either an :class:`EmbeddingDataset`,
        a :class:`RaggedEmbeddingDataset`, or an :class:`EmbeddingSubset`.
    :param indices: Positional row indices to select.
    :return: View dataset preserving the source's catList ordering.
    """
    idx_tensor = torch.as_tensor(indices, dtype=torch.long)

    if isinstance(dataset, RaggedEmbeddingDataset):
        composed = (
            idx_tensor if dataset._doc_indices is None
            else dataset._doc_indices.index_select(0, idx_tensor)
        )
        selected_corpus = _subset_corpus(base_corpus=dataset._base_corpus, indices=composed.tolist())
        return RaggedEmbeddingDataset(
            corpus=selected_corpus,
            article_x=dataset._article_x_base,
            entity_flat=dataset._entity_flat,
            entity_offsets=dataset._entity_offsets,
            y=dataset._y_base,
            doc_indices=composed,
            base_corpus=dataset._base_corpus,
        )

    if isinstance(dataset, EmbeddingSubset):
        composed = dataset._indices.index_select(0, idx_tensor)
        selected_corpus = _subset_corpus(base_corpus=dataset._base_corpus, indices=composed.tolist())
        return EmbeddingSubset(
            base_x=dataset._base_x,
            base_y=dataset._base_y,
            base_corpus=dataset._base_corpus,
            indices=composed,
            corpus=selected_corpus,
        )

    base_x = dataset.X if isinstance(dataset.X, torch.Tensor) else torch.as_tensor(dataset.X, dtype=torch.float32)
    if hasattr(dataset, 'Y'):
        base_y = dataset.Y if isinstance(dataset.Y, torch.Tensor) else torch.as_tensor(dataset.Y, dtype=torch.float32)
    else:
        raise ValueError('slice_dataset requires the source dataset to expose a Y tensor')
    selected_corpus = _subset_corpus(base_corpus=dataset.corpus, indices=indices)
    return EmbeddingSubset(
        base_x=base_x,
        base_y=base_y,
        base_corpus=dataset.corpus,
        indices=idx_tensor,
        corpus=selected_corpus,
    )

