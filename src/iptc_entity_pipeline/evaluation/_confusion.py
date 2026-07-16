'''Shared low-level matrix builders and confusion helpers.'''

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd

from iptc_entity_pipeline.evaluation.evaluate import normalize_pred_cats

if TYPE_CHECKING:
    from iptc_entity_pipeline.evaluation.run_loading import GoldLabelMap


def confusion_counts(
    *, pred: np.ndarray, gold: np.ndarray, axis: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''Compute TP, FP, FN, TN along ``axis``.

    :param pred: Boolean prediction matrix.
    :param gold: Boolean gold matrix (same shape as ``pred``).
    :param axis: Reduction axis (0 for per-class, 1 for per-article).
    :return: ``(tp, fp, fn, tn)`` arrays.
    '''
    gold_bool = gold.astype(bool)
    not_gold = np.logical_not(gold_bool)
    tp = np.logical_and(pred, gold_bool).sum(axis=axis, dtype=np.int64)
    fp = np.logical_and(pred, not_gold).sum(axis=axis, dtype=np.int64)
    fn = np.logical_and(np.logical_not(pred), gold_bool).sum(axis=axis, dtype=np.int64)
    tn = np.logical_and(np.logical_not(pred), not_gold).sum(axis=axis, dtype=np.int64)
    return tp, fp, fn, tn


def build_score_matrix(*, df: pd.DataFrame, cat_ids: Sequence[str]) -> np.ndarray:
    '''Build dense score matrix for selected category ids.'''
    cols = [f'prob_{cat_id}' for cat_id in cat_ids]
    return df.reindex(columns=cols, fill_value=0.0).to_numpy(dtype=float)


def build_pred_matrix(*, df: pd.DataFrame, cat_ids: Sequence[str], thr_vec: np.ndarray) -> np.ndarray:
    '''Build ancestor-normalized boolean prediction matrix aligned with ``cat_ids``.

    Thresholded predictions are pushed through :func:`normalize_pred_cats` so
    they match the gold matrix's ancestor-closure convention (a predicted leaf
    implies its ancestors) and drop ``REMOVED_CAT_IDS``. Without this, parent
    categories reached only via ancestor closure would count as false negatives,
    making confusion counts, McNemar tests, Hamming loss, and per-article F1
    disagree with the normalized headline F1 tables.

    :param df: Aligned per-article probability table.
    :param cat_ids: Category ids matching the matrix columns.
    :param thr_vec: Per-class threshold vector aligned with ``cat_ids``.
    :return: Boolean ``(n_docs, n_classes)`` matrix of normalized predictions.
    '''
    score_matrix = build_score_matrix(df=df, cat_ids=cat_ids)
    keep = score_matrix >= thr_vec
    cats = list(cat_ids)
    raw_cats = [[cats[k] for k in np.where(keep[row_idx])[0]] for row_idx in range(score_matrix.shape[0])]
    norm_cats = normalize_pred_cats(pred_cats=raw_cats)
    cat_to_idx = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}
    pred_matrix = np.zeros((len(raw_cats), len(cat_ids)), dtype=bool)
    for row_idx, cats_row in enumerate(norm_cats):
        for cat_id in cats_row:
            col_idx = cat_to_idx.get(cat_id)
            if col_idx is not None:
                pred_matrix[row_idx, col_idx] = True
    return pred_matrix


def build_paired_matrices(
    *,
    current_df: pd.DataFrame,
    base_df: pd.DataFrame,
    gold_map: 'GoldLabelMap',
    cat_ids: Sequence[str],
    current_thr_vec: np.ndarray,
    base_thr_vec: np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    '''Build aligned article_ids, gold, current_pred, base_pred matrices.

    :return: ``(article_ids, gold_matrix, current_pred, base_pred)`` tuple.
    '''
    article_ids = list(current_df['article_id'])
    gold_matrix = gold_map.gold_matrix(article_ids=article_ids, cat_ids=cat_ids)
    current_pred = build_pred_matrix(df=current_df, cat_ids=cat_ids, thr_vec=current_thr_vec)
    base_pred = build_pred_matrix(df=base_df, cat_ids=cat_ids, thr_vec=base_thr_vec)
    return article_ids, gold_matrix, current_pred, base_pred


def safe_mean(values: pd.Series) -> float:
    '''Mean over finite numeric values; ``NaN`` when no finite values exist.'''
    numeric = pd.to_numeric(values, errors='coerce')
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return float('nan')
    return float(finite.mean())
