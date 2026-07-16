"""Tests for per-class threshold tuning helpers."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
from geneea.catlib.data import Corpus, Doc

from iptc_entity_pipeline.config import ThresholdTuningCnf
from iptc_entity_pipeline.evaluation.evaluate import filter_and_normalize, pred_cats_from_matrix
from iptc_entity_pipeline.threshold_tuning import (
    _safe_mode,
    aggregate_fold_thresholds,
    eval_at_thresholds,
    eval_at_thresholds_dense,
    select_thresholds_by_f1,
    tune_thresholds,
    tune_thresholds_dense,
)

CAT_A = '01000000'
CAT_B = '02000000'
CAT_C = '03000000'


def _build_score_matrix_corpus() -> tuple[np.ndarray, list[str], list[list[tuple[str, float]]], Corpus]:
    """Build paired dense/sparse predictions plus matching gold corpus.

    Uses float32-exact scores (multiples of 1/16) and thresholds chosen
    away from those values so the dense ``>=`` (float32 vs float32) and
    sparse ``>=`` (Python-float upcast) paths are guaranteed to agree
    bit-for-bit.
    """
    cat_list = [CAT_A, CAT_B, CAT_C]
    score_matrix = np.asarray(
        [
            [0.875, 0.125, 0.1875],
            [0.375, 0.8125, 0.5625],
            [0.1875, 0.3125, 0.6875],
            [0.5625, 0.625, 0.4375],
        ],
        dtype=np.float32,
    )
    pred_wgh_cats = [
        [(cat_list[k], float(score_matrix[i, k])) for k in range(score_matrix.shape[1])]
        for i in range(score_matrix.shape[0])
    ]
    docs = [
        Doc(id=f'doc_{i}', title='', lead='', text='', cats=cats, metadata={'corpusName': 'c'})
        for i, cats in enumerate([[CAT_A], [CAT_B], [CAT_C], [CAT_A, CAT_B]])
    ]
    return score_matrix, cat_list, pred_wgh_cats, Corpus(docs)


class FakeStats:
    """Minimal stand-in for ``geneea.evaluation.utils.ClassData``."""

    def __init__(self, *, f1: float = 0.0) -> None:
        self._f1 = float(f1)

    def fmeasure(self, *, beta: float = 1.0) -> float:
        return self._f1 if abs(beta - 1.0) < 1e-12 else self._f1 * 0.9


def test_threshold_tuning_default_grid_spans_025_to_065() -> None:
    cfg = ThresholdTuningCnf()
    assert cfg.thresholds[0] == pytest.approx(0.25)
    assert cfg.thresholds[-1] == pytest.approx(0.65)
    assert len(cfg.thresholds) == 9
    assert cfg.aggregation == 'mean'
    assert cfg.f_beta == 1.0
    assert cfg.min_folds_for_tuning == 3
    assert cfg.selection_metric == 'F1_micro'
    assert cfg.enabled is False


def test_select_thresholds_by_f1_picks_argmax_per_class() -> None:
    thr_stats: dict[float, dict[str, Any]] = {
        0.10: {'A': FakeStats(f1=0.20), 'B': FakeStats(f1=0.50)},
        0.30: {'A': FakeStats(f1=0.60), 'B': FakeStats(f1=0.40)},
        0.60: {'A': FakeStats(f1=0.50), 'B': FakeStats(f1=0.10)},
    }
    selected = select_thresholds_by_f1(thr_stats=thr_stats, f_beta=1.0)
    assert selected['A'][0] == pytest.approx(0.30)
    assert selected['B'][0] == pytest.approx(0.10)


def test_select_thresholds_by_f1_handles_class_only_in_some_thresholds() -> None:
    thr_stats: dict[float, dict[str, Any]] = {
        0.10: {'X': FakeStats(f1=0.10)},
        0.30: {'X': FakeStats(f1=0.40), 'Y': FakeStats(f1=0.70)},
    }
    selected = select_thresholds_by_f1(thr_stats=thr_stats, f_beta=1.0)
    assert selected['X'][0] == pytest.approx(0.30)
    assert selected['Y'][0] == pytest.approx(0.30)


def test_safe_mode_returns_most_common_breaking_ties_by_first_seen() -> None:
    assert _safe_mode([0.3, 0.2, 0.3, 0.4]) == pytest.approx(0.3)
    assert _safe_mode([0.4, 0.5, 0.4, 0.5]) == pytest.approx(0.4)


def test_aggregate_fold_thresholds_mean_aggregation() -> None:
    fold_thresholds = [
        {'A': 0.30, 'B': 0.40},
        {'A': 0.50, 'B': 0.60},
        {'A': 0.40, 'B': 0.50},
    ]
    result = aggregate_fold_thresholds(
        fold_thresholds=fold_thresholds,
        cat_list=['A', 'B', 'C'],
        default_threshold=0.5,
        aggregation='mean',
    )
    assert result.cat_to_threshold['A'] == pytest.approx(0.40)
    assert result.cat_to_threshold['B'] == pytest.approx(0.50)
    assert result.cat_to_threshold['C'] == pytest.approx(0.50)
    assert isinstance(result.report_df, pd.DataFrame)
    row_a = result.report_df.set_index('category_id').loc['A']
    assert row_a['threshold_mean'] == pytest.approx(0.40)
    assert row_a['threshold_median'] == pytest.approx(0.40)
    assert row_a['n_folds'] == 3
    assert row_a['threshold_std'] == pytest.approx(0.0816496580927726, rel=1e-3)
    row_c = result.report_df.set_index('category_id').loc['C']
    assert row_c['n_folds'] == 0
    assert math.isnan(row_c['threshold_mean'])
    assert math.isnan(row_c['threshold_median'])
    assert row_c['threshold_selected'] == pytest.approx(0.5)


def test_aggregate_fold_thresholds_median_aggregation() -> None:
    fold_thresholds = [
        {'A': 0.30, 'B': 0.40},
        {'A': 0.50, 'B': 0.60},
        {'A': 0.40, 'B': 0.50},
    ]
    result = aggregate_fold_thresholds(
        fold_thresholds=fold_thresholds,
        cat_list=['A', 'B', 'C'],
        default_threshold=0.5,
        aggregation='median',
    )
    assert result.cat_to_threshold['A'] == pytest.approx(0.40)
    assert result.cat_to_threshold['B'] == pytest.approx(0.50)
    assert result.cat_to_threshold['C'] == pytest.approx(0.50)
    row_a = result.report_df.set_index('category_id').loc['A']
    assert row_a['threshold_selected'] == pytest.approx(0.40)


def test_aggregate_fold_thresholds_mode_aggregation() -> None:
    fold_thresholds = [
        {'A': 0.30},
        {'A': 0.30},
        {'A': 0.50},
        {'A': 0.50},
        {'A': 0.30},
    ]
    result = aggregate_fold_thresholds(
        fold_thresholds=fold_thresholds,
        cat_list=['A'],
        default_threshold=0.5,
        aggregation='mode',
    )
    assert result.cat_to_threshold['A'] == pytest.approx(0.30)


def test_aggregate_fold_thresholds_uses_default_below_min_fold_count() -> None:
    fold_thresholds = [
        {'A': 0.10},
        {'B': 0.70},
        {'B': 0.60},
    ]
    result = aggregate_fold_thresholds(
        fold_thresholds=fold_thresholds,
        cat_list=['A', 'B'],
        default_threshold=0.5,
        aggregation='mean',
        min_folds_for_tuning=3,
    )
    assert result.cat_to_threshold['A'] == pytest.approx(0.5)
    assert result.cat_to_threshold['B'] == pytest.approx(0.5)
    rows = result.report_df.set_index('category_id')
    assert rows.loc['A', 'n_folds'] == 1
    assert rows.loc['A', 'threshold_selected'] == pytest.approx(0.5)
    assert rows.loc['B', 'n_folds'] == 2
    assert rows.loc['B', 'threshold_selected'] == pytest.approx(0.5)


def test_aggregate_fold_thresholds_rejects_unknown_aggregation() -> None:
    with pytest.raises(ValueError, match='Unsupported threshold aggregation'):
        aggregate_fold_thresholds(
            fold_thresholds=[{'A': 0.4}],
            cat_list=['A'],
            default_threshold=0.5,
            aggregation='weighted_mean',
        )


def test_aggregate_fold_thresholds_rejects_invalid_min_folds_for_tuning() -> None:
    with pytest.raises(ValueError, match='min_folds_for_tuning must be >= 1'):
        aggregate_fold_thresholds(
            fold_thresholds=[{'A': 0.4}],
            cat_list=['A'],
            default_threshold=0.5,
            min_folds_for_tuning=0,
        )


def test_pred_cats_from_matrix_matches_filter_and_normalize() -> None:
    score_matrix, cat_list, pred_wgh_cats, _ = _build_score_matrix_corpus()

    from_matrix = pred_cats_from_matrix(score_matrix=score_matrix, cat_list=cat_list, threshold=0.5)
    from_wghs = filter_and_normalize(pred_wgh_cats=pred_wgh_cats, thr=0.5)

    assert from_matrix == from_wghs


def test_pred_cats_from_matrix_respects_per_class_overrides() -> None:
    score_matrix, cat_list, pred_wgh_cats, _ = _build_score_matrix_corpus()
    cat_to_thr = {CAT_A: 0.625, CAT_C: 0.75}

    from_matrix = pred_cats_from_matrix(
        score_matrix=score_matrix, cat_list=cat_list, threshold=0.5, cat_to_thr=cat_to_thr,
    )
    from_wghs = filter_and_normalize(pred_wgh_cats=pred_wgh_cats, thr=0.5, cat_to_thr=cat_to_thr)

    assert from_matrix == from_wghs


def test_eval_at_thresholds_dense_matches_sparse_version() -> None:
    score_matrix, cat_list, pred_wgh_cats, corpus = _build_score_matrix_corpus()
    grid = [0.25, 0.5, 0.75]

    dense = eval_at_thresholds_dense(
        score_matrix=score_matrix, cat_list=cat_list, eval_corpus=corpus, thresholds=grid,
    )
    sparse = eval_at_thresholds(pred_wgh_cats=pred_wgh_cats, eval_corpus=corpus, thresholds=grid)

    assert dense.keys() == sparse.keys()
    for thr in grid:
        dense_stats = dense[thr]
        sparse_stats = sparse[thr]
        assert dense_stats.keys() == sparse_stats.keys()
        for cat in dense_stats:
            assert dense_stats[cat].fmeasure(beta=1) == pytest.approx(
                sparse_stats[cat].fmeasure(beta=1)
            )


def test_tune_thresholds_dense_matches_sparse_version() -> None:
    score_matrix, cat_list, pred_wgh_cats, corpus = _build_score_matrix_corpus()
    cfg = ThresholdTuningCnf(thresholds=(0.25, 0.5, 0.75))

    dense_map = tune_thresholds_dense(
        score_matrix=score_matrix, cat_list=cat_list, eval_corpus=corpus, tuning_cfg=cfg,
    )
    sparse_map = tune_thresholds(pred_wgh_cats=pred_wgh_cats, eval_corpus=corpus, tuning_cfg=cfg)

    assert dense_map == sparse_map


def test_tune_thresholds_dense_rejects_empty_grid() -> None:
    score_matrix, cat_list, _, corpus = _build_score_matrix_corpus()
    cfg = ThresholdTuningCnf(thresholds=())
    with pytest.raises(ValueError, match='must not be empty'):
        tune_thresholds_dense(
            score_matrix=score_matrix, cat_list=cat_list, eval_corpus=corpus, tuning_cfg=cfg,
        )


def test_pred_cats_from_matrix_validates_shape() -> None:
    with pytest.raises(ValueError, match='columns'):
        pred_cats_from_matrix(
            score_matrix=np.zeros((2, 5), dtype=np.float32),
            cat_list=[CAT_A, CAT_B],
            threshold=0.5,
        )
