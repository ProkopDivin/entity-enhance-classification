"""Tests for pure helpers in :mod:`iptc_entity_pipeline.cross_validation`.

These cover Optuna direction handling, per-fold metric extraction, trial
aggregation, dense/list prediction subsetting, mean-of-eval-tables alignment,
final-epochs resolution, and CV dev-row assembly. All helpers are pure and
run without ClearML or torch.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from iptc_entity_pipeline.cross_validation import (
    FoldMetrics,
    _build_cv_dev_row,
    _is_better_score,
    _mean_eval_tables,
    _resolve_final_epochs,
    _subset_predictions,
    extract_metric_rows,
    summarize_combination,
)
from iptc_entity_pipeline.evaluation.evaluate import CLASS_RELEVANT_MACRO_ROW


def _make_fold_metric(
    *,
    fold_id: int,
    epochs: float,
    best_epoch: int,
    loss: float,
    f1_micro: float,
) -> FoldMetrics:
    """Build a FoldMetrics with sensible defaults so tests only override what matters."""
    return FoldMetrics(
        trial_id=0,
        fold_id=fold_id,
        params='{}',
        epochs=epochs,
        best_epoch=best_epoch,
        loss=loss,
        precision_macro_relevant=0.5,
        recall_macro_relevant=0.5,
        f1_macro_relevant=0.5,
        precision_micro=0.5,
        recall_micro=0.5,
        f1_micro=f1_micro,
    )


# ---------------------------------------------------------------------------
# _is_better_score
# ---------------------------------------------------------------------------

def test_is_better_score_maximize() -> None:
    assert _is_better_score(candidate=0.9, best=0.8, direction='maximize')
    assert not _is_better_score(candidate=0.7, best=0.8, direction='maximize')


def test_is_better_score_minimize() -> None:
    assert _is_better_score(candidate=0.1, best=0.2, direction='minimize')
    assert not _is_better_score(candidate=0.3, best=0.2, direction='minimize')


def test_is_better_score_equal_is_not_better() -> None:
    assert not _is_better_score(candidate=0.5, best=0.5, direction='maximize')
    assert not _is_better_score(candidate=0.5, best=0.5, direction='minimize')


def test_is_better_score_rejects_unknown_direction() -> None:
    with pytest.raises(ValueError, match='Unsupported Optuna direction'):
        _is_better_score(candidate=0.5, best=0.5, direction='sideways')


# ---------------------------------------------------------------------------
# extract_metric_rows
# ---------------------------------------------------------------------------

def test_extract_metric_rows_returns_dicts_for_expected_labels() -> None:
    corpora_df = pd.DataFrame(
        {'Precision': [0.7, 0.8], 'Recall': [0.6, 0.9]},
        index=['All_micro', 'named_corpus'],
    )
    classes_df = pd.DataFrame(
        {'Precision': [0.4, 0.55], 'F1': [0.45, 0.5]},
        index=['All_micro', CLASS_RELEVANT_MACRO_ROW],
    )

    objective, macro_relevant = extract_metric_rows(
        df_corpora_fold=corpora_df,
        df_classes_fold=classes_df,
        objective_row='All_micro',
        averaging_type='micro',
    )

    assert objective == {'Precision': 0.7, 'Recall': 0.6}
    assert macro_relevant == {'Precision': 0.55, 'F1': 0.5}


def test_extract_metric_rows_raises_when_row_missing() -> None:
    corpora_df = pd.DataFrame({'F1': [0.7]}, index=['other'])
    classes_df = pd.DataFrame({'F1': [0.5]}, index=[CLASS_RELEVANT_MACRO_ROW])
    with pytest.raises(KeyError):
        extract_metric_rows(
            df_corpora_fold=corpora_df,
            df_classes_fold=classes_df,
            objective_row='All_micro',
            averaging_type='micro',
        )


# ---------------------------------------------------------------------------
# summarize_combination
# ---------------------------------------------------------------------------

def test_summarize_combination_computes_mean_and_sample_std() -> None:
    fms = [
        _make_fold_metric(fold_id=0, epochs=10.0, best_epoch=8, loss=0.4, f1_micro=0.60),
        _make_fold_metric(fold_id=1, epochs=12.0, best_epoch=9, loss=0.5, f1_micro=0.80),
        _make_fold_metric(fold_id=2, epochs=14.0, best_epoch=10, loss=0.6, f1_micro=0.70),
    ]

    row = summarize_combination(combo_idx=3, params_json='{"lr":0.001}', fold_metrics=fms)

    assert row['trial_id'] == 3
    assert row['params'] == '{"lr":0.001}'
    assert row['epochs'] == pytest.approx(12.0)
    assert row['best_epoch'] == pytest.approx(9.0)
    assert row['Loss_mean'] == pytest.approx(0.5)
    # ddof=1 sample std over [0.4, 0.5, 0.6] equals 0.1 exactly.
    assert row['Loss_std'] == pytest.approx(0.1)
    assert row['F1_micro_mean'] == pytest.approx(0.7)
    # ddof=1 sample std over [0.60, 0.80, 0.70].
    assert row['F1_micro_std'] == pytest.approx(np.std([0.60, 0.80, 0.70], ddof=1))


def test_summarize_combination_std_is_nan_for_single_fold() -> None:
    fm = _make_fold_metric(fold_id=0, epochs=5.0, best_epoch=4, loss=0.3, f1_micro=0.9)

    row = summarize_combination(combo_idx=0, params_json='{}', fold_metrics=[fm])

    assert row['Loss_mean'] == pytest.approx(0.3)
    assert np.isnan(row['Loss_std'])
    assert np.isnan(row['F1_micro_std'])


# ---------------------------------------------------------------------------
# _mean_eval_tables
# ---------------------------------------------------------------------------

def test_mean_eval_tables_averages_overlapping_numeric_columns() -> None:
    first = pd.DataFrame(
        {'F1': [0.4, 0.8], 'Precision': [0.5, 0.9]},
        index=['row_a', 'row_b'],
    )
    second = pd.DataFrame(
        {'F1': [0.6, 0.2], 'Precision': [0.7, 0.1]},
        index=['row_a', 'row_b'],
    )

    out = _mean_eval_tables(first_df=first, second_df=second)

    assert out.loc['row_a', 'F1'] == pytest.approx(0.5)
    assert out.loc['row_b', 'F1'] == pytest.approx(0.5)
    assert out.loc['row_a', 'Precision'] == pytest.approx(0.6)
    assert out.loc['row_b', 'Precision'] == pytest.approx(0.5)


def test_mean_eval_tables_aligns_disjoint_rows_and_columns() -> None:
    first = pd.DataFrame(
        {'F1': [0.4], 'Only_first': [1.0]},
        index=['row_a'],
    )
    second = pd.DataFrame(
        {'F1': [0.8], 'Only_second': [2.0]},
        index=['row_b'],
    )

    out = _mean_eval_tables(first_df=first, second_df=second)

    assert set(out.index) == {'row_a', 'row_b'}
    assert set(out.columns) == {'F1', 'Only_first', 'Only_second'}
    # Non-overlapping numeric cells: mean-of-one equals the value itself.
    assert out.loc['row_a', 'F1'] == pytest.approx(0.4)
    assert out.loc['row_b', 'F1'] == pytest.approx(0.8)
    assert out.loc['row_a', 'Only_first'] == pytest.approx(1.0)
    assert out.loc['row_b', 'Only_second'] == pytest.approx(2.0)


def test_mean_eval_tables_preserves_non_numeric_columns() -> None:
    first = pd.DataFrame(
        {'F1': [0.4], 'label': ['alpha']},
        index=['row_a'],
    )
    second = pd.DataFrame(
        {'F1': [0.6], 'label': ['alpha']},
        index=['row_a'],
    )

    out = _mean_eval_tables(first_df=first, second_df=second)

    assert out.loc['row_a', 'label'] == 'alpha'
    assert out.loc['row_a', 'F1'] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _subset_predictions
# ---------------------------------------------------------------------------

def test_subset_predictions_dense_ndarray_returns_rows_by_index() -> None:
    scores = np.arange(15, dtype=np.float32).reshape(5, 3)

    sub = _subset_predictions(pred_scores=scores, indices=[4, 0, 2])

    assert isinstance(sub, np.ndarray)
    np.testing.assert_array_equal(sub, scores[[4, 0, 2]])


def test_subset_predictions_list_path_preserves_order() -> None:
    scores = [
        [('A', 0.1)],
        [('B', 0.2)],
        [('C', 0.3)],
        [('D', 0.4)],
    ]

    sub = _subset_predictions(pred_scores=scores, indices=[3, 1])

    assert sub == [[('D', 0.4)], [('B', 0.2)]]
    assert sub is not scores


# ---------------------------------------------------------------------------
# _resolve_final_epochs
# ---------------------------------------------------------------------------

def test_resolve_final_epochs_uncapped_uses_epochs_minus_patience() -> None:
    fms = [
        _make_fold_metric(fold_id=0, epochs=15.0, best_epoch=8, loss=0.0, f1_micro=0.0),
        _make_fold_metric(fold_id=1, epochs=17.0, best_epoch=10, loss=0.0, f1_micro=0.0),
    ]
    best_trial = {'epochs': 16.0, 'best_epoch': 9.0}

    epochs = _resolve_final_epochs(
        best_trial=best_trial, fold_metrics=fms, max_epochs=100, patience=5,
    )

    assert epochs == 16 - 5


def test_resolve_final_epochs_capped_uses_best_epoch_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fms = [
        _make_fold_metric(fold_id=0, epochs=100.0, best_epoch=42, loss=0.0, f1_micro=0.0),
        _make_fold_metric(fold_id=1, epochs=80.0, best_epoch=40, loss=0.0, f1_micro=0.0),
    ]
    best_trial = {'epochs': 90.0, 'best_epoch': 41.0}

    with caplog.at_level(logging.WARNING, logger='iptc_entity_pipeline.cross_validation'):
        epochs = _resolve_final_epochs(
            best_trial=best_trial, fold_metrics=fms, max_epochs=100, patience=5,
        )

    assert epochs == 41
    assert any('reached the maximum epoch limit' in rec.message for rec in caplog.records)


def test_resolve_final_epochs_returns_at_least_one() -> None:
    fms = [_make_fold_metric(fold_id=0, epochs=3.0, best_epoch=1, loss=0.0, f1_micro=0.0)]
    best_trial = {'epochs': 3.0, 'best_epoch': 1.0}

    epochs = _resolve_final_epochs(
        best_trial=best_trial, fold_metrics=fms, max_epochs=100, patience=10,
    )

    assert epochs == 1


# ---------------------------------------------------------------------------
# _build_cv_dev_row
# ---------------------------------------------------------------------------

def test_build_cv_dev_row_preserves_expected_keys_and_types() -> None:
    best_trial = {
        'params': '{"lr":0.001}',
        'epochs': 15.0,
        'best_epoch': 9.0,
        'Loss_mean': 0.42,
        'Loss_std': 0.03,
        'Precision_micro_mean': 0.71,
        'Precision_micro_std': 0.02,
        'Recall_micro_mean': 0.68,
        'Recall_micro_std': 0.03,
        'F1_micro_mean': 0.7,
        'F1_micro_std': 0.02,
        'Precision_macro_relevant_mean': 0.5,
        'Precision_macro_relevant_std': 0.04,
        'Recall_macro_relevant_mean': 0.55,
        'Recall_macro_relevant_std': 0.05,
        'F1_macro_relevant_mean': 0.52,
        'F1_macro_relevant_std': 0.06,
    }

    row = _build_cv_dev_row(best_trial)

    expected_keys = {
        'params', 'epochs', 'best_epoch',
        'Loss', 'Loss_std',
        'Precision_micro', 'Precision_micro_std',
        'Recall_micro', 'Recall_micro_std',
        'F1_micro', 'F1_micro_std',
        'Precision_macro_relevant', 'Precision_macro_relevant_std',
        'Recall_macro_relevant', 'Recall_macro_relevant_std',
        'F1_macro_relevant', 'F1_macro_relevant_std',
    }
    assert set(row) == expected_keys
    assert row['params'] == '{"lr":0.001}'
    assert row['epochs'] == pytest.approx(15.0)
    assert row['Loss'] == pytest.approx(0.42)
    assert row['F1_micro'] == pytest.approx(0.7)
    assert row['F1_macro_relevant_std'] == pytest.approx(0.06)
    numeric_keys = expected_keys - {'params'}
    for key in numeric_keys:
        assert isinstance(row[key], float)


def test_build_cv_dev_row_raises_on_missing_key() -> None:
    with pytest.raises(KeyError):
        _build_cv_dev_row({'params': '{}'})
