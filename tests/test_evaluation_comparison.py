"""Tests for pure helpers in :mod:`iptc_entity_pipeline.evaluation.comparison`."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import iptc_entity_pipeline.evaluation.comparison as ec
import iptc_entity_pipeline.evaluation.significance as sig
from iptc_entity_pipeline.evaluation.comparison import (
    average_precision,
    benjamini_hochberg,
    build_class_thresholds_df,
    build_cmp_df,
    build_path,
    build_score_matrix,
    class_subset_by_ids,
    diff_only_df,
    label_from_cat_id,
    mcnemar_p_value,
    micro_pr_auc,
    shared_article_ids,
    subset_by_ids,
)

CAT_A = '01000000'
CAT_B = '02000000'
CAT_C = '03000000'


@pytest.mark.parametrize('y_true,y_score,expected', [
    # Perfectly ordered: both positives ranked first, AP=1 exactly.
    (np.array([1, 1, 0, 0]), np.array([0.9, 0.8, 0.3, 0.1]), 1.0),
    # No positives: AP is undefined; helper must return NaN (not 0).
    (np.array([0, 0]), np.array([0.5, 0.3]), np.nan),
    # Interleaved 0/1/0/1: AP = 0.5*0.5 + 0.5*0.5 = 0.5 exactly.
    (np.array([0, 1, 0, 1]), np.array([0.9, 0.8, 0.7, 0.6]), 0.5),
])
def test_average_precision(y_true: np.ndarray, y_score: np.ndarray, expected: float) -> None:
    result = average_precision(y_true=y_true, y_score=y_score)
    if np.isnan(expected):
        assert np.isnan(result)
    else:
        assert result == pytest.approx(expected, abs=1e-9)


def test_build_score_matrix() -> None:
    df = pd.DataFrame({'prob_A': [0.1, 0.2], 'prob_B': [0.3, 0.4], 'extra': [1, 2]})
    matrix = build_score_matrix(df=df, cat_ids=['A', 'B'])
    np.testing.assert_array_almost_equal(matrix, [[0.1, 0.3], [0.2, 0.4]])
    matrix_missing = build_score_matrix(df=df, cat_ids=['A', 'C'])
    np.testing.assert_array_almost_equal(matrix_missing, [[0.1, 0.0], [0.2, 0.0]])


def test_shared_article_ids_and_no_overlap() -> None:
    df1 = pd.DataFrame({'article_id': ['a1', 'a2', 'a3']})
    df2 = pd.DataFrame({'article_id': ['a2', 'a3', 'a4']})
    assert shared_article_ids(current_df=df1, base_df=df2) == ['a2', 'a3']
    with pytest.raises(ValueError, match='do not share'):
        shared_article_ids(current_df=df1, base_df=pd.DataFrame({'article_id': ['a5']}))


def test_subset_by_ids() -> None:
    df = pd.DataFrame({'article_id': ['a1', 'a2', 'a3'], 'val': [10, 20, 30]})
    result = subset_by_ids(df=df, article_ids=['a3', 'a1'])
    assert list(result['article_id']) == ['a3', 'a1']
    assert list(result['val']) == [30, 10]
    assert len(subset_by_ids(df=df, article_ids=[])) == 0


def test_build_cmp_df_and_diff_only() -> None:
    current = pd.DataFrame({
        'Precision': [0.8, 0.6], 'Recall': [0.7, 0.5], 'F1': [0.75, 0.55], 'Data Count': [10, 20],
    }, index=pd.Index(['a', 'b'], name='Key'))
    base = pd.DataFrame({
        'Precision': [0.7, 0.7], 'Recall': [0.6, 0.6], 'F1': [0.65, 0.65], 'Data Count': [10, 20],
    }, index=pd.Index(['a', 'b'], name='Key'))

    cmp = build_cmp_df(current_df=current, base_df=base, key_col='Key', info_cols=['Data Count'])
    assert 'Precision_diff' in cmp.columns
    assert cmp.iloc[0]['Precision_diff'] == pytest.approx(0.1)

    diff = diff_only_df(df=cmp, key_col='Key')
    assert 'Precision_base' not in diff.columns
    assert 'Precision_diff' in diff.columns


def test_build_output_path(tmp_path) -> None:
    """`build_path` places a timestamped workbook under a sanitized subdirectory."""
    path = build_path(output_root=tmp_path, config_name='my config!')

    assert path.parent.parent == tmp_path
    assert path.suffix == '.xlsx'
    assert path.name == 'evaluation_comparison_my_config_.xlsx'
    assert path.parent.name.startswith('my_config__')
    assert ' ' not in str(path) and '!' not in str(path)


# ---------------------------------------------------------------------------
# benjamini_hochberg
# ---------------------------------------------------------------------------

def test_benjamini_hochberg_single_value_returns_input_unchanged() -> None:
    corrected = benjamini_hochberg(p_values=[0.03])

    assert corrected.shape == (1,)
    assert corrected[0] == pytest.approx(0.03)


def test_benjamini_hochberg_ties_get_identical_adjusted_values() -> None:
    corrected = benjamini_hochberg(p_values=[0.02, 0.02, 0.02])

    assert corrected.shape == (3,)
    assert corrected[0] == pytest.approx(corrected[1])
    assert corrected[1] == pytest.approx(corrected[2])


def test_benjamini_hochberg_monotone_and_bounded_by_one() -> None:
    p_values = [0.001, 0.01, 0.03, 0.05, 0.2]
    corrected = benjamini_hochberg(p_values=p_values)

    # BH is monotone non-decreasing in the sorted p-order.
    for original, adjusted in zip(p_values, corrected):
        assert adjusted >= original
    assert corrected[-1] <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# mcnemar_p_value
# ---------------------------------------------------------------------------

def test_mcnemar_p_value_uses_exact_test_below_25(monkeypatch: pytest.MonkeyPatch) -> None:
    """Total disagreements strictly below 25 must route through the exact branch."""
    captured: dict[str, object] = {}

    class _FakeResult:
        pvalue = 0.123

    def fake_mcnemar(table, *, exact, correction):
        captured['table'] = table
        captured['exact'] = exact
        captured['correction'] = correction
        return _FakeResult()

    monkeypatch.setattr(sig, 'mcnemar', fake_mcnemar)

    result = mcnemar_p_value(n10=15, n01=5)

    assert captured['exact'] is True
    assert captured['correction'] is True
    assert captured['table'] == [[0, 5], [15, 0]]
    assert result == pytest.approx(0.123)


def test_mcnemar_p_value_uses_asymptotic_test_at_25_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    """The branch is ``< 25``, so a total of exactly 25 disagreements is asymptotic."""
    captured: dict[str, object] = {}

    class _FakeResult:
        pvalue = 0.456

    def fake_mcnemar(table, *, exact, correction):
        captured['exact'] = exact
        return _FakeResult()

    monkeypatch.setattr(sig, 'mcnemar', fake_mcnemar)

    result = mcnemar_p_value(n10=20, n01=5)

    assert captured['exact'] is False
    assert result == pytest.approx(0.456)


def test_mcnemar_p_value_symmetric_in_n10_n01(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swapping ``n10`` and ``n01`` must give the same p-value (McNemar is two-sided)."""
    monkeypatch.setattr(sig, 'mcnemar', lambda table, *, exact, correction: type(
        '_R', (), {'pvalue': float(table[0][1] + table[1][0])},
    )())

    p_ab = mcnemar_p_value(n10=7, n01=2)
    p_ba = mcnemar_p_value(n10=2, n01=7)

    assert p_ab == pytest.approx(p_ba)


# ---------------------------------------------------------------------------
# micro_pr_auc
# ---------------------------------------------------------------------------

def test_micro_pr_auc_empty_returns_nan() -> None:
    empty_gold = np.zeros((0, 0), dtype=bool)
    empty_scores = np.zeros((0, 0), dtype=float)
    assert np.isnan(micro_pr_auc(gold_matrix=empty_gold, scores=empty_scores))


def test_micro_pr_auc_no_positives_returns_nan() -> None:
    gold = np.zeros((3, 2), dtype=bool)
    scores = np.array([[0.9, 0.1], [0.5, 0.5], [0.2, 0.8]], dtype=float)
    assert np.isnan(micro_pr_auc(gold_matrix=gold, scores=scores))


def test_micro_pr_auc_perfect_ranking_equals_one() -> None:
    gold = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=bool)
    scores = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]], dtype=float)
    assert micro_pr_auc(gold_matrix=gold, scores=scores) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# class_subset_by_ids
# ---------------------------------------------------------------------------

def _make_class_cmp_df(*, cat_ids: list[str]) -> pd.DataFrame:
    """Build a minimal classes_cmp fixture with aggregate rows and per-class rows."""
    per_class_labels = [label_from_cat_id(cat_id=c) for c in cat_ids]
    labels = per_class_labels + ['All_micro', 'All_macro_head']
    per_class_scores = [0.5 + 0.05 * idx for idx, _ in enumerate(cat_ids)]
    scores = per_class_scores + [0.75, 0.75]
    return pd.DataFrame({'IPTC Category': labels, 'F1_current': scores})


def test_class_subset_by_ids_filters_out_aggregate_rows() -> None:
    df = _make_class_cmp_df(cat_ids=[CAT_A, CAT_B, CAT_C])

    subset = class_subset_by_ids(
        classes_cmp=df,
        class_ids={CAT_A, CAT_B},
        require_all=True,
        subset_name='relevant',
    )

    assert set(subset['cat_id']) == {CAT_A, CAT_B}
    assert 'All_micro' not in subset['IPTC Category'].tolist()
    assert 'All_macro_head' not in subset['IPTC Category'].tolist()


def test_class_subset_by_ids_require_all_raises_on_missing() -> None:
    df = _make_class_cmp_df(cat_ids=[CAT_A])

    with pytest.raises(ValueError, match='Missing relevant classes'):
        class_subset_by_ids(
            classes_cmp=df,
            class_ids={CAT_A, CAT_B},
            require_all=True,
            subset_name='relevant',
        )


def test_class_subset_by_ids_empty_result_raises() -> None:
    df = _make_class_cmp_df(cat_ids=[CAT_A])

    with pytest.raises(ValueError, match='No rows matched relevant category set'):
        class_subset_by_ids(
            classes_cmp=df,
            class_ids={CAT_B},
            require_all=False,
            subset_name='relevant',
        )


# ---------------------------------------------------------------------------
# build_class_thresholds_df
# ---------------------------------------------------------------------------

def test_build_class_thresholds_df_defaults_to_default_when_missing() -> None:
    df = build_class_thresholds_df(
        cat_ids=[CAT_A, CAT_B],
        default_threshold=0.5,
        current_thresholds={CAT_A: 0.6},
        base_thresholds={CAT_B: 0.4},
        class_supports={CAT_A: 10, CAT_B: 20},
    )

    row_a = df.set_index('class_id').loc[CAT_A]
    row_b = df.set_index('class_id').loc[CAT_B]

    assert row_a['threshold_current'] == pytest.approx(0.6)
    assert row_a['threshold_base'] == pytest.approx(0.5)
    assert row_a['threshold_diff'] == pytest.approx(0.1)
    assert row_a['count'] == 10

    assert row_b['threshold_current'] == pytest.approx(0.5)
    assert row_b['threshold_base'] == pytest.approx(0.4)
    assert row_b['threshold_diff'] == pytest.approx(0.1)
    assert row_b['count'] == 20


def test_build_class_thresholds_df_supports_defaults_to_zero() -> None:
    df = build_class_thresholds_df(
        cat_ids=[CAT_A],
        default_threshold=0.3,
        current_thresholds={},
        base_thresholds={},
    )

    row_a = df.iloc[0]
    assert row_a['count'] == 0
    assert row_a['threshold_current'] == pytest.approx(0.3)
    assert row_a['threshold_base'] == pytest.approx(0.3)
    assert row_a['threshold_diff'] == pytest.approx(0.0)
