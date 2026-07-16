'''McNemar significance tests, PR-AUC builders, and average precision helpers.'''

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from sklearn.metrics import average_precision_score
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
import numpy as np
import pandas as pd

from iptc_entity_pipeline.evaluation._confusion import (
    build_pred_matrix,
    build_score_matrix,
    safe_mean,
)
from iptc_entity_pipeline.evaluation.comparison_tables import (
    LANG_PREFIXES,
    MACRO_HEAD_MIN_SUPPORT,
    SUPPORT_BUCKETS,
    safe_cat_label,
)
from iptc_entity_pipeline.evaluation.run_loading import GoldLabelMap

MCNEMAR_ALPHA = 0.05
MCNEMAR_MIN_DISAGREEMENTS = 25
EUROSPORT_TOKEN = 'eurosport'


# ---------------------------------------------------------------------------
# McNemar significance
# ---------------------------------------------------------------------------

def build_mcnemar_significance_df(
    *,
    current_df: pd.DataFrame,
    base_df: pd.DataFrame,
    gold_map: GoldLabelMap,
    cat_ids: Sequence[str],
    current_thr_vec: np.ndarray,
    base_thr_vec: np.ndarray,
    alpha: float = MCNEMAR_ALPHA,
    min_disagreements: int = MCNEMAR_MIN_DISAGREEMENTS,
) -> pd.DataFrame:
    '''Run per-class McNemar tests on paired current/base predictions.

    ``n10`` counts articles where the current model is correct and the base
    model is wrong. ``n01`` counts the opposite. Rows with too few
    disagreements do not pass significance and receive ``NaN`` p-values.

    Raw ``mcnemar_p_value`` entries are adjusted **across classes** with
    Benjamini-Hochberg FDR (``mcnemar_p_value_fdr``). Pass flags use the FDR
    column with ``alpha``, same pattern as Brier and bootstrap PR-AUC.

    :param current_thr_vec: Per-class threshold vector for the current run
        (aligned with ``cat_ids``).
    :param base_thr_vec: Per-class threshold vector for the base run.
    '''
    article_ids = list(current_df['article_id'])
    gold_matrix = gold_map.gold_matrix(article_ids=article_ids, cat_ids=cat_ids).astype(bool)
    current_pred = build_pred_matrix(df=current_df, cat_ids=cat_ids, thr_vec=current_thr_vec)
    base_pred = build_pred_matrix(df=base_df, cat_ids=cat_ids, thr_vec=base_thr_vec)
    current_correct = current_pred == gold_matrix
    base_correct = base_pred == gold_matrix

    rows = []
    for idx, cat_id in enumerate(cat_ids):
        n10 = int(np.logical_and(current_correct[:, idx], np.logical_not(base_correct[:, idx])).sum())
        n01 = int(np.logical_and(np.logical_not(current_correct[:, idx]), base_correct[:, idx]).sum())
        disagreements = n10 + n01
        skipped = int(disagreements < min_disagreements)
        p_value = (
            float('nan')
            if skipped
            else mcnemar_p_value(n10=n10, n01=n01)
        )
        rows.append(
            {
                'IPTC Category': safe_cat_label(cat_id=cat_id),
                'cat_id': cat_id,
                'mcnemar_p_value': p_value,
                'mcnemar_n10_current_only_correct': n10,
                'mcnemar_n01_base_only_correct': n01,
                'mcnemar_disagreements': disagreements,
            }
        )
    df = pd.DataFrame(rows)
    p_numpy = df['mcnemar_p_value'].to_numpy(dtype=float)
    fdr = np.full(len(df), np.nan, dtype=float)
    ok = np.isfinite(p_numpy)
    if np.any(ok):
        fdr[ok] = benjamini_hochberg(p_values=p_numpy[ok])
    df['mcnemar_p_value_fdr'] = fdr
    fdr_ok = np.isfinite(df['mcnemar_p_value_fdr'].to_numpy(dtype=float))
    n10s = df['mcnemar_n10_current_only_correct'].to_numpy(dtype=int)
    n01s = df['mcnemar_n01_base_only_correct'].to_numpy(dtype=int)
    df['mcnemar_current_significant'] = (
        fdr_ok & (df['mcnemar_p_value_fdr'] < alpha) & (n10s > n01s)
    ).astype(int)
    df['mcnemar_base_significant'] = (
        fdr_ok & (df['mcnemar_p_value_fdr'] < alpha) & (n01s > n10s)
    ).astype(int)
    return df


def mcnemar_p_value(*, n10: int, n01: int) -> float:
    '''Return asymptotic McNemar p-value with continuity correction.'''
    table = [[0, n01],
             [n10, 0]]
    if n10 + n01 < 25:
        return float(mcnemar(table, exact=True, correction=True).pvalue)
    else:
        return float(mcnemar(table, exact=False, correction=True).pvalue)


def add_mcnemar_to_top_change_dfs(
    *,
    improved_df: pd.DataFrame,
    degraded_df: pd.DataFrame,
    mcnemar_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''Attach McNemar significance columns to top improved/degraded tables.'''
    return (
        _add_mcnemar_to_top_change_df(
            df=improved_df,
            mcnemar_df=mcnemar_df,
            pass_col='mcnemar_current_significant',
        ),
        _add_mcnemar_to_top_change_df(
            df=degraded_df,
            mcnemar_df=mcnemar_df,
            pass_col='mcnemar_base_significant',
        ),
    )


def _add_mcnemar_to_top_change_df(
    *,
    df: pd.DataFrame,
    mcnemar_df: pd.DataFrame,
    pass_col: str,
) -> pd.DataFrame:
    '''Merge McNemar rows and expose a direction-aware pass flag.'''
    cols = [
        'IPTC Category',
        'mcnemar_p_value',
        'mcnemar_p_value_fdr',
        'mcnemar_n10_current_only_correct',
        'mcnemar_n01_base_only_correct',
        pass_col,
    ]
    result = df.merge(mcnemar_df.reindex(columns=cols), on='IPTC Category', how='left')
    result['mcnemar_pass'] = result[pass_col].fillna(0).astype(int)
    result = result.drop(columns=[pass_col])
    metric_cols = [
        'mcnemar_pass',
        'mcnemar_p_value',
        'mcnemar_p_value_fdr',
        'mcnemar_n10_current_only_correct',
        'mcnemar_n01_base_only_correct',
    ]
    base_cols = [col for col in result.columns if col not in metric_cols]
    return result.loc[:, base_cols + metric_cols]


def benjamini_hochberg(*, p_values: Sequence[float]) -> np.ndarray:
    '''Apply Benjamini-Hochberg FDR correction preserving input order.'''
    adjusted = multipletests(p_values, method='fdr_bh')[1]
    return np.asarray(adjusted, dtype=float)


# ---------------------------------------------------------------------------
# PR-AUC
# ---------------------------------------------------------------------------

def build_pr_auc_dfs(
    *,
    current_df: pd.DataFrame,
    base_df: pd.DataFrame,
    gold_map: GoldLabelMap,
    cat_ids: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''Build per-class and summary PR-AUC tables.'''
    article_ids = list(current_df['article_id'])
    gold_matrix = gold_map.gold_matrix(article_ids=article_ids, cat_ids=cat_ids)
    current_scores = build_score_matrix(df=current_df, cat_ids=cat_ids)
    base_scores = build_score_matrix(df=base_df, cat_ids=cat_ids)

    rows = []
    for idx, cat_id in enumerate(cat_ids):
        y_true = gold_matrix[:, idx].astype(np.int8)
        current_pr_auc = average_precision(y_true=y_true, y_score=current_scores[:, idx])
        base_pr_auc = average_precision(y_true=y_true, y_score=base_scores[:, idx])
        rows.append(
            {
                'IPTC Category': safe_cat_label(cat_id=cat_id),
                'cat_id': cat_id,
                'pr_auc_current': current_pr_auc,
                'pr_auc_base': base_pr_auc,
                'pr_auc_diff': current_pr_auc - base_pr_auc
                if not np.isnan(current_pr_auc) and not np.isnan(base_pr_auc)
                else np.nan,
                'positive_support': int(y_true.sum()),
                'article_frequency': int(y_true.sum()),
            }
        )

    pr_auc_df = pd.DataFrame(rows)
    pr_auc_summary_df = build_pr_auc_summary_df(
        pr_auc_df=pr_auc_df,
        current_df=current_df,
        gold_matrix=gold_matrix,
        current_scores=current_scores,
        base_scores=base_scores,
    )
    return pr_auc_df, pr_auc_summary_df


def build_pr_auc_summary_df(
    *,
    pr_auc_df: pd.DataFrame,
    current_df: pd.DataFrame,
    gold_matrix: np.ndarray,
    current_scores: np.ndarray,
    base_scores: np.ndarray,
) -> pd.DataFrame:
    '''Build PR-AUC summary rows: micro, macro, support buckets, per-corpus groups.'''
    rows: list[dict[str, Any]] = []

    rows.append(
        _pr_auc_row(
            aggregation='macro_all',
            current=safe_mean(pr_auc_df['pr_auc_current']),
            base=safe_mean(pr_auc_df['pr_auc_base']),
        )
    )
    tail_mask = pr_auc_df['positive_support'] < MACRO_HEAD_MIN_SUPPORT
    tail_sub = pr_auc_df[tail_mask]
    rows.append(
        _pr_auc_row(
            aggregation='macro_tail',
            current=safe_mean(tail_sub['pr_auc_current']),
            base=safe_mean(tail_sub['pr_auc_base']),
        )
    )
    rows.append(
        _pr_auc_row(
            aggregation='micro',
            current=micro_pr_auc(gold_matrix=gold_matrix, scores=current_scores),
            base=micro_pr_auc(gold_matrix=gold_matrix, scores=base_scores),
        )
    )

    for low, high, label in SUPPORT_BUCKETS:
        mask = (pr_auc_df['positive_support'] >= low) & (pr_auc_df['positive_support'] < high)
        sub = pr_auc_df[mask]
        rows.append(
            _pr_auc_row(
                aggregation=f'macro_support_{label}',
                current=safe_mean(sub['pr_auc_current']),
                base=safe_mean(sub['pr_auc_base']),
            )
        )

    current_per_corpus = per_corpus_pr_auc(df=current_df, scores=current_scores, gold_matrix=gold_matrix)
    base_per_corpus = per_corpus_pr_auc(df=current_df, scores=base_scores, gold_matrix=gold_matrix)

    for prefix in LANG_PREFIXES:
        rows.append(
            _pr_auc_row(
                aggregation=f'macro_over_corpora_prefix_{prefix}',
                current=_avg_filtered(values_dict=current_per_corpus, predicate=_prefix_predicate(prefix)),
                base=_avg_filtered(values_dict=base_per_corpus, predicate=_prefix_predicate(prefix)),
            )
        )

    rows.append(
        _pr_auc_row(
            aggregation=f'macro_over_corpora_{EUROSPORT_TOKEN}',
            current=_avg_filtered(values_dict=current_per_corpus, predicate=_contains_predicate(EUROSPORT_TOKEN)),
            base=_avg_filtered(values_dict=base_per_corpus, predicate=_contains_predicate(EUROSPORT_TOKEN)),
        )
    )

    rows.append(
        _pr_auc_row(
            aggregation='macro_over_corpora',
            current=safe_mean(pd.Series(list(current_per_corpus.values()), dtype=float)),
            base=safe_mean(pd.Series(list(base_per_corpus.values()), dtype=float)),
        )
    )
    return pd.DataFrame(rows)


def _pr_auc_row(*, aggregation: str, current: float, base: float) -> dict[str, Any]:
    '''Build one PR-AUC summary row with diff.'''
    diff = current - base if not np.isnan(current) and not np.isnan(base) else np.nan
    return {'aggregation': aggregation, 'current': current, 'base': base, 'diff': diff}


def _avg_filtered(*, values_dict: Mapping[str, float], predicate: Callable[[str], bool]) -> float:
    '''Mean of dict values whose key matches ``predicate`` and is not NaN.'''
    matched = [v for k, v in values_dict.items() if predicate(k) and not np.isnan(v)]
    return float(np.mean(matched)) if matched else float('nan')


def _prefix_predicate(prefix: str) -> Callable[[str], bool]:
    '''Predicate matching corpus names starting with ``<prefix>_``.'''
    return lambda name: name.startswith(f'{prefix}_')


def _contains_predicate(token: str) -> Callable[[str], bool]:
    '''Predicate matching corpus names containing ``token`` (case-insensitive).'''
    lower = token.lower()
    return lambda name: lower in name.lower()


# ---------------------------------------------------------------------------
# Score matrix and average precision
# ---------------------------------------------------------------------------

def micro_pr_auc(*, gold_matrix: np.ndarray, scores: np.ndarray) -> float:
    '''Compute micro PR-AUC by flattening across all classes.'''
    if gold_matrix.size == 0 or int(gold_matrix.sum()) == 0:
        return float('nan')
    return float(average_precision_score(gold_matrix, scores, average='micro'))


def per_corpus_pr_auc(
    *,
    df: pd.DataFrame,
    scores: np.ndarray,
    gold_matrix: np.ndarray,
) -> dict[str, float]:
    '''Compute per-corpus macro PR-AUC over classes with positive support in that corpus.'''
    corpus_to_value: dict[str, float] = {}
    corpus_names = df['corpus_name'].to_numpy()
    for corpus_name in sorted({name for name in corpus_names if name}):
        mask = corpus_names == corpus_name
        per_class_values = []
        for idx in range(scores.shape[1]):
            y_true = gold_matrix[mask, idx].astype(np.int8)
            if y_true.sum() == 0:
                continue
            per_class_values.append(average_precision(y_true=y_true, y_score=scores[mask, idx]))
        corpus_to_value[corpus_name] = (
            float(np.mean(per_class_values)) if per_class_values else float('nan')
        )
    return corpus_to_value


def average_precision(*, y_true: np.ndarray, y_score: np.ndarray) -> float:
    '''Compute discrete PR-AUC as average precision.'''
    if int(y_true.sum()) == 0:
        return np.nan
    return float(average_precision_score(y_true, y_score))
