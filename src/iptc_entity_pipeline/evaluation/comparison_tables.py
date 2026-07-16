'''Comparison table builders and label/formatting utilities.'''

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from iptc_entity_pipeline.evaluation._confusion import (
    build_pred_matrix,
    confusion_counts,
    safe_mean,
)
from iptc_entity_pipeline.evaluation.evaluate import (
    CLASS_RELEVANT_MACRO_ROW,
    get_cat_name,
)
from iptc_entity_pipeline.evaluation.run_loading import GoldLabelMap, RunEval

LOG = logging.getLogger(__name__)

# Output-facing aggregate row labels aligned with the thesis vocabulary.
MATCH_HEAD_ROW = CLASS_RELEVANT_MACRO_ROW
MACRO_HEAD_ROW = 'All_macro_head'
MACRO_TAIL_ROW = 'All_macro_tail'
CLASS_MACRO_ROW = MACRO_HEAD_ROW
CLASS_TAIL_ROW = MACRO_TAIL_ROW
AGG_CLASS_ROWS = frozenset({'All_micro', MACRO_HEAD_ROW, MACRO_TAIL_ROW, 'All_datapoint'})
AGG_CORPUS_ROWS = frozenset({'All_macro_corpora', 'All_micro', 'All_datapoint'})
SUMMARY_ROWS = {
    'micro': ('classes', 'All_micro'),
}
# Support buckets for binned macro metrics. The thesis splits classes at 15 test
# samples (Tail < 15, Head >= 15) and further subdivides the Head into
# 15-100, 100-1000, and 1000+. Bucketing uses ``low <= support < high``.
SUPPORT_BUCKETS: tuple[tuple[int, int, str], ...] = (
    (0, 15, '0-15'),
    (15, 100, '15-100'),
    (100, 1000, '100-1000'),
    (1000, 10**9, '1000+'),
)
MACRO_HEAD_MIN_SUPPORT = 15
LANG_PREFIXES: tuple[str, ...] = ('en', 'es', 'nl', 'fr', 'de', 'cs')

_CMP_METRICS_CORE: tuple[tuple[str, str], ...] = (
    ('precision', 'Precision'),
    ('recall', 'Recall'),
    ('f1', 'F1'),
)
_CMP_METRICS_REPORT: tuple[tuple[str, str], ...] = _CMP_METRICS_CORE
_CMP_METRICS_CLASSES: tuple[tuple[str, str], ...] = _CMP_METRICS_CORE + (
    ('false_positive_count', 'False Positive Count'),
)
_CMP_METRICS_CORPORA: tuple[tuple[str, str], ...] = _CMP_METRICS_CORE + (
    ('false_positive_rate', 'False Positive Rate'),
)
# Per-class diagnostics keep the absolute False Positive count for the current
# run; the base/diff FP columns are dropped to keep ``classes_comparison`` compact.
_CLASSES_COMPARISON_DROP_COLS: tuple[str, ...] = (
    'False Positive Count_base',
    'False Positive Count_diff',
)


# ---------------------------------------------------------------------------
# Label / formatting utilities
# ---------------------------------------------------------------------------

def safe_cat_label(*, cat_id: str) -> str:
    '''Return the quoted long name used by other tables, falling back to the id.'''
    try:
        return '"' + get_cat_name(cat_id) + '"'
    except KeyError:
        LOG.warning('No IPTC name for cat_id=%s; using raw id in table', cat_id)
        return cat_id


def label_from_cat_id(*, cat_id: str) -> str:
    '''Return class label format used in per-class tables for one cat id.'''
    return '"' + get_cat_name(cat_id) + '"'


def labels_for_cat_ids(*, cat_ids: set[str]) -> dict[str, str]:
    '''Map class table label to category id for requested ids.'''
    return {label_from_cat_id(cat_id=cat_id): cat_id for cat_id in cat_ids}


def build_label_to_cat_id_map(*, cat_ids: Sequence[str]) -> dict[str, str]:
    '''Build mapping from class label to raw class id for stable joins.'''
    return {safe_cat_label(cat_id=cat_id): cat_id for cat_id in cat_ids}


def format_class_id(cat_id: Any) -> str:
    '''Format a class id as plain ``<id>`` or empty string for missing values.'''
    if cat_id is None or (isinstance(cat_id, float) and np.isnan(cat_id)):
        return ''
    cat_id_str = str(cat_id).strip()
    if not cat_id_str:
        return ''
    if cat_id_str.startswith('(') and cat_id_str.endswith(')'):
        return cat_id_str[1:-1].strip()
    return cat_id_str


def with_class_id_column(
    *,
    df: pd.DataFrame,
    key_col: str,
    label_to_cat_id: Mapping[str, str],
) -> pd.DataFrame:
    '''Attach ``class_id`` column to one class-level dataframe.'''
    result = df.copy()
    if key_col not in result.columns and result.index.name == key_col:
        result = result.reset_index()
    if 'class_id' in result.columns:
        result['class_id'] = result['class_id'].map(format_class_id)
        return result
    if 'cat_id' in result.columns:
        cat_ids = result['cat_id']
    elif key_col in result.columns:
        cat_ids = result[key_col].map(label_to_cat_id)
    else:
        result['class_id'] = ''
        return result
    result['class_id'] = cat_ids.map(format_class_id)
    if key_col in result.columns:
        cols = [key_col, 'class_id']
        cols.extend(col for col in result.columns if col not in cols)
        return result.loc[:, cols]
    return result


# ---------------------------------------------------------------------------
# Comparison table builders
# ---------------------------------------------------------------------------

def build_cmp_df(
    *,
    current_df: pd.DataFrame,
    base_df: pd.DataFrame,
    key_col: str,
    info_cols: Sequence[str],
    cmp_metrics: tuple[tuple[str, str], ...] = _CMP_METRICS_CLASSES,
) -> pd.DataFrame:
    '''Join current/base metric tables and compute current-base deltas.'''
    metric_cols = [col for _, col in cmp_metrics]
    df = current_df.reset_index().merge(
        base_df.reset_index(),
        on=key_col,
        how='outer',
        suffixes=('_current', '_base'),
        sort=False,
    )
    for _, mcol in cmp_metrics:
        for suffix in ('_current', '_base'):
            name = f'{mcol}{suffix}'
            if name not in df.columns:
                df[name] = np.nan
    for metric in metric_cols:
        df[f'{metric}_diff'] = df[f'{metric}_current'] - df[f'{metric}_base']

    cols = [key_col]
    for info_col in info_cols:
        cols.extend([f'{info_col}_current', f'{info_col}_base'])
    for metric in metric_cols:
        cols.extend([f'{metric}_current', f'{metric}_base', f'{metric}_diff'])
    return df.reindex(columns=cols)


def diff_only_df(
    *,
    df: pd.DataFrame,
    key_col: str,
    cmp_metrics: tuple[tuple[str, str], ...] = _CMP_METRICS_CLASSES,
) -> pd.DataFrame:
    '''Drop base metric columns while preserving current values and diffs.'''
    drop_cols = {f'{col}_base' for _, col in cmp_metrics}
    cols = [key_col]
    cols.extend(col for col in df.columns if col != key_col and col not in drop_cols)
    return df.loc[:, cols]


def build_language_cmp_df(
    *,
    corpora_cmp: pd.DataFrame,
    key_col: str = 'Language',
) -> pd.DataFrame:
    '''Aggregate per-corpus comparison rows into per-language rows (macro over corpora).'''
    corpora = corpora_cmp[~corpora_cmp['Corpus Name'].isin(AGG_CORPUS_ROWS)].copy()
    corpora['Language'] = corpora['Corpus Name'].map(language_from_corpus_name)
    corpora = corpora[corpora['Language'].notna()].copy()
    if corpora.empty:
        return pd.DataFrame(columns=_language_cmp_columns(key_col=key_col))

    metric_cols = [col for _, col in _CMP_METRICS_CORPORA]
    grouped = corpora.groupby('Language', sort=True)
    rows: list[dict[str, Any]] = []
    for language, group in grouped:
        row: dict[str, Any] = {key_col: language}
        for info_col in ('Data Count', 'Docs No Labels', 'Decent Labels'):
            row[f'{info_col}_current'] = _safe_int(group[f'{info_col}_current'].sum())
            row[f'{info_col}_base'] = _safe_int(group[f'{info_col}_base'].sum())
        for metric_col in metric_cols:
            row[f'{metric_col}_current'] = safe_mean(group[f'{metric_col}_current'])
            row[f'{metric_col}_base'] = safe_mean(group[f'{metric_col}_base'])
            row[f'{metric_col}_diff'] = row[f'{metric_col}_current'] - row[f'{metric_col}_base']
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.reindex(columns=_language_cmp_columns(key_col=key_col))


def build_corpora_macro_head_cmp_df(
    *,
    current_df: pd.DataFrame,
    base_df: pd.DataFrame,
    gold_map: GoldLabelMap,
    cat_ids: Sequence[str],
    current_thr_vec: np.ndarray,
    base_thr_vec: np.ndarray,
    corpora_cmp_reference: pd.DataFrame,
    min_support: int = MACRO_HEAD_MIN_SUPPORT,
) -> pd.DataFrame:
    '''Build corpus-level comparison table with macro-head metrics (support >= ``min_support``).'''
    article_ids = list(current_df['article_id'])
    gold_matrix = gold_map.gold_matrix(article_ids=article_ids, cat_ids=cat_ids).astype(bool)
    current_pred = build_pred_matrix(df=current_df, cat_ids=cat_ids, thr_vec=current_thr_vec)
    base_pred = build_pred_matrix(df=base_df, cat_ids=cat_ids, thr_vec=base_thr_vec)
    support = gold_matrix.sum(axis=0, dtype=np.int64)
    head_mask = support >= int(min_support)
    corpus_names = current_df['corpus_name'].fillna('').astype(str)
    row_names = sorted({name for name in corpus_names if name})
    info_lookup = corpora_cmp_reference.set_index('Corpus Name', drop=False)
    rows: list[dict[str, Any]] = []

    for corpus_name in row_names:
        mask = corpus_names == corpus_name
        current_stats = _macro_head_metrics(gold=gold_matrix[mask][:, head_mask], pred=current_pred[mask][:, head_mask])
        base_stats = _macro_head_metrics(gold=gold_matrix[mask][:, head_mask], pred=base_pred[mask][:, head_mask])
        info_row = info_lookup.loc[corpus_name] if corpus_name in info_lookup.index else None
        if isinstance(info_row, pd.DataFrame):
            info_row = info_row.iloc[0]
        rows.append(
            _macro_head_corpora_row(
                corpus_name=corpus_name,
                current_stats=current_stats,
                base_stats=base_stats,
                info_row=info_row,
            )
        )

    current_stats_all = _macro_head_metrics(gold=gold_matrix[:, head_mask], pred=current_pred[:, head_mask])
    base_stats_all = _macro_head_metrics(gold=gold_matrix[:, head_mask], pred=base_pred[:, head_mask])
    all_info_row = info_lookup.loc['All_micro'] if 'All_micro' in info_lookup.index else None
    if isinstance(all_info_row, pd.DataFrame):
        all_info_row = all_info_row.iloc[0]
    rows.append(
        _macro_head_corpora_row(
            corpus_name='All_macro_head_corpora',
            current_stats=current_stats_all,
            base_stats=base_stats_all,
            info_row=all_info_row,
        )
    )
    df = pd.DataFrame(rows)
    return df.reindex(columns=corpora_cmp_reference.columns.tolist())


# ---------------------------------------------------------------------------
# Macro-head metrics helper
# ---------------------------------------------------------------------------

def _macro_head_metrics(*, gold: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    '''Compute macro-head metrics for one document subset and class subset.'''
    if gold.size == 0 or pred.size == 0 or gold.shape[1] == 0:
        return {
            'Data Count': int(gold.shape[0]) if gold.ndim == 2 else 0,
            'Docs No Labels': int(gold.shape[0]) if gold.ndim == 2 else 0,
            'Decent Labels': 0,
            'Precision': float('nan'),
            'Recall': float('nan'),
            'F1': float('nan'),
            'False Positive Rate': float('nan'),
        }

    tp, fp, fn, _tn = confusion_counts(pred=pred, gold=gold, axis=0)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision, dtype=float),
        where=(precision + recall) > 0,
    )
    beta_sq = 0.4 * 0.4
    f_beta = np.divide(
        (1.0 + beta_sq) * precision * recall,
        beta_sq * precision + recall,
        out=np.zeros_like(precision, dtype=float),
        where=(beta_sq * precision + recall) > 0,
    )
    support = gold.sum(axis=0, dtype=np.int64)
    decent_mask = (precision >= 0.6) & (f_beta >= 0.5) & (support >= 10)
    docs_no_labels = int((pred.sum(axis=1, dtype=np.int64) == 0).sum())
    negatives = int(np.logical_not(gold).sum(dtype=np.int64))
    fp_total = int(fp.sum(dtype=np.int64))
    false_positive_rate = float(fp_total / negatives) if negatives > 0 else float('nan')
    # Restrict the macro average to head classes present in this subset; absent
    # classes (zero gold here) would otherwise enter the mean as F1=0 and deflate
    # per-corpus numbers.
    present = support > 0
    return {
        'Data Count': int(gold.shape[0]),
        'Docs No Labels': docs_no_labels,
        'Decent Labels': int(decent_mask.sum(dtype=np.int64)),
        'Precision': float(precision[present].mean()) if present.any() else float('nan'),
        'Recall': float(recall[present].mean()) if present.any() else float('nan'),
        'F1': float(f1[present].mean()) if present.any() else float('nan'),
        'False Positive Rate': false_positive_rate,
    }


def _macro_head_corpora_row(
    *,
    corpus_name: str,
    current_stats: Mapping[str, float],
    base_stats: Mapping[str, float],
    info_row: pd.Series | None,
) -> dict[str, Any]:
    '''Build one corpus row with macro-head metrics and info columns.'''
    row: dict[str, Any] = {'Corpus Name': corpus_name}
    for info_col in ('Data Count', 'Docs No Labels', 'Decent Labels'):
        default_current = _safe_int(current_stats.get(info_col, float('nan')))
        default_base = _safe_int(base_stats.get(info_col, float('nan')))
        row[f'{info_col}_current'] = _safe_row_int(row=info_row, col=f'{info_col}_current', default=default_current)
        row[f'{info_col}_base'] = _safe_row_int(row=info_row, col=f'{info_col}_base', default=default_base)
    for _, metric_col in _CMP_METRICS_CORPORA:
        current_value = float(current_stats.get(metric_col, float('nan')))
        base_value = float(base_stats.get(metric_col, float('nan')))
        row[f'{metric_col}_current'] = current_value
        row[f'{metric_col}_base'] = base_value
        row[f'{metric_col}_diff'] = current_value - base_value
    return row


def _safe_row_int(*, row: pd.Series | None, col: str, default: int) -> int:
    '''Read integer column from optional row, else return default.'''
    if row is None:
        return default
    value = row.get(col, np.nan)
    return _safe_int(value, default=default)


def _safe_int(value: Any, default: int = 0) -> int:
    '''Convert numeric value to ``int`` with fallback for missing values.'''
    if value is None:
        return default
    try:
        if np.isnan(value):
            return default
    except TypeError:
        pass
    return int(value)


def _language_cmp_columns(*, key_col: str) -> list[str]:
    '''Column layout shared by language comparison tables.'''
    columns = [key_col]
    for info_col in ('Data Count', 'Docs No Labels', 'Decent Labels'):
        columns.extend([f'{info_col}_current', f'{info_col}_base'])
    for _, metric_col in _CMP_METRICS_CORPORA:
        columns.extend([f'{metric_col}_current', f'{metric_col}_base', f'{metric_col}_diff'])
    return columns


def language_from_corpus_name(corpus_name: Any) -> str | None:
    '''Extract language prefix from ``<lang>_...`` corpus names.'''
    name = str(corpus_name).strip()
    if '_' not in name:
        return None
    prefix = name.split('_', 1)[0].lower()
    return prefix if prefix in LANG_PREFIXES else None


# ---------------------------------------------------------------------------
# Validation and subset helpers
# ---------------------------------------------------------------------------

def validate_subset_ids_in_corpora(
    *,
    current_corpus: Any,
    base_corpus: Any,
    subset_ids: set[str],
    subset_name: str,
    require_all: bool = True,
) -> None:
    '''Validate subset ids against corpus cat lists as an early-fast check.'''
    current_ids = {str(cat_id) for cat_id in getattr(current_corpus, 'catList', [])}
    base_ids = {str(cat_id) for cat_id in getattr(base_corpus, 'catList', [])}
    if not require_all:
        return
    missing_current = sorted(subset_ids - current_ids)
    missing_base = sorted(subset_ids - base_ids)
    if missing_current or missing_base:
        cur_preview = ', '.join(missing_current[:10]) if missing_current else 'none'
        base_preview = ', '.join(missing_base[:10]) if missing_base else 'none'
        raise ValueError(
            'Missing required subset classes at startup: '
            f'subset={subset_name} '
            f'missing_in_current={cur_preview} '
            f'missing_in_base={base_preview}'
        )


def class_subset_by_ids(
    *,
    classes_cmp: pd.DataFrame,
    class_ids: set[str],
    require_all: bool,
    subset_name: str,
) -> pd.DataFrame:
    '''Filter non-aggregate class rows to a category-id subset.'''
    classes_filtered = classes_cmp[~classes_cmp['IPTC Category'].isin(AGG_CLASS_ROWS)].copy()
    label_to_id = labels_for_cat_ids(cat_ids=class_ids)
    classes_filtered['cat_id'] = classes_filtered['IPTC Category'].map(label_to_id).astype(object)
    present_ids = set(classes_filtered['cat_id'].dropna().astype(str).tolist())
    if require_all:
        missing = sorted(class_ids - present_ids)
        if missing:
            preview = ', '.join(missing[:10])
            raise ValueError(f'Missing {subset_name} classes in comparison output: {preview}')
    subset = classes_filtered[classes_filtered['cat_id'].isin(class_ids)].copy()
    if subset.empty:
        raise ValueError(f'No rows matched {subset_name} category set')
    return subset


def replace_macro_row_in_run_classes_df(
    *,
    classes_df: pd.DataFrame,
    class_ids: set[str],
    row_label: str = CLASS_MACRO_ROW,
    subset_name: str,
    require_all: bool,
    insert_after_label: str | None = None,
    match_label: str | None = None,
) -> pd.DataFrame:
    '''Rebuild one run's macro aggregate row as the mean of its per-class metrics.

    The rebuilt row uses standard macro averaging (mean of per-class Precision,
    Recall, F1 and False Positive Count over ``class_ids``). When ``match_label``
    is given, an existing row with that label is located and relabeled to
    ``row_label`` in place; this maps ``evaluate.py``'s ``All_macro_relevant``
    row onto the thesis ``All_macro_head`` label. When no matching row exists the
    row is inserted (optionally after ``insert_after_label``).
    '''
    lookup_label = match_label if match_label is not None else row_label
    table = classes_df.reset_index().copy()
    classes_filtered = table[~table['IPTC Category'].isin(AGG_CLASS_ROWS)].copy()
    label_to_id = labels_for_cat_ids(cat_ids=class_ids)
    classes_filtered['cat_id'] = classes_filtered['IPTC Category'].map(label_to_id).astype(object)
    subset = classes_filtered[classes_filtered['cat_id'].isin(class_ids)].copy()
    if subset.empty:
        raise ValueError(f'No rows matched {subset_name} category set for classes row replacement')
    if require_all:
        present_ids = set(classes_filtered['cat_id'].dropna().astype(str))
        missing = sorted(class_ids - present_ids)
        if missing:
            preview = ', '.join(missing[:10])
            raise ValueError(f'Missing {subset_name} classes in classes table: {preview}')

    row: dict[str, Any] = {'IPTC Category': row_label}
    row['Data Count'] = int(len(subset))
    for col in ('Precision', 'Recall', 'F1', 'False Positive Count'):
        if col in table.columns:
            row[col] = float(subset[col].mean())

    mask = table['IPTC Category'] == lookup_label
    if mask.any():
        for key, value in row.items():
            table.loc[mask, key] = value
    else:
        insert_at: int | None = None
        if insert_after_label is not None:
            after_idx = table.index[table['IPTC Category'] == insert_after_label]
            if len(after_idx) > 0:
                insert_at = int(after_idx[0]) + 1
        row_df = pd.DataFrame([row])
        if insert_at is None:
            table = pd.concat([table, row_df], ignore_index=True)
        else:
            table = pd.concat(
                [table.iloc[:insert_at], row_df, table.iloc[insert_at:]],
                ignore_index=True,
            )
    return table.set_index('IPTC Category')


def apply_macro_rows(
    *,
    classes_df: pd.DataFrame,
    relevant_cat_ids: set[str],
    tail_cat_ids: set[str],
) -> pd.DataFrame:
    '''Apply head and tail macro-row replacements to one run's classes table.'''
    result = replace_macro_row_in_run_classes_df(
        classes_df=classes_df,
        class_ids=relevant_cat_ids,
        row_label=CLASS_MACRO_ROW,
        subset_name='head',
        require_all=False,
        match_label=MATCH_HEAD_ROW,
    )
    return replace_macro_row_in_run_classes_df(
        classes_df=result,
        class_ids=tail_cat_ids,
        row_label=CLASS_TAIL_ROW,
        subset_name='tail',
        require_all=True,
        insert_after_label=CLASS_MACRO_ROW,
    )


# ---------------------------------------------------------------------------
# Confusion count and threshold tables
# ---------------------------------------------------------------------------

def build_class_confusion_counts_df(
    *,
    current_df: pd.DataFrame,
    base_df: pd.DataFrame,
    gold_map: GoldLabelMap,
    cat_ids: Sequence[str],
    current_thr_vec: np.ndarray,
    base_thr_vec: np.ndarray,
) -> pd.DataFrame:
    '''Build per-class confusion-count comparison table for current vs base.'''
    article_ids = list(current_df['article_id'])
    gold_matrix = gold_map.gold_matrix(article_ids=article_ids, cat_ids=cat_ids).astype(bool)
    current_pred = build_pred_matrix(df=current_df, cat_ids=cat_ids, thr_vec=current_thr_vec)
    base_pred = build_pred_matrix(df=base_df, cat_ids=cat_ids, thr_vec=base_thr_vec)

    tp_cur, fp_cur, fn_cur, tn_cur = confusion_counts(pred=current_pred, gold=gold_matrix, axis=0)
    tp_base, fp_base, fn_base, tn_base = confusion_counts(pred=base_pred, gold=gold_matrix, axis=0)

    rows: list[dict[str, Any]] = []
    for idx, cat_id in enumerate(cat_ids):
        rows.append(
            {
                'IPTC Category': safe_cat_label(cat_id=cat_id),
                'class_id': format_class_id(cat_id=cat_id),
                'Fp_current': int(fp_cur[idx]),
                'Fp_base': int(fp_base[idx]),
                'Fp_diff': int(fp_cur[idx] - fp_base[idx]),
                'Fn_current': int(fn_cur[idx]),
                'Fn_base': int(fn_base[idx]),
                'Fn_diff': int(fn_cur[idx] - fn_base[idx]),
                'Tp_current': int(tp_cur[idx]),
                'Tp_base': int(tp_base[idx]),
                'Tp_diff': int(tp_cur[idx] - tp_base[idx]),
                'Tn_current': int(tn_cur[idx]),
                'Tn_base': int(tn_base[idx]),
                'Tn_diff': int(tn_cur[idx] - tn_base[idx]),
            }
        )
    return pd.DataFrame(rows)


def build_class_thresholds_df(
    *,
    cat_ids: Sequence[str],
    default_threshold: float,
    current_thresholds: Mapping[str, float],
    base_thresholds: Mapping[str, float],
    class_supports: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    '''Build per-class threshold table for current/base runs.'''
    supports = class_supports or {}
    rows: list[dict[str, Any]] = []
    for cat_id in cat_ids:
        current_thr = float(current_thresholds.get(cat_id, default_threshold))
        base_thr = float(base_thresholds.get(cat_id, default_threshold))
        rows.append(
            {
                'IPTC Category': safe_cat_label(cat_id=cat_id),
                'class_id': format_class_id(cat_id=cat_id),
                'count': int(supports.get(cat_id, 0)),
                'threshold_current': current_thr,
                'threshold_base': base_thr,
                'threshold_diff': current_thr - base_thr,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary and metric row helpers
# ---------------------------------------------------------------------------

def _metric_row(
    *,
    summary_key: str,
    current: pd.Series,
    base: pd.Series,
    cmp_metrics: tuple[tuple[str, str], ...],
) -> dict[str, Any]:
    '''Build one summary row from current/base metric series.'''
    row: dict[str, Any] = {'summary_key': summary_key}
    for key, col in cmp_metrics:
        row[f'{key}_current'] = current[col]
        row[f'{key}_base'] = base[col]
        row[f'{key}_diff'] = current[col] - base[col]
    return row


def _avg_metrics_row(
    *,
    summary_key: str,
    sub_df: pd.DataFrame,
    cmp_metrics: tuple[tuple[str, str], ...],
) -> dict[str, Any]:
    '''Macro-average precision/recall/f1 over the rows of ``sub_df``.'''
    row: dict[str, Any] = {'summary_key': summary_key}
    if sub_df.empty:
        nan = float('nan')
        for key, _ in cmp_metrics:
            row[f'{key}_current'] = nan
            row[f'{key}_base'] = nan
            row[f'{key}_diff'] = nan
        return row
    for key, col in cmp_metrics:
        row[f'{key}_current'] = float(sub_df[f'{col}_current'].mean())
        row[f'{key}_base'] = float(sub_df[f'{col}_base'].mean())
        row[f'{key}_diff'] = float(sub_df[f'{col}_diff'].mean())
    return row


def build_summary_df(
    *,
    current_run: RunEval,
    base_run: RunEval,
    classes_cmp: pd.DataFrame,
    relevant_cat_ids: set[str],
    tail_cat_ids: set[str],
) -> pd.DataFrame:
    '''Build compact summary rows from aggregate evaluation outputs.'''
    rows: list[dict[str, Any]] = []
    tables = {
        'corpora_current': current_run.corpora_df,
        'corpora_base': base_run.corpora_df,
        'classes_current': current_run.classes_df,
        'classes_base': base_run.classes_df,
    }
    for summary_key, (group_name, row_name) in SUMMARY_ROWS.items():
        current_row = tables[f'{group_name}_current'].loc[row_name]
        base_row = tables[f'{group_name}_base'].loc[row_name]
        mets = _CMP_METRICS_REPORT
        rows.append(
            _metric_row(summary_key=summary_key, current=current_row, base=base_row, cmp_metrics=mets),
        )

    classes_filtered = classes_cmp[~classes_cmp['IPTC Category'].isin(AGG_CLASS_ROWS)].copy()
    classes_head = class_subset_by_ids(
        classes_cmp=classes_cmp,
        class_ids=relevant_cat_ids,
        require_all=False,
        subset_name='head',
    )
    rows.append(
        _avg_metrics_row(
            summary_key='macro_head',
            sub_df=classes_head,
            cmp_metrics=_CMP_METRICS_REPORT,
        )
    )
    classes_tail = class_subset_by_ids(
        classes_cmp=classes_cmp,
        class_ids=tail_cat_ids,
        require_all=True,
        subset_name='tail',
    )
    rows.append(
        _avg_metrics_row(
            summary_key='macro_tail',
            sub_df=classes_tail,
            cmp_metrics=_CMP_METRICS_REPORT,
        )
    )

    classes_filtered['support'] = classes_filtered['Data Count_current'].combine_first(
        classes_filtered['Data Count_base']
    )
    for low, high, label in SUPPORT_BUCKETS:
        mask = (classes_filtered['support'] >= low) & (classes_filtered['support'] < high)
        sub = classes_filtered[mask]
        rows.append(
            _avg_metrics_row(
                summary_key=f'macro_support_{label}',
                sub_df=sub,
                cmp_metrics=_CMP_METRICS_REPORT,
            )
        )
    return pd.DataFrame(rows)
