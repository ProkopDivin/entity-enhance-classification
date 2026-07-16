'''Article-level F1 deltas, confusion diffs, entity impact, and top-change analysis.'''

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from iptc_entity_pipeline.evaluation._confusion import (
    build_pred_matrix,
    confusion_counts,
)
from iptc_entity_pipeline.evaluation.comparison_tables import AGG_CLASS_ROWS
from iptc_entity_pipeline.evaluation.run_loading import GoldLabelMap

CHANGE_THRESHOLDS: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7)
TOP_CHANGE_N: int = 100


# ---------------------------------------------------------------------------
# Article-level alignment helpers
# ---------------------------------------------------------------------------

def shared_article_ids(*, current_df: pd.DataFrame, base_df: pd.DataFrame) -> list[str]:
    '''Return article ids shared by current and base runs in current order.'''
    base_ids = set(base_df['article_id'])
    article_ids = [article_id for article_id in current_df['article_id'] if article_id in base_ids]
    if not article_ids:
        raise ValueError('Current and base runs do not share any aligned article_id values.')
    return article_ids


def subset_by_ids(*, df: pd.DataFrame, article_ids: Sequence[str]) -> pd.DataFrame:
    '''Keep rows for the requested article ids in that exact order.'''
    if not article_ids:
        return df.iloc[0:0].copy()
    return df.set_index('article_id', drop=False).loc[list(article_ids)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Article-level F1 deltas
# ---------------------------------------------------------------------------

def build_article_f1_diff_df(
    *,
    current_df: pd.DataFrame,
    base_df: pd.DataFrame,
    gold_map: GoldLabelMap,
    cat_ids: Sequence[str],
    current_thr_vec: np.ndarray,
    base_thr_vec: np.ndarray,
) -> pd.DataFrame:
    '''Compute per-article F1 for current/base and their delta.

    :param current_thr_vec: Per-class threshold vector for the current run.
    :param base_thr_vec: Per-class threshold vector for the base run.
    '''
    article_ids = list(current_df['article_id'])
    gold_matrix = gold_map.gold_matrix(article_ids=article_ids, cat_ids=cat_ids)
    current_pred = build_pred_matrix(df=current_df, cat_ids=cat_ids, thr_vec=current_thr_vec)
    base_pred = build_pred_matrix(df=base_df, cat_ids=cat_ids, thr_vec=base_thr_vec)
    current_f1 = compute_article_f1(pred_matrix=current_pred, gold_matrix=gold_matrix)
    base_f1 = compute_article_f1(pred_matrix=base_pred, gold_matrix=gold_matrix)
    article_f1_df = pd.DataFrame(
        {
            'article_id': article_ids,
            'corpus_name': current_df['corpus_name'].tolist(),
            'f1_current': current_f1,
            'f1_base': base_f1,
        }
    )
    article_f1_df['f1_diff'] = article_f1_df['f1_current'] - article_f1_df['f1_base']
    if 'article_length' in current_df.columns:
        article_f1_df['article_length'] = pd.to_numeric(current_df['article_length'], errors='coerce')
    return article_f1_df


def compute_article_f1(*, pred_matrix: np.ndarray, gold_matrix: np.ndarray) -> np.ndarray:
    '''Compute per-article F1 scores from prediction and gold matrices.

    :param pred_matrix: Boolean ancestor-normalized prediction matrix.
    '''
    tp, fp, fn, _tn = confusion_counts(pred=pred_matrix, gold=gold_matrix, axis=1)
    denom = 2 * tp + fp + fn
    f1 = np.zeros_like(denom, dtype=float)
    valid = denom > 0
    f1[valid] = (2 * tp[valid]) / denom[valid]
    return f1


# ---------------------------------------------------------------------------
# Article-level confusion diffs
# ---------------------------------------------------------------------------

def build_article_confusion_diff_df(
    *,
    current_df: pd.DataFrame,
    base_df: pd.DataFrame,
    gold_map: GoldLabelMap,
    cat_ids: Sequence[str],
    current_thr_vec: np.ndarray,
    base_thr_vec: np.ndarray,
) -> pd.DataFrame:
    '''Build per-article TP/TN/FP/FN diff (current minus base).

    :param current_thr_vec: Per-class threshold vector for the current run.
    :param base_thr_vec: Per-class threshold vector for the base run.
    '''
    article_ids = list(current_df['article_id'])
    gold_matrix = gold_map.gold_matrix(article_ids=article_ids, cat_ids=cat_ids)
    current_pred = build_pred_matrix(df=current_df, cat_ids=cat_ids, thr_vec=current_thr_vec)
    base_pred = build_pred_matrix(df=base_df, cat_ids=cat_ids, thr_vec=base_thr_vec)
    current_conf = compute_article_confusion(pred_matrix=current_pred, gold_matrix=gold_matrix)
    base_conf = compute_article_confusion(pred_matrix=base_pred, gold_matrix=gold_matrix)
    return pd.DataFrame(
        {
            'article_id': article_ids,
            'corpus_name': current_df['corpus_name'].tolist(),
            'tp_diff': (current_conf['tp'] - base_conf['tp']).astype(int),
            'tn_diff': (current_conf['tn'] - base_conf['tn']).astype(int),
            'fp_diff': (current_conf['fp'] - base_conf['fp']).astype(int),
            'fn_diff': (current_conf['fn'] - base_conf['fn']).astype(int),
        }
    )


def compute_article_confusion(
    *, pred_matrix: np.ndarray, gold_matrix: np.ndarray,
) -> Mapping[str, np.ndarray]:
    '''Compute per-article TP/TN/FP/FN counts from prediction and gold matrices.

    :param pred_matrix: Boolean ancestor-normalized prediction matrix.
    '''
    tp, fp, fn, _tn = confusion_counts(pred=pred_matrix, gold=gold_matrix, axis=1)
    tn = np.int32(gold_matrix.shape[1]) - tp - fp - fn
    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}


# ---------------------------------------------------------------------------
# Entity impact analysis
# ---------------------------------------------------------------------------

def build_entity_impact_all_df(
    *,
    current_df: pd.DataFrame,
    article_f1_df: pd.DataFrame,
) -> pd.DataFrame:
    '''Build the complete, un-thresholded entity impact table sorted by entity score.

    :return: All entities ranked by ``entity_score`` with an ``AVG`` footer row.
    '''
    footer_ids = {'gkbid': 'AVG', 'stdform': 'AVG'}
    entity_df = build_entity_impact_df(current_df=current_df, article_f1_df=article_f1_df)
    if entity_df.empty:
        empty = entity_df.reindex(columns=entity_impact_columns())
        return append_avg_footer_row(df=empty, id_col_values=footer_ids)

    all_entities = entity_df.sort_values(by='entity_score', ascending=False)
    return append_avg_footer_row(df=all_entities, id_col_values=footer_ids)


def build_entity_impact_df(*, current_df: pd.DataFrame, article_f1_df: pd.DataFrame) -> pd.DataFrame:
    '''Aggregate entity impact from article F1 deltas.'''
    exploded = explode_entities(df=current_df)
    if exploded.empty:
        return pd.DataFrame(columns=entity_impact_columns())

    exploded = exploded.merge(article_f1_df[['article_id', 'f1_diff']], on='article_id', how='inner')
    if exploded.empty:
        return pd.DataFrame(columns=entity_impact_columns())

    score_df = exploded[['gkbid', 'article_id', 'f1_diff']].drop_duplicates(subset=['gkbid', 'article_id'])
    score_agg = score_df.groupby('gkbid', as_index=False).agg(
        entity_score=('f1_diff', 'sum'),
        article_count=('article_id', 'nunique'),
    )
    relevance_agg = exploded.groupby('gkbid', as_index=False).agg(avg_relevance=('relevance', 'mean'))
    mentions_per_article = exploded.groupby(['gkbid', 'article_id'], as_index=False).agg(
        mention_per_article=('mention_count', 'mean')
    )
    mention_agg = mentions_per_article.groupby('gkbid', as_index=False).agg(
        avg_mentions_count=('mention_per_article', 'mean')
    )
    stdform = choose_mode_by_gkbid(df=exploded, value_col='stdform')
    entity_type = choose_mode_by_gkbid(df=exploded, value_col='entity_type')
    entity_df = (
        score_agg.merge(relevance_agg, on='gkbid', how='left')
        .merge(mention_agg, on='gkbid', how='left')
        .merge(stdform, on='gkbid', how='left')
        .merge(entity_type, on='gkbid', how='left')
    )
    entity_df['normalized'] = entity_df['entity_score'] / entity_df['article_count'].replace(0, np.nan)
    entity_df = entity_df.reindex(columns=entity_impact_columns())
    return entity_df


def explode_entities(*, df: pd.DataFrame) -> pd.DataFrame:
    '''Explode article entities into one row per entity occurrence.'''
    empty_cols = ['article_id', 'gkbid', 'stdform', 'entity_type', 'relevance', 'mention_count']
    if 'entities' not in df.columns or 'article_id' not in df.columns:
        return pd.DataFrame(columns=empty_cols)
    entity_rows = df[['article_id', 'entities']].copy()
    entity_rows = entity_rows.explode('entities')
    entity_rows = entity_rows.dropna(subset=['entities'])
    if entity_rows.empty:
        return pd.DataFrame(columns=empty_cols)
    entity_rows['gkbid'] = entity_rows['entities'].map(lambda item: item.gkb_id if item is not None else None)
    entity_rows['stdform'] = entity_rows['entities'].map(lambda item: item.std_form if item is not None else None)
    entity_rows['entity_type'] = entity_rows['entities'].map(
        lambda item: item.entity_type if item is not None else None
    )
    entity_rows['relevance'] = entity_rows['entities'].map(lambda item: item.relevance if item is not None else None)
    entity_rows['mention_count'] = entity_rows['entities'].map(
        lambda item: item.mention_count if item is not None else None
    )
    entity_rows = entity_rows.drop(columns=['entities'])
    entity_rows['gkbid'] = entity_rows['gkbid'].astype(object)
    entity_rows = entity_rows[entity_rows['gkbid'].notna() & (entity_rows['gkbid'].astype(str).str.len() > 0)]
    entity_rows['relevance'] = pd.to_numeric(entity_rows['relevance'], errors='coerce')
    entity_rows['mention_count'] = pd.to_numeric(entity_rows['mention_count'], errors='coerce')
    return entity_rows


def choose_mode_by_gkbid(*, df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    '''Select representative value per gkbid by most frequent non-empty entry.

    :param value_col: Column whose most frequent non-empty value is picked per gkbid.
    '''
    names = df[['gkbid', value_col]].dropna(subset=['gkbid']).copy()
    names[value_col] = names[value_col].fillna('').astype(str).str.strip()
    names = names[names[value_col] != '']
    if names.empty:
        return pd.DataFrame(columns=['gkbid', value_col])
    counts = names.groupby(['gkbid', value_col], as_index=False).size()
    counts = counts.sort_values(by=['gkbid', 'size', value_col], ascending=[True, False, True])
    return counts.drop_duplicates(subset=['gkbid'], keep='first')[['gkbid', value_col]]


def append_avg_footer_row(*, df: pd.DataFrame, id_col_values: Mapping[str, str]) -> pd.DataFrame:
    '''Append one footer row with numeric means and fixed identifier labels.'''
    footer: dict[str, Any] = {**id_col_values}
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    for col in numeric_cols:
        footer[col] = float(df[col].mean()) if not df.empty else float('nan')
    return pd.concat([df, pd.DataFrame([footer], columns=df.columns)], ignore_index=True)


def entity_impact_columns() -> list[str]:
    '''Return output column order for entity impact tables.'''
    return [
        'gkbid',
        'stdform',
        'entity_type',
        'avg_relevance',
        'avg_mentions_count',
        'entity_score',
        'article_count',
        'normalized',
    ]


# ---------------------------------------------------------------------------
# Top-change analysis
# ---------------------------------------------------------------------------

def build_top_change_dfs(*, classes_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''Build ranked improved and degraded category tables.'''
    df = classes_df[~classes_df['IPTC Category'].isin(AGG_CLASS_ROWS)].copy()
    df['article_frequency'] = df['Data Count_current'].combine_first(df['Data Count_base'])
    df['ranking_score'] = df['F1_diff'].abs()
    cols = [
        'IPTC Category',
        'article_frequency',
        'Precision_current',
        'Precision_base',
        'Precision_diff',
        'Recall_current',
        'Recall_base',
        'Recall_diff',
        'F1_current',
        'F1_base',
        'F1_diff',
    ]
    improved_df = df[df['F1_diff'] > 0].sort_values(
        by=['F1_diff', 'ranking_score', 'article_frequency'],
        ascending=[False, False, False],
        na_position='last',
    )
    degraded_df = df[df['F1_diff'] < 0].sort_values(
        by=['F1_diff', 'ranking_score', 'article_frequency'],
        ascending=[True, False, False],
        na_position='last',
    )
    return improved_df.reindex(columns=cols), degraded_df.reindex(columns=cols)


def build_top_change_stats_dfs(
    *,
    improved_df: pd.DataFrame,
    degraded_df: pd.DataFrame,
    top_n: int = TOP_CHANGE_N,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''Build summary stats tables for the improved and degraded category rankings.'''
    improved_stats = _change_stats_rows(df=improved_df, label='improved', top_n=top_n)
    degraded_stats = _change_stats_rows(df=degraded_df, label='degraded', top_n=top_n)
    return pd.DataFrame(improved_stats), pd.DataFrame(degraded_stats)


def _change_stats_rows(*, df: pd.DataFrame, label: str, top_n: int) -> list[dict[str, Any]]:
    '''Build the metric/value rows for one change-direction table.'''
    abs_diff = df['F1_diff'].abs() if not df.empty else pd.Series(dtype=float)
    rows: list[dict[str, Any]] = [
        {'metric': f'count_{label}', 'value': int(len(df))},
    ]
    for thr in CHANGE_THRESHOLDS:
        rows.append(
            {'metric': f'count_{label}_f1_diff_gt_{thr}', 'value': int((abs_diff > thr).sum())}
        )

    top = df.head(top_n)
    rows.append(
        {
            'metric': f'avg_f1_diff_top_{top_n}',
            'value': float(top['F1_diff'].mean()) if not top.empty else float('nan'),
        }
    )
    rows.append(
        {
            'metric': f'avg_article_frequency_top_{top_n}',
            'value': float(top['article_frequency'].mean()) if not top.empty else float('nan'),
        }
    )

    if not top.empty:
        top_levels = top['IPTC Category'].map(top_level_from_label).value_counts().sort_values(ascending=False)
        for name, count in top_levels.items():
            rows.append({'metric': f'top_level_top_{top_n}::{name}', 'value': int(count)})
    return rows


def top_level_from_label(label: str) -> str:
    '''Extract the IPTC top-level category name from a quoted long-name label.

    Example labels::

        '"sport >> chess (20001154)"'         -> 'sport'
        '"arts+ - arts, culture, ... (...)"'  -> 'arts+'

    :param label: Quoted IPTC long-name label.
    :return: Top-level category name (best-effort string parse).
    '''
    inner = str(label).strip().strip('"').strip()
    delim_positions = [pos for pos in (inner.find(' >'), inner.find(' -')) if pos != -1]
    if delim_positions:
        return inner[: min(delim_positions)].strip()
    paren_idx = inner.find('(')
    return inner[:paren_idx].strip() if paren_idx != -1 else inner
