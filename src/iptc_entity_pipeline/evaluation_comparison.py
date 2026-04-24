"""Compare evaluation outputs between a current and base run.

Each run is a directory holding two pickled artifacts produced by
:func:`iptc_entity_pipeline.model_io.save_final_model_outputs`:

- ``predictions.pkl``: raw weighted predictions, ``list[list[tuple[str, float]]]``
  aligned positionally with the corpus.
- ``eval_corpus.pkl``: the ``geneea.catlib.data.Corpus`` used during evaluation,
  carrying gold labels and ``corpusName`` metadata.

Gold labels for cross-run metrics are derived from the *current* run's corpus.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

if __package__ is None or __package__ == '':
    src_root = Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

from iptc_entity_pipeline.data_loading import sanitize_name
from iptc_entity_pipeline.evaluate import REMOVED_CAT_IDS, evaluate_predictions, get_cat_name, get_iptc_topics
from iptc_entity_pipeline.model_io import EVAL_CORPUS_FILENAME, PREDICTIONS_FILENAME

LOG = logging.getLogger(__name__)

AGG_CLASS_ROWS = frozenset({'All - micro avg', 'All - macro avg', 'All - datapoint avg'})
AGG_CORPUS_ROWS = frozenset({'All-macro', 'All-micro', 'All-datapoint'})
SUMMARY_ROWS = {
    'macro_over_classes_all': ('classes', 'All - macro avg'),
    'micro_over_labels': ('classes', 'All - micro avg'),
}
SUPPORT_BUCKETS: tuple[tuple[int, int, str], ...] = (
    (0, 10, '0-10'),
    (10, 100, '10-100'),
    (100, 1000, '100-1000'),
    (1000, 10000, '1000-10000'),
)
LANG_PREFIXES: tuple[str, ...] = ('en', 'es', 'nl', 'fr', 'de', 'cs')
EUROSPORT_TOKEN = 'eurosport'


@dataclass(frozen=True)
class ComparisonResult:
    """Structured outputs from one evaluation comparison run."""

    corpora_comparison: pd.DataFrame
    classes_comparison: pd.DataFrame
    summary_comparison: pd.DataFrame
    top_improved_categories: pd.DataFrame
    top_degraded_categories: pd.DataFrame
    top_improved_stats: pd.DataFrame
    top_degraded_stats: pd.DataFrame
    hamming_loss_comparison: pd.DataFrame
    pr_auc_per_class: pd.DataFrame
    pr_auc_summary: pd.DataFrame
    current_corpora: pd.DataFrame
    current_classes: pd.DataFrame
    base_corpora: pd.DataFrame
    base_classes: pd.DataFrame
    excel_path: Path | None = None


@dataclass(frozen=True)
class GoldArticle:
    """Gold metadata for one article."""

    article_id: str
    corpus_name: str
    gold_categories: tuple[str, ...]


@dataclass(frozen=True)
class GoldLabelMap:
    """Gold labels loaded once and reused across all comparisons."""

    df: pd.DataFrame
    article_map: Mapping[str, GoldArticle]

    @classmethod
    def from_corpus(cls, *, corpus: Any) -> 'GoldLabelMap':
        """Build a gold label map from a ``geneea.catlib.data.Corpus``."""
        df = gold_df_from_corpus(corpus=corpus)
        article_map = {
            row.article_id: GoldArticle(
                article_id=row.article_id,
                corpus_name=row.corpus_name,
                gold_categories=row.gold_categories,
            )
            for row in df.itertuples(index=False)
        }
        return cls(df=df, article_map=article_map)

    def cat_ids(self, *, prob_dfs: Sequence[pd.DataFrame]) -> list[str]:
        """Collect category ids from gold labels plus all probability tables."""
        cat_ids = {cat_id for gold in self.article_map.values() for cat_id in gold.gold_categories}
        for prob_df in prob_dfs:
            cat_ids.update(col.removeprefix('prob_') for col in prob_df.columns if col.startswith('prob_'))
        return sorted(cat_ids)

    def gold_matrix(self, *, article_ids: Sequence[str], cat_ids: Sequence[str]) -> np.ndarray:
        """Build gold binary matrix for selected articles and categories."""
        cat_to_idx = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}
        matrix = np.zeros((len(article_ids), len(cat_ids)), dtype=np.int8)
        for row_idx, article_id in enumerate(article_ids):
            gold = self.article_map.get(article_id)
            if gold is None:
                continue
            for cat_id in gold.gold_categories:
                cat_idx = cat_to_idx.get(cat_id)
                if cat_idx is not None:
                    matrix[row_idx, cat_idx] = 1
        return matrix


@dataclass(frozen=True)
class RunEval:
    """One run after table rebuild and per-article alignment."""

    aligned_df: pd.DataFrame
    corpora_df: pd.DataFrame
    classes_df: pd.DataFrame


def load_run(*, run_dir: str | Path) -> tuple[list[list[tuple[str, float]]], Any]:
    """Load raw predictions and eval corpus from a saved run directory.

    :param run_dir: Directory containing ``predictions.pkl`` and ``eval_corpus.pkl``.
    :return: ``(pred_scores, eval_corpus)`` tuple aligned positionally.
    """
    run_path = Path(run_dir)
    predictions_path = run_path / PREDICTIONS_FILENAME
    eval_corpus_path = run_path / EVAL_CORPUS_FILENAME
    with open(predictions_path, 'rb') as f:
        pred_scores = pickle.load(f)
    with open(eval_corpus_path, 'rb') as f:
        eval_corpus = pickle.load(f)
    if len(pred_scores) != len(eval_corpus):
        raise ValueError(
            f'Misaligned run {run_path}: predictions={len(pred_scores)} corpus={len(eval_corpus)}'
        )
    return pred_scores, eval_corpus


def compare_runs(
    *,
    current_run_dir: str | Path,
    base_run_dir: str | Path,
    threshold_eval: float,
    averaging_type: str = 'datapoint',
    top_n: int = 20,
    only_diff: bool = False,
    output_path: str | Path | None = None,
) -> ComparisonResult:
    """Compare current and base runs loaded from their saved directories.

    Gold labels are derived from the current run's pickled corpus and reused
    for both runs' cross-run metrics (Hamming loss, PR-AUC).

    :param current_run_dir: Directory of the current run.
    :param base_run_dir: Directory of the base run.
    :param threshold_eval: Threshold applied during evaluation.
    :param averaging_type: One of ``'datapoint'``, ``'micro'``, ``'macro'``.
    :param top_n: Preview size for improved/degraded categories logging.
    :param only_diff: Drop base metric columns from comparison sheets.
    :param output_path: Optional Excel workbook path.
    :return: Structured :class:`ComparisonResult`.
    """
    current_pred, current_corpus = load_run(run_dir=current_run_dir)
    base_pred, base_corpus = load_run(run_dir=base_run_dir)

    gold_map = GoldLabelMap.from_corpus(corpus=current_corpus)

    current_run = build_run(
        pred_scores=current_pred,
        eval_corpus=current_corpus,
        threshold_eval=threshold_eval,
        averaging_type=averaging_type,
    )
    base_run = build_run(
        pred_scores=base_pred,
        eval_corpus=base_corpus,
        threshold_eval=threshold_eval,
        averaging_type=averaging_type,
    )

    corpora_cmp_full = build_cmp_df(
        current_df=current_run.corpora_df,
        base_df=base_run.corpora_df,
        key_col='Corpus Name',
        info_cols=['Data Count', 'Docs No Labels', 'Decent Labels'],
    )
    classes_cmp_full = build_cmp_df(
        current_df=current_run.classes_df,
        base_df=base_run.classes_df,
        key_col='IPTC Category',
        info_cols=['Data Count'],
    )
    corpora_cmp_df = diff_only_df(df=corpora_cmp_full, key_col='Corpus Name') if only_diff else corpora_cmp_full
    classes_cmp_df = diff_only_df(df=classes_cmp_full, key_col='IPTC Category') if only_diff else classes_cmp_full

    shared_ids = shared_article_ids(current_df=current_run.aligned_df, base_df=base_run.aligned_df)
    cat_ids = gold_map.cat_ids(prob_dfs=[current_run.aligned_df, base_run.aligned_df])

    summary_df = build_summary_df(
        current_run=current_run,
        base_run=base_run,
        classes_cmp=classes_cmp_full,
        corpora_cmp=corpora_cmp_full,
    )
    top_improved_df, top_degraded_df = build_top_change_dfs(classes_df=classes_cmp_full)
    top_improved_stats_df, top_degraded_stats_df = build_top_change_stats_dfs(
        improved_df=top_improved_df,
        degraded_df=top_degraded_df,
    )
    hamming_df = build_hamming_df(
        current_df=subset_by_ids(df=current_run.aligned_df, article_ids=shared_ids),
        base_df=subset_by_ids(df=base_run.aligned_df, article_ids=shared_ids),
        gold_map=gold_map,
        cat_ids=cat_ids,
        threshold_eval=threshold_eval,
    )
    pr_auc_df, pr_auc_summary_df = build_pr_auc_dfs(
        current_df=subset_by_ids(df=current_run.aligned_df, article_ids=shared_ids),
        base_df=subset_by_ids(df=base_run.aligned_df, article_ids=shared_ids),
        gold_map=gold_map,
        cat_ids=cat_ids,
    )

    excel_path = Path(output_path) if output_path is not None else None
    result = ComparisonResult(
        corpora_comparison=corpora_cmp_df,
        classes_comparison=classes_cmp_df,
        summary_comparison=summary_df,
        top_improved_categories=top_improved_df,
        top_degraded_categories=top_degraded_df,
        top_improved_stats=top_improved_stats_df,
        top_degraded_stats=top_degraded_stats_df,
        hamming_loss_comparison=hamming_df,
        pr_auc_per_class=pr_auc_df,
        pr_auc_summary=pr_auc_summary_df,
        current_corpora=current_run.corpora_df,
        current_classes=current_run.classes_df,
        base_corpora=base_run.corpora_df,
        base_classes=base_run.classes_df,
        excel_path=excel_path,
    )
    if excel_path is not None:
        write_csv(result=result, output_path=excel_path.parent)
        write_excel(result=result, output_path=excel_path)
        log_top_changes(result=result, top_n=top_n)
    return result


def build_run(
    *,
    pred_scores: Sequence[Any],
    eval_corpus: Any,
    threshold_eval: float,
    averaging_type: str,
) -> RunEval:
    """Evaluate one run and build its aligned per-article dataframe."""
    corpora_df, classes_df = evaluate_predictions(
        pred_wgh_cats=pred_scores,
        eval_corpus=eval_corpus,
        thr=threshold_eval,
        cat_to_thr=None,
        per_corpus=True,
        per_class=True,
        averaging_type=averaging_type,
    )
    aligned_df = build_aligned_df(eval_corpus=eval_corpus, pred_scores=pred_scores)
    return RunEval(aligned_df=aligned_df, corpora_df=corpora_df, classes_df=classes_df)


def build_aligned_df(*, eval_corpus: Any, pred_scores: Sequence[Any]) -> pd.DataFrame:
    """Build per-article dataframe with gold categories and dense ``prob_*`` columns."""
    if len(pred_scores) != len(eval_corpus):
        raise ValueError(
            f'Misaligned predictions: predictions={len(pred_scores)} corpus={len(eval_corpus)}'
        )
    rows = []
    for doc, doc_pred in zip(eval_corpus, pred_scores):
        rows.append(
            {
                'article_id': str(doc.id),
                'corpus_name': str(doc.metadata.get('corpusName', '')),
                'gold_categories': tuple(sorted(norm_cat_ids(cat_ids=doc.cats))),
                'pred_scores': [(str(cat_id), float(score)) for cat_id, score in doc_pred],
            }
        )
    base_df = pd.DataFrame(rows)
    return add_prob_columns(df=base_df)


def add_prob_columns(*, df: pd.DataFrame) -> pd.DataFrame:
    """Expand ``pred_scores`` into dense ``prob_<cat_id>`` columns, preserving other columns."""
    cat_ids = sorted({cat_id for pred_scores in df['pred_scores'] for cat_id, _ in pred_scores})
    score_dicts = df['pred_scores'].map(dict)
    prob_data = {
        f'prob_{cat_id}': score_dicts.map(lambda d, _cid=cat_id: float(d.get(_cid, 0.0)))
        for cat_id in cat_ids
    }
    prob_df = pd.DataFrame(prob_data, index=df.index)
    return pd.concat([df, prob_df], axis=1)


def gold_df_from_corpus(*, corpus: Any) -> pd.DataFrame:
    """Extract normalized gold dataframe from a ``geneea.catlib.data.Corpus``."""
    rows = []
    for doc in corpus:
        article_id = str(doc.id)
        rows.append(
            {
                'article_id': article_id,
                'corpus_name': str(doc.metadata.get('corpusName', '')),
                'gold_categories': tuple(sorted(norm_cat_ids(cat_ids=doc.cats))),
            }
        )
    df = pd.DataFrame(rows)
    dup_ids = df['article_id'][df['article_id'].duplicated()].astype(str).unique().tolist()
    if dup_ids:
        raise ValueError(f'Duplicate article_id values in eval corpus: {dup_ids[:5]}')
    return df


def norm_cat_ids(*, cat_ids: Sequence[str]) -> list[str]:
    """Apply the same IPTC normalization used by the legacy evaluator."""
    iptc_topics = get_iptc_topics()
    valid_cats = []
    for cat_id in cat_ids:
        if not cat_id:
            continue
        try:
            valid_cats.append(iptc_topics.getCategory(str(cat_id)))
        except KeyError:
            LOG.warning('Skipping unknown IPTC category during normalization: cat_id=%s', cat_id)
    norm_cats = iptc_topics.normalizeCategories(valid_cats)
    return [cat.id for cat in norm_cats if cat.id and cat.id not in REMOVED_CAT_IDS]


def build_cmp_df(
    *,
    current_df: pd.DataFrame,
    base_df: pd.DataFrame,
    key_col: str,
    info_cols: Sequence[str],
) -> pd.DataFrame:
    """Join current/base metric tables and compute current-base deltas."""
    df = current_df.reset_index().merge(
        base_df.reset_index(),
        on=key_col,
        how='outer',
        suffixes=('_current', '_base'),
        sort=False,
    )
    for metric in ['Precision', 'Recall', 'F1']:
        df[f'{metric}_diff'] = df[f'{metric}_current'] - df[f'{metric}_base']

    cols = [key_col]
    for info_col in info_cols:
        cols.extend([f'{info_col}_current', f'{info_col}_base'])
    for metric in ['Precision', 'Recall', 'F1']:
        cols.extend([f'{metric}_current', f'{metric}_base', f'{metric}_diff'])
    return df.reindex(columns=cols)


def diff_only_df(*, df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    """Drop base metric columns while preserving current values and diffs."""
    drop_cols = {f'{metric}_base' for metric in ['Precision', 'Recall', 'F1']}
    cols = [key_col]
    cols.extend(col for col in df.columns if col != key_col and col not in drop_cols)
    return df.loc[:, cols]


def build_summary_df(
    *,
    current_run: RunEval,
    base_run: RunEval,
    classes_cmp: pd.DataFrame,
    corpora_cmp: pd.DataFrame,
) -> pd.DataFrame:
    """Build compact summary rows from aggregate evaluation outputs."""
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
        rows.append(_metric_row(summary_key=summary_key, current=current_row, base=base_row))

    classes_filtered = classes_cmp[~classes_cmp['IPTC Category'].isin(AGG_CLASS_ROWS)].copy()
    classes_filtered['support'] = classes_filtered['Data Count_current'].combine_first(
        classes_filtered['Data Count_base']
    )
    for low, high, label in SUPPORT_BUCKETS:
        mask = (classes_filtered['support'] >= low) & (classes_filtered['support'] < high)
        sub = classes_filtered[mask]
        rows.append(
            _avg_metrics_row(summary_key=f'macro_over_classes_support_{label}', sub_df=sub)
        )

    corpora_filtered = corpora_cmp[~corpora_cmp['Corpus Name'].isin(AGG_CORPUS_ROWS)].copy()
    for prefix in LANG_PREFIXES:
        mask = corpora_filtered['Corpus Name'].str.startswith(f'{prefix}_')
        sub = corpora_filtered[mask]
        rows.append(
            _avg_metrics_row(summary_key=f'macro_over_corpora_prefix_{prefix}', sub_df=sub)
        )

    eurosport_mask = corpora_filtered['Corpus Name'].str.contains(EUROSPORT_TOKEN, case=False, na=False)
    rows.append(
        _avg_metrics_row(
            summary_key=f'macro_over_corpora_{EUROSPORT_TOKEN}',
            sub_df=corpora_filtered[eurosport_mask],
        )
    )

    macro_corpora_current = current_run.corpora_df.loc['All-macro']
    macro_corpora_base = base_run.corpora_df.loc['All-macro']
    rows.append(
        _metric_row(
            summary_key='macro_over_corpora',
            current=macro_corpora_current,
            base=macro_corpora_base,
        )
    )
    return pd.DataFrame(rows)


def _metric_row(*, summary_key: str, current: pd.Series, base: pd.Series) -> dict[str, Any]:
    """Build one summary row from current/base metric series."""
    return {
        'summary_key': summary_key,
        'precision_current': current['Precision'],
        'precision_base': base['Precision'],
        'precision_diff': current['Precision'] - base['Precision'],
        'recall_current': current['Recall'],
        'recall_base': base['Recall'],
        'recall_diff': current['Recall'] - base['Recall'],
        'f1_current': current['F1'],
        'f1_base': base['F1'],
        'f1_diff': current['F1'] - base['F1'],
    }


def _avg_metrics_row(*, summary_key: str, sub_df: pd.DataFrame) -> dict[str, Any]:
    """Macro-average precision/recall/f1 over the rows of ``sub_df``."""
    if sub_df.empty:
        nan = float('nan')
        return {
            'summary_key': summary_key,
            'precision_current': nan,
            'precision_base': nan,
            'precision_diff': nan,
            'recall_current': nan,
            'recall_base': nan,
            'recall_diff': nan,
            'f1_current': nan,
            'f1_base': nan,
            'f1_diff': nan,
        }
    return {
        'summary_key': summary_key,
        'precision_current': float(sub_df['Precision_current'].mean()),
        'precision_base': float(sub_df['Precision_base'].mean()),
        'precision_diff': float(sub_df['Precision_diff'].mean()),
        'recall_current': float(sub_df['Recall_current'].mean()),
        'recall_base': float(sub_df['Recall_base'].mean()),
        'recall_diff': float(sub_df['Recall_diff'].mean()),
        'f1_current': float(sub_df['F1_current'].mean()),
        'f1_base': float(sub_df['F1_base'].mean()),
        'f1_diff': float(sub_df['F1_diff'].mean()),
    }


def build_top_change_dfs(*, classes_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build ranked improved and degraded category tables."""
    df = classes_df[~classes_df['IPTC Category'].isin(AGG_CLASS_ROWS)].copy()
    df['article_frequency'] = df['Data Count_current'].combine_first(df['Data Count_base'])
    df['ranking_score'] = df['F1_diff'].abs()
    cols = [
        'IPTC Category',
        'article_frequency',
        'ranking_score',
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


CHANGE_THRESHOLDS: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7)
TOP_CHANGE_N: int = 100


def build_top_change_stats_dfs(
    *,
    improved_df: pd.DataFrame,
    degraded_df: pd.DataFrame,
    top_n: int = TOP_CHANGE_N,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build summary stats tables for the improved and degraded category rankings."""
    improved_stats = _change_stats_rows(df=improved_df, label='improved', top_n=top_n)
    degraded_stats = _change_stats_rows(df=degraded_df, label='degraded', top_n=top_n)
    return pd.DataFrame(improved_stats), pd.DataFrame(degraded_stats)


def _change_stats_rows(*, df: pd.DataFrame, label: str, top_n: int) -> list[dict[str, Any]]:
    """Build the metric/value rows for one change-direction table."""
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
    """Extract the IPTC top-level category name from a quoted long-name label.

    Example labels::

        '"sport >> chess (20001154)"'         -> 'sport'
        '"arts+ - arts, culture, ... (...)"'  -> 'arts+'

    :param label: Quoted IPTC long-name label.
    :return: Top-level category name (best-effort string parse).
    """
    inner = str(label).strip().strip('"').strip()
    delim_positions = [pos for pos in (inner.find(' >'), inner.find(' -')) if pos != -1]
    if delim_positions:
        return inner[: min(delim_positions)].strip()
    paren_idx = inner.find('(')
    return inner[:paren_idx].strip() if paren_idx != -1 else inner


def shared_article_ids(*, current_df: pd.DataFrame, base_df: pd.DataFrame) -> list[str]:
    """Return article ids shared by current and base runs in current order."""
    base_ids = set(base_df['article_id'])
    article_ids = [article_id for article_id in current_df['article_id'] if article_id in base_ids]
    if not article_ids:
        raise ValueError('Current and base runs do not share any aligned article_id values.')
    return article_ids


def subset_by_ids(*, df: pd.DataFrame, article_ids: Sequence[str]) -> pd.DataFrame:
    """Keep rows for the requested article ids in that exact order."""
    if not article_ids:
        return df.iloc[0:0].copy()
    return df.set_index('article_id', drop=False).loc[list(article_ids)].reset_index(drop=True)


def build_hamming_df(
    *,
    current_df: pd.DataFrame,
    base_df: pd.DataFrame,
    gold_map: GoldLabelMap,
    cat_ids: Sequence[str],
    threshold_eval: float,
) -> pd.DataFrame:
    """Build current/base Hamming loss comparison."""
    article_ids = list(current_df['article_id'])
    gold_matrix = gold_map.gold_matrix(article_ids=article_ids, cat_ids=cat_ids)
    current_loss = compute_hamming(df=current_df, cat_ids=cat_ids, threshold_eval=threshold_eval, gold_matrix=gold_matrix)
    base_loss = compute_hamming(df=base_df, cat_ids=cat_ids, threshold_eval=threshold_eval, gold_matrix=gold_matrix)
    return pd.DataFrame(
        [{'metric': 'hamming_loss', 'current': current_loss, 'base': base_loss, 'diff': current_loss - base_loss}]
    )


def compute_hamming(
    *,
    df: pd.DataFrame,
    cat_ids: Sequence[str],
    threshold_eval: float,
    gold_matrix: np.ndarray,
) -> float:
    """Compute Hamming loss for one aligned probability table."""
    score_matrix = build_score_matrix(df=df, cat_ids=cat_ids)
    pred_matrix = (score_matrix >= threshold_eval).astype(np.int8)
    return float(np.mean(pred_matrix != gold_matrix))


def build_pr_auc_dfs(
    *,
    current_df: pd.DataFrame,
    base_df: pd.DataFrame,
    gold_map: GoldLabelMap,
    cat_ids: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build per-class and summary PR-AUC tables."""
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


def safe_cat_label(*, cat_id: str) -> str:
    """Return the quoted long name used by other tables, falling back to the id."""
    try:
        return '"' + get_cat_name(cat_id) + '"'
    except KeyError:
        LOG.warning('No IPTC name for cat_id=%s; using raw id in PR-AUC table', cat_id)
        return cat_id


def build_pr_auc_summary_df(
    *,
    pr_auc_df: pd.DataFrame,
    current_df: pd.DataFrame,
    gold_matrix: np.ndarray,
    current_scores: np.ndarray,
    base_scores: np.ndarray,
) -> pd.DataFrame:
    """Build PR-AUC summary rows: micro, macro, support buckets, per-corpus groups."""
    rows: list[dict[str, Any]] = []

    rows.append(
        _pr_auc_row(
            aggregation='macro_over_classes_all',
            current=_safe_mean(values=pr_auc_df['pr_auc_current']),
            base=_safe_mean(values=pr_auc_df['pr_auc_base']),
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
                aggregation=f'macro_over_classes_support_{label}',
                current=_safe_mean(values=sub['pr_auc_current']),
                base=_safe_mean(values=sub['pr_auc_base']),
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
            current=_safe_mean(values=pd.Series(list(current_per_corpus.values()), dtype=float)),
            base=_safe_mean(values=pd.Series(list(base_per_corpus.values()), dtype=float)),
        )
    )
    return pd.DataFrame(rows)


def _pr_auc_row(*, aggregation: str, current: float, base: float) -> dict[str, Any]:
    """Build one PR-AUC summary row with diff."""
    diff = current - base if not np.isnan(current) and not np.isnan(base) else np.nan
    return {'aggregation': aggregation, 'current': current, 'base': base, 'diff': diff}


def _safe_mean(*, values: pd.Series) -> float:
    """Mean over non-NaN values; NaN if empty."""
    cleaned = values.dropna()
    return float(cleaned.mean()) if not cleaned.empty else float('nan')


def _avg_filtered(*, values_dict: Mapping[str, float], predicate: Any) -> float:
    """Mean of dict values whose key matches ``predicate`` and is not NaN."""
    matched = [v for k, v in values_dict.items() if predicate(k) and not np.isnan(v)]
    return float(np.mean(matched)) if matched else float('nan')


def _prefix_predicate(prefix: str) -> Any:
    """Predicate matching corpus names starting with ``<prefix>_``."""
    return lambda name: name.startswith(f'{prefix}_')


def _contains_predicate(token: str) -> Any:
    """Predicate matching corpus names containing ``token`` (case-insensitive)."""
    lower = token.lower()
    return lambda name: lower in name.lower()


def micro_pr_auc(*, gold_matrix: np.ndarray, scores: np.ndarray) -> float:
    """Compute micro PR-AUC by flattening across all classes."""
    if gold_matrix.size == 0:
        return float('nan')
    return average_precision(y_true=gold_matrix.flatten().astype(np.int8), y_score=scores.flatten())


def per_corpus_pr_auc(
    *,
    df: pd.DataFrame,
    scores: np.ndarray,
    gold_matrix: np.ndarray,
) -> dict[str, float]:
    """Compute per-corpus macro PR-AUC over classes with positive support in that corpus."""
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


def build_score_matrix(*, df: pd.DataFrame, cat_ids: Sequence[str]) -> np.ndarray:
    """Build dense score matrix for selected category ids."""
    cols = [f'prob_{cat_id}' for cat_id in cat_ids]
    return df.reindex(columns=cols, fill_value=0.0).to_numpy(dtype=float)


def average_precision(*, y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute discrete PR-AUC as average precision."""
    positives = int(y_true.sum())
    if positives == 0:
        return np.nan

    order = np.argsort(-y_score, kind='mergesort')
    sorted_true = y_true[order]
    tp = np.cumsum(sorted_true)
    fp = np.cumsum(1 - sorted_true)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives
    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def write_csv(*, result: ComparisonResult, output_path: Path) -> None:
    """Persist comparison outputs into CSV files."""
    output_path.mkdir(parents=True, exist_ok=True)
    result.corpora_comparison.to_csv(output_path / 'corpora_comparison.csv', index=False)
    result.classes_comparison.to_csv(output_path / 'classes_comparison.csv', index=False)
    result.summary_comparison.to_csv(output_path / 'summary_comparison.csv', index=False)
    result.top_improved_categories.to_csv(output_path / 'top_improved.csv', index=False)
    result.top_degraded_categories.to_csv(output_path / 'top_degraded.csv', index=False)
    result.top_improved_stats.to_csv(output_path / 'top_improved_stats.csv', index=False)
    result.top_degraded_stats.to_csv(output_path / 'top_degraded_stats.csv', index=False)
    result.hamming_loss_comparison.to_csv(output_path / 'hamming_loss.csv', index=False)
    result.pr_auc_per_class.to_csv(output_path / 'pr_auc_per_class.csv', index=False)
    result.pr_auc_summary.to_csv(output_path / 'pr_auc_summary.csv', index=False)
    result.current_corpora.to_csv(output_path / 'current_corpora.csv', index=False)
    result.current_classes.to_csv(output_path / 'current_classes.csv', index=False)
    result.base_corpora.to_csv(output_path / 'base_corpora.csv', index=False)
    result.base_classes.to_csv(output_path / 'base_classes.csv', index=False)


def write_excel(*, result: ComparisonResult, output_path: Path) -> None:
    """Persist comparison outputs into one Excel workbook."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path) as writer:
        result.corpora_comparison.to_excel(writer, sheet_name='corpora_comparison', index=False)
        result.classes_comparison.to_excel(writer, sheet_name='classes_comparison', index=False)
        result.summary_comparison.to_excel(writer, sheet_name='summary_comparison', index=False)
        result.top_improved_categories.to_excel(writer, sheet_name='top_improved', index=False)
        result.top_degraded_categories.to_excel(writer, sheet_name='top_degraded', index=False)
        result.top_improved_stats.to_excel(writer, sheet_name='top_improved_stats', index=False)
        result.top_degraded_stats.to_excel(writer, sheet_name='top_degraded_stats', index=False)
        result.hamming_loss_comparison.to_excel(writer, sheet_name='hamming_loss', index=False)
        result.pr_auc_per_class.to_excel(writer, sheet_name='pr_auc_per_class', index=False)
        result.pr_auc_summary.to_excel(writer, sheet_name='pr_auc_summary', index=False)
        result.current_corpora.to_excel(writer, sheet_name='current_corpora')
        result.current_classes.to_excel(writer, sheet_name='current_classes')
        result.base_corpora.to_excel(writer, sheet_name='base_corpora')
        result.base_classes.to_excel(writer, sheet_name='base_classes')


def log_top_changes(*, result: ComparisonResult, top_n: int) -> None:
    """Log concise previews of top improved and degraded categories."""
    if not result.top_improved_categories.empty:
        LOG.info('Top improved categories:\n%s', result.top_improved_categories.head(top_n).to_string(index=False))
    if not result.top_degraded_categories.empty:
        LOG.info('Top degraded categories:\n%s', result.top_degraded_categories.head(top_n).to_string(index=False))
    if result.excel_path is not None:
        LOG.info('Saved comparison report to %s', result.excel_path)


def build_path(*, output_root: str | Path, config_name: str) -> Path:
    """Create timestamped comparison workbook path."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_name = sanitize_name(value=config_name)
    output_dir = Path(output_root) / f'{safe_name}_{timestamp}'
    return output_dir / f'evaluation_comparison_{safe_name}.xlsx'


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser for standalone run comparison."""
    parser = argparse.ArgumentParser(description='Compare current and base evaluation runs.')
    parser.add_argument(
        '--current-run', '-c', required=True,
        help='Current run directory containing predictions.pkl and eval_corpus.pkl.',
    )
    parser.add_argument(
        '--base-run', '-b', required=True,
        help='Base run directory containing predictions.pkl and eval_corpus.pkl.',
    )
    parser.add_argument('--config-name', default='comparison', help='Output name fragment.')
    parser.add_argument('--threshold-eval', type=float, default=0.5, help='Evaluation threshold.')
    parser.add_argument('--averaging-type', default='datapoint', choices=['datapoint', 'micro', 'macro'])
    parser.add_argument('--top-n', type=int, default=20, help='Preview size for improved/degraded categories.')
    parser.add_argument('--only-diff', action='store_true', help='Drop base metric columns from comparison sheets.')
    parser.add_argument(
        '--output-root',
        default='results/comparisons',
        help='Directory where the Excel report should be written.',
    )
    return parser


def main() -> None:
    """Run the comparison CLI."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    args = build_arg_parser().parse_args()
    output_path = build_path(output_root=args.output_root, config_name=args.config_name)
    compare_runs(
        current_run_dir=args.current_run,
        base_run_dir=args.base_run,
        threshold_eval=args.threshold_eval,
        averaging_type=args.averaging_type,
        top_n=args.top_n,
        only_diff=args.only_diff,
        output_path=output_path,
    )


if __name__ == '__main__':
    main()
