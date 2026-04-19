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
from iptc_entity_pipeline.evaluate import REMOVED_CAT_IDS, evaluate_predictions, get_iptc_topics
from iptc_entity_pipeline.model_io import EVAL_CORPUS_FILENAME, PREDICTIONS_FILENAME

LOG = logging.getLogger(__name__)

AGG_CLASS_ROWS = frozenset({'All - micro avg', 'All - macro avg', 'All - datapoint avg'})
SUMMARY_ROWS = {
    'macro_over_corpora': ('corpora', 'All-macro'),
    'macro_over_classes_all': ('classes', 'All - macro avg'),
    'micro_over_labels': ('classes', 'All - micro avg'),
}
PR_AUC_AGG = 'macro'


@dataclass(frozen=True)
class ComparisonResult:
    """Structured outputs from one evaluation comparison run."""

    corpora_comparison: pd.DataFrame
    classes_comparison: pd.DataFrame
    summary_comparison: pd.DataFrame
    top_improved_categories: pd.DataFrame
    top_degraded_categories: pd.DataFrame
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

    summary_df = build_summary_df(current_run=current_run, base_run=base_run)
    top_improved_df, top_degraded_df = build_top_change_dfs(classes_df=classes_cmp_full)
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


def build_summary_df(*, current_run: RunEval, base_run: RunEval) -> pd.DataFrame:
    """Build compact summary rows from aggregate evaluation outputs."""
    rows = []
    tables = {
        'corpora_current': current_run.corpora_df,
        'corpora_base': base_run.corpora_df,
        'classes_current': current_run.classes_df,
        'classes_base': base_run.classes_df,
    }
    for summary_key, (group_name, row_name) in SUMMARY_ROWS.items():
        current_row = tables[f'{group_name}_current'].loc[row_name]
        base_row = tables[f'{group_name}_base'].loc[row_name]
        rows.append(
            {
                'summary_key': summary_key,
                'precision_current': current_row['Precision'],
                'precision_base': base_row['Precision'],
                'precision_diff': current_row['Precision'] - base_row['Precision'],
                'recall_current': current_row['Recall'],
                'recall_base': base_row['Recall'],
                'recall_diff': current_row['Recall'] - base_row['Recall'],
                'f1_current': current_row['F1'],
                'f1_base': base_row['F1'],
                'f1_diff': current_row['F1'] - base_row['F1'],
            }
        )
    return pd.DataFrame(rows)


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
                'IPTC Category': cat_id,
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
    current_macro = float(pr_auc_df['pr_auc_current'].dropna().mean()) if not pr_auc_df.empty else np.nan
    base_macro = float(pr_auc_df['pr_auc_base'].dropna().mean()) if not pr_auc_df.empty else np.nan
    pr_auc_summary_df = pd.DataFrame(
        [
            {
                'aggregation': PR_AUC_AGG,
                'current': current_macro,
                'base': base_macro,
                'diff': current_macro - base_macro
                if not np.isnan(current_macro) and not np.isnan(base_macro)
                else np.nan,
            }
        ]
    )
    return pr_auc_df, pr_auc_summary_df


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


def build_output_path(*, output_root: str | Path, config_name: str) -> Path:
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
    output_path = build_output_path(output_root=args.output_root, config_name=args.config_name)
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
