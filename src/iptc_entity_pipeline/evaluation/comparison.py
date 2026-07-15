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
import json
import logging
import pickle
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from sklearn.metrics import average_precision_score
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
import numpy as np
import pandas as pd

from iptc_entity_pipeline.category_sets import load_relevant_cat_ids, load_tail_cat_ids
from iptc_entity_pipeline.config import EvaluationCnf
from iptc_entity_pipeline.data_loading import sanitize_name
from iptc_entity_pipeline.evaluation.evaluate import (
    CLASS_RELEVANT_MACRO_ROW,
    REMOVED_CAT_IDS,
    evaluate_predictions,
    get_cat_name,
    get_iptc_topics,
    normalize_pred_cats,
)
from iptc_entity_pipeline.model_io import EVAL_CORPUS_FILENAME, PREDICTIONS_FILENAME

LOG = logging.getLogger(__name__)

# Supported on-disk filenames carrying per-class thresholds for a saved run.
# ``custom_thresholds.json`` matches the legacy IPTC pipeline convention; the
# alternative ``thresholds.json`` is kept for backward compatibility with runs
# saved by earlier versions of this pipeline before the rename.
THRESHOLD_FILENAMES: tuple[str, ...] = ('custom_thresholds.json', 'thresholds.json')

# Output-facing aggregate row labels aligned with the thesis vocabulary. The
# thesis "Macro Head" set (categories with >= 15 test samples) is exactly the
# ``relevant_cats.yaml`` set produced by ``evaluate.py`` as
# ``CLASS_RELEVANT_MACRO_ROW``; we relabel it to ``All_macro_head`` in the
# comparison outputs. ``MATCH_HEAD_ROW`` is the label emitted by ``evaluate.py``
# and is used only to locate that row before relabeling it.
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
EUROSPORT_TOKEN = 'eurosport'
MCNEMAR_ALPHA = 0.05
MCNEMAR_MIN_DISAGREEMENTS = 25

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
# run (thesis: monitor per-class FP volume); the base/diff FP columns are
# dropped to keep ``classes_comparison`` compact.
_CLASSES_COMPARISON_DROP_COLS: tuple[str, ...] = (
    'False Positive Count_base',
    'False Positive Count_diff',
)

_RESULT_SHEETS: tuple[tuple[str, str, bool], ...] = (
    ('corpora_comparison', 'corpora_comparison', False),
    ('corpora_comparison_macro_head', 'corpora_comparison_macro_head', False),
    ('language_comparison', 'language_comparison', False),
    ('language_comparison_macro_head', 'language_comparison_macro_head', False),
    ('classes_comparison', 'classes_comparison', False),
    ('class_confusion_counts', 'class_confusion_counts', False),
    ('class_thresholds', 'class_thresholds', False),
    ('summary_comparison', 'summary_comparison', False),
    ('top_improved_categories', 'top_improved', False),
    ('top_degraded_categories', 'top_degraded', False),
    ('top_improved_stats', 'top_improved_stats', False),
    ('top_degraded_stats', 'top_degraded_stats', False),
    ('pr_auc_per_class', 'pr_auc_per_class', False),
    ('pr_auc_summary', 'pr_auc_summary', False),
    ('entity_impact_all', 'entity_impact_all', False),
    ('article_confusion_diff', 'article_confusion_diff', False),
    ('current_corpora', 'current_corpora', True),
    ('current_classes', 'current_classes', True),
    ('base_corpora', 'base_corpora', True),
    ('base_classes', 'base_classes', True),
)
_TOP_CHANGE_CATEGORY_SHEETS: tuple[tuple[str, str, bool], ...] = (
    ('top_improved_categories', 'top_improved', False),
    ('top_degraded_categories', 'top_degraded', False),
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ComparisonResult:
    """Structured outputs from one evaluation comparison run."""

    corpora_comparison: pd.DataFrame
    corpora_comparison_macro_head: pd.DataFrame
    language_comparison: pd.DataFrame
    language_comparison_macro_head: pd.DataFrame
    classes_comparison: pd.DataFrame
    class_confusion_counts: pd.DataFrame
    class_thresholds: pd.DataFrame
    summary_comparison: pd.DataFrame
    top_improved_categories: pd.DataFrame
    top_degraded_categories: pd.DataFrame
    top_improved_stats: pd.DataFrame
    top_degraded_stats: pd.DataFrame
    pr_auc_per_class: pd.DataFrame
    pr_auc_summary: pd.DataFrame
    entity_impact_all: pd.DataFrame
    article_confusion_diff: pd.DataFrame
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
class  GoldLabelMap:
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


@dataclass(frozen=True)
class ArticleEntity:
    """One normalized entity mention attached to an article."""

    gkb_id: str | None
    wdids: tuple[str, ...]
    entity_type: str | None
    std_form: str | None
    relevance: float | None
    mention_count: int | None


@dataclass(frozen=True)
class ArticleEvalRecord:
    """Normalized article payload used before DataFrame conversion."""

    article_id: str
    corpus_name: str
    gold_categories: tuple[str, ...]
    pred_scores: tuple[tuple[str, float], ...]
    article_text: str | None
    article_length: int | None
    entities: tuple[ArticleEntity, ...]


# ---------------------------------------------------------------------------
# Run loading and comparison orchestration
# ---------------------------------------------------------------------------

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


def load_custom_thresholds(*, run_dir: str | Path) -> dict[str, float]:
    """Load per-class thresholds from a saved-run directory.

    Tries each filename in :data:`THRESHOLD_FILENAMES` in order and returns
    the first successfully parsed JSON object as a ``{cat_id: threshold}``
    mapping. Returns an empty dict if no file is present or the file is not
    a JSON object.

    :param run_dir: Saved-run directory possibly containing a thresholds JSON.
    :return: Mapping ``category_id -> threshold`` (empty when not available).
    """
    run_path = Path(run_dir)
    for filename in THRESHOLD_FILENAMES:
        candidate = run_path / filename
        if not candidate.is_file():
            continue
        with open(candidate, encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, Mapping):
            LOG.warning(f'Custom thresholds file is not a JSON object: {candidate}')
            continue
        thresholds = {str(cid): float(thr) for cid, thr in data.items()}
        LOG.info(f'Loaded {len(thresholds)} per-class thresholds from {candidate}')
        return thresholds
    return {}


def thresholds_vector(
    *,
    cat_ids: Sequence[str],
    cat_to_thr: Mapping[str, float] | None,
    default_threshold: float,
) -> np.ndarray:
    """Build a per-class threshold vector aligned with ``cat_ids``.

    Classes missing from ``cat_to_thr`` (or all classes when ``cat_to_thr`` is
    ``None``/empty) fall back to ``default_threshold``.

    :param cat_ids: Category id sequence defining the column order.
    :param cat_to_thr: Optional ``cat_id -> threshold`` map.
    :param default_threshold: Fallback threshold for unmapped classes.
    :return: 1-D ``float64`` array of shape ``(len(cat_ids),)``.
    """
    if not cat_to_thr:
        return np.full(len(cat_ids), float(default_threshold), dtype=float)
    return np.asarray(
        [float(cat_to_thr.get(cid, default_threshold)) for cid in cat_ids],
        dtype=float,
    )


def compare_runs(
    *,
    current_run_dir: str | Path,
    base_run_dir: str | Path,
    threshold_eval: float,
    averaging_type: str = 'micro',
    top_n: int = 20,
    only_diff: bool = False,
    top_changes_only: bool = False,
    output_path: str | Path | None = None,
    use_saved_thresholds: bool = True,
) -> ComparisonResult:
    """Compare current and base runs loaded from their saved directories.

    Gold labels are derived from the current run's pickled corpus and reused
    for both runs' cross-run metrics (Hamming loss, PR-AUC).

    Per-class thresholds (saved as ``custom_thresholds.json`` or, for older
    runs, ``thresholds.json``) are auto-loaded per run when
    ``use_saved_thresholds`` is ``True``. Each run uses its own thresholds for
    F1/precision/recall tables, McNemar, Hamming, and per-article F1 deltas.
    Runs without a thresholds file fall back to ``threshold_eval`` for every
    class. Score-based metrics (PR-AUC) ignore thresholds in the statistic
    itself. Brier likewise uses calibrated probabilities versus gold labels, but
    Brier table rows attach effective thresholds for diagnostics.
    :param current_run_dir: Directory of the current run.
    :param base_run_dir: Directory of the base run.
    :param threshold_eval: Default threshold for classes missing from a run's
        per-class map and for runs that have no thresholds file at all.
    :param averaging_type: One of ``'datapoint'``, ``'micro'``, ``'macro'``.
        Defaults to ``'micro'`` so per-corpus rows report micro-averaged metrics,
        matching the thesis per-corpus reporting.
    :param top_n: Preview size for improved/degraded categories logging.
    :param only_diff: Drop base metric columns from comparison sheets.
    :param top_changes_only: Persist only top improved/degraded category tables.
    :param output_path: Optional Excel workbook path.
    :param use_saved_thresholds: When ``True`` (default), auto-load per-class
        thresholds from each run directory. Set to ``False`` to force a
        uniform comparison under ``threshold_eval``.
    :return: Structured :class:`ComparisonResult`.
    """
    relevant_cat_ids = load_relevant_cat_ids()
    tail_cat_ids = load_tail_cat_ids()
    current_pred, current_corpus = load_run(run_dir=current_run_dir)
    base_pred, base_corpus = load_run(run_dir=base_run_dir)
    validate_subset_ids_in_corpora(
        current_corpus=current_corpus,
        base_corpus=base_corpus,
        subset_ids=tail_cat_ids,
        subset_name='tail',
    )
    validate_subset_ids_in_corpora(
        current_corpus=current_corpus,
        base_corpus=base_corpus,
        subset_ids=relevant_cat_ids,
        subset_name='head',
        require_all=False,
    )

    current_thresholds = load_custom_thresholds(run_dir=current_run_dir) if use_saved_thresholds else {}
    base_thresholds = load_custom_thresholds(run_dir=base_run_dir) if use_saved_thresholds else {}

    evaluation_config = EvaluationCnf(
        threshold_eval=threshold_eval,
        averaging_type=averaging_type,
    )

    current_run = build_run(
        pred_scores=current_pred,
        eval_corpus=current_corpus,
        evaluation_config=evaluation_config,
        cat_to_thr=current_thresholds or None,
    )
    base_run = build_run(
        pred_scores=base_pred,
        eval_corpus=base_corpus,
        evaluation_config=evaluation_config,
        cat_to_thr=base_thresholds or None,
    )
    current_classes_df = replace_macro_row_in_run_classes_df(
        classes_df=current_run.classes_df,
        class_ids=relevant_cat_ids,
        row_label=CLASS_MACRO_ROW,
        subset_name='head',
        require_all=False,
        match_label=MATCH_HEAD_ROW,
    )
    base_classes_df = replace_macro_row_in_run_classes_df(
        classes_df=base_run.classes_df,
        class_ids=relevant_cat_ids,
        row_label=CLASS_MACRO_ROW,
        subset_name='head',
        require_all=False,
        match_label=MATCH_HEAD_ROW,
    )
    current_classes_df = replace_macro_row_in_run_classes_df(
        classes_df=current_classes_df,
        class_ids=tail_cat_ids,
        row_label=CLASS_TAIL_ROW,
        subset_name='tail',
        require_all=True,
        insert_after_label=CLASS_MACRO_ROW,
    )
    base_classes_df = replace_macro_row_in_run_classes_df(
        classes_df=base_classes_df,
        class_ids=tail_cat_ids,
        row_label=CLASS_TAIL_ROW,
        subset_name='tail',
        require_all=True,
        insert_after_label=CLASS_MACRO_ROW,
    )
    current_run_summary = RunEval(
        aligned_df=current_run.aligned_df,
        corpora_df=current_run.corpora_df,
        classes_df=current_classes_df,
    )
    base_run_summary = RunEval(
        aligned_df=base_run.aligned_df,
        corpora_df=base_run.corpora_df,
        classes_df=base_classes_df,
    )

    classes_cmp_full = build_cmp_df(
        current_df=current_classes_df,
        base_df=base_classes_df,
        key_col='IPTC Category',
        info_cols=['Data Count'],
        cmp_metrics=_CMP_METRICS_CLASSES,
    )
    top_improved_df, top_degraded_df = build_top_change_dfs(classes_df=classes_cmp_full)
    gold_map = GoldLabelMap.from_corpus(corpus=current_corpus)
    shared_ids = shared_article_ids(current_df=current_run.aligned_df, base_df=base_run.aligned_df)
    cat_ids = gold_map.cat_ids(prob_dfs=[current_run.aligned_df, base_run.aligned_df])
    current_aligned_df = subset_by_ids(df=current_run.aligned_df, article_ids=shared_ids)
    base_aligned_df = subset_by_ids(df=base_run.aligned_df, article_ids=shared_ids)
    current_thr_vec = thresholds_vector(
        cat_ids=cat_ids, cat_to_thr=current_thresholds, default_threshold=threshold_eval,
    )
    base_thr_vec = thresholds_vector(
        cat_ids=cat_ids, cat_to_thr=base_thresholds, default_threshold=threshold_eval,
    )
    class_confusion_df = build_class_confusion_counts_df(
        current_df=current_aligned_df,
        base_df=base_aligned_df,
        gold_map=gold_map,
        cat_ids=cat_ids,
        current_thr_vec=current_thr_vec,
        base_thr_vec=base_thr_vec,
    )
    support_gold_matrix = gold_map.gold_matrix(article_ids=shared_ids, cat_ids=cat_ids)
    class_supports = {
        cat_id: int(support_gold_matrix[:, idx].sum(dtype=np.int64))
        for idx, cat_id in enumerate(cat_ids)
    }
    class_thresholds_df = build_class_thresholds_df(
        cat_ids=cat_ids,
        default_threshold=threshold_eval,
        current_thresholds=current_thresholds,
        base_thresholds=base_thresholds,
        class_supports=class_supports,
    )
    mcnemar_df = build_mcnemar_significance_df(
        current_df=current_aligned_df,
        base_df=base_aligned_df,
        gold_map=gold_map,
        cat_ids=cat_ids,
        current_thr_vec=current_thr_vec,
        base_thr_vec=base_thr_vec,
    )
    top_improved_df, top_degraded_df = add_mcnemar_to_top_change_dfs(
        improved_df=top_improved_df,
        degraded_df=top_degraded_df,
        mcnemar_df=mcnemar_df,
    )
    label_to_cat_id = build_label_to_cat_id_map(cat_ids=cat_ids)
    if top_changes_only:
        empty_df = pd.DataFrame()
        excel_path = Path(output_path) if output_path is not None else None
        result = ComparisonResult(
            corpora_comparison=empty_df,
            corpora_comparison_macro_head=empty_df,
            language_comparison=empty_df,
            language_comparison_macro_head=empty_df,
            classes_comparison=empty_df,
            class_confusion_counts=empty_df,
            class_thresholds=empty_df,
            summary_comparison=empty_df,
            top_improved_categories=with_class_id_column(
                df=top_improved_df,
                key_col='IPTC Category',
                label_to_cat_id=label_to_cat_id,
            ),
            top_degraded_categories=with_class_id_column(
                df=top_degraded_df,
                key_col='IPTC Category',
                label_to_cat_id=label_to_cat_id,
            ),
            top_improved_stats=empty_df,
            top_degraded_stats=empty_df,
            pr_auc_per_class=empty_df,
            pr_auc_summary=empty_df,
            entity_impact_all=empty_df,
            article_confusion_diff=empty_df,
            current_corpora=current_run.corpora_df,
            current_classes=with_class_id_column(
                df=current_classes_df,
                key_col='IPTC Category',
                label_to_cat_id=label_to_cat_id,
            ),
            base_corpora=base_run.corpora_df,
            base_classes=with_class_id_column(
                df=base_classes_df,
                key_col='IPTC Category',
                label_to_cat_id=label_to_cat_id,
            ),
            excel_path=excel_path,
        )
        if excel_path is not None:
            write_csv(result=result, output_path=excel_path.parent, result_sheets=_TOP_CHANGE_CATEGORY_SHEETS)
            write_excel(result=result, output_path=excel_path, result_sheets=_TOP_CHANGE_CATEGORY_SHEETS)
            log_top_changes(result=result, top_n=top_n)
        return result

    corpora_cmp_full = build_cmp_df(
        current_df=current_run.corpora_df,
        base_df=base_run.corpora_df,
        key_col='Corpus Name',
        info_cols=['Data Count', 'Docs No Labels', 'Decent Labels'],
        cmp_metrics=_CMP_METRICS_CORPORA,
    )
    corpora_cmp_df = (
        diff_only_df(df=corpora_cmp_full, key_col='Corpus Name', cmp_metrics=_CMP_METRICS_CORPORA)
        if only_diff
        else corpora_cmp_full
    )
    corpora_cmp_macro_head_full = build_corpora_macro_head_cmp_df(
        current_df=current_aligned_df,
        base_df=base_aligned_df,
        gold_map=gold_map,
        cat_ids=cat_ids,
        current_thr_vec=current_thr_vec,
        base_thr_vec=base_thr_vec,
        corpora_cmp_reference=corpora_cmp_full,
    )
    corpora_cmp_macro_head_df = (
        diff_only_df(
            df=corpora_cmp_macro_head_full,
            key_col='Corpus Name',
            cmp_metrics=_CMP_METRICS_CORPORA,
        )
        if only_diff
        else corpora_cmp_macro_head_full
    )
    language_cmp_full = build_language_cmp_df(corpora_cmp=corpora_cmp_full)
    language_cmp_df = (
        diff_only_df(df=language_cmp_full, key_col='Language', cmp_metrics=_CMP_METRICS_CORPORA)
        if only_diff
        else language_cmp_full
    )
    language_cmp_macro_head_full = build_language_cmp_df(corpora_cmp=corpora_cmp_macro_head_full)
    language_cmp_macro_head_df = (
        diff_only_df(df=language_cmp_macro_head_full, key_col='Language', cmp_metrics=_CMP_METRICS_CORPORA)
        if only_diff
        else language_cmp_macro_head_full
    )
    summary_language_df = language_cmp_df.copy()
    classes_cmp_clean = classes_cmp_full.drop(columns=list(_CLASSES_COMPARISON_DROP_COLS), errors='ignore')
    classes_cmp_df = (
        diff_only_df(df=classes_cmp_clean, key_col='IPTC Category', cmp_metrics=_CMP_METRICS_REPORT)
        if only_diff
        else classes_cmp_clean
    )
    summary_df = build_summary_df(
        current_run=current_run_summary,
        base_run=base_run_summary,
        classes_cmp=classes_cmp_full,
        relevant_cat_ids=relevant_cat_ids,
        tail_cat_ids=tail_cat_ids,
    )
    top_improved_stats_df, top_degraded_stats_df = build_top_change_stats_dfs(
        improved_df=top_improved_df,
        degraded_df=top_degraded_df,
    )
    pr_auc_df, pr_auc_summary_df = build_pr_auc_dfs(
        current_df=current_aligned_df,
        base_df=base_aligned_df,
        gold_map=gold_map,
        cat_ids=cat_ids,
    )
    article_f1_df = build_article_f1_diff_df(
        current_df=current_aligned_df,
        base_df=base_aligned_df,
        gold_map=gold_map,
        cat_ids=cat_ids,
        current_thr_vec=current_thr_vec,
        base_thr_vec=base_thr_vec,
    )
    entity_all_df = build_entity_impact_all_df(
        current_df=current_aligned_df,
        article_f1_df=article_f1_df,
    )
    article_confusion_diff_df = build_article_confusion_diff_df(
        current_df=current_aligned_df,
        base_df=base_aligned_df,
        gold_map=gold_map,
        cat_ids=cat_ids,
        current_thr_vec=current_thr_vec,
        base_thr_vec=base_thr_vec,
    )

    excel_path = Path(output_path) if output_path is not None else None
    result = ComparisonResult(
        corpora_comparison=corpora_cmp_df,
        corpora_comparison_macro_head=corpora_cmp_macro_head_df,
        language_comparison=language_cmp_df,
        language_comparison_macro_head=language_cmp_macro_head_df,
        classes_comparison=with_class_id_column(
            df=classes_cmp_df,
            key_col='IPTC Category',
            label_to_cat_id=label_to_cat_id,
        ),
        class_confusion_counts=class_confusion_df,
        class_thresholds=class_thresholds_df,
        summary_comparison=summary_df,
        top_improved_categories=with_class_id_column(
            df=top_improved_df,
            key_col='IPTC Category',
            label_to_cat_id=label_to_cat_id,
        ),
        top_degraded_categories=with_class_id_column(
            df=top_degraded_df,
            key_col='IPTC Category',
            label_to_cat_id=label_to_cat_id,
        ),
        top_improved_stats=top_improved_stats_df,
        top_degraded_stats=top_degraded_stats_df,
        pr_auc_per_class=pr_auc_df,
        pr_auc_summary=pr_auc_summary_df,
        entity_impact_all=entity_all_df,
        article_confusion_diff=article_confusion_diff_df,
        current_corpora=current_run.corpora_df,
        current_classes=with_class_id_column(
            df=current_classes_df,
            key_col='IPTC Category',
            label_to_cat_id=label_to_cat_id,
        ),
        base_corpora=base_run.corpora_df,
        base_classes=with_class_id_column(
            df=base_classes_df,
            key_col='IPTC Category',
            label_to_cat_id=label_to_cat_id,
        ),
        excel_path=excel_path,
    )
    if excel_path is not None:
        result_sheets = _TOP_CHANGE_CATEGORY_SHEETS if top_changes_only else _RESULT_SHEETS
        write_csv(result=result, output_path=excel_path.parent, result_sheets=result_sheets)
        write_excel(result=result, output_path=excel_path, result_sheets=result_sheets)
        log_top_changes(result=result, top_n=top_n)
    return result


# ---------------------------------------------------------------------------
# Run evaluation helpers
# ---------------------------------------------------------------------------

def build_run(
    *,
    pred_scores: Sequence[Any],
    eval_corpus: Any,
    evaluation_config: EvaluationCnf,
    cat_to_thr: Mapping[str, float] | None = None,
) -> RunEval:
    """Evaluate one run and build its aligned per-article dataframe.

    :param cat_to_thr: Optional per-class thresholds applied during the
        F1/precision/recall table computation.
    """
    corpora_df, classes_df = evaluate_predictions(
        pred_wgh_cats=pred_scores,
        eval_corpus=eval_corpus,
        evaluation_config=evaluation_config,
        cat_to_thr=cat_to_thr,
    )
    aligned_df = build_aligned_df(eval_corpus=eval_corpus, pred_scores=pred_scores)
    return RunEval(aligned_df=aligned_df, corpora_df=corpora_df, classes_df=classes_df)


def build_aligned_df(*, eval_corpus: Any, pred_scores: Sequence[Any]) -> pd.DataFrame:
    """Build per-article dataframe with gold categories and dense ``prob_*`` columns."""
    records = build_article_records(eval_corpus=eval_corpus, pred_scores=pred_scores)
    return records_to_df(records=records)


def build_article_records(*, eval_corpus: Any, pred_scores: Sequence[Any]) -> list[ArticleEvalRecord]:
    """Build normalized article records from corpus docs and prediction tuples."""
    if len(pred_scores) != len(eval_corpus):
        raise ValueError(
            f'Misaligned predictions: predictions={len(pred_scores)} corpus={len(eval_corpus)}'
        )
    records: list[ArticleEvalRecord] = []
    for doc, doc_pred in zip(eval_corpus, pred_scores):
        metadata = _doc_metadata(doc=doc)
        records.append(
            ArticleEvalRecord(
                article_id=str(doc.id),
                corpus_name=str(metadata.get('corpusName', '')),
                gold_categories=tuple(sorted(norm_cat_ids(cat_ids=doc.cats))),
                pred_scores=tuple((str(cat_id), float(score)) for cat_id, score in doc_pred),
                article_text=_extract_article_text(doc=doc),
                article_length=_extract_article_length(doc=doc, metadata=metadata),
                entities=_extract_entities(doc=doc, metadata=metadata),
            )
        )
    return records


def records_to_df(*, records: Sequence[ArticleEvalRecord]) -> pd.DataFrame:
    """Convert article records to the aligned DataFrame used by metric code."""
    rows = [
        {
            'article_id': record.article_id,
            'corpus_name': record.corpus_name,
            'gold_categories': record.gold_categories,
            'pred_scores': list(record.pred_scores),
            'article_text': record.article_text,
            'article_length': record.article_length,
            'entities': list(record.entities),
        }
        for record in records
    ]
    base_df = pd.DataFrame(rows)
    return add_prob_columns(df=base_df)


def _doc_metadata(*, doc: Any) -> Mapping[str, Any]:
    """Return document metadata mapping or an empty mapping."""
    metadata = getattr(doc, 'metadata', None)
    return metadata if isinstance(metadata, Mapping) else {}


def _extract_article_text(*, doc: Any) -> str | None:
    """Extract article text from document text payload."""
    text = getattr(doc, 'text', None)
    if isinstance(text, str) and text:
        return text
    return None


def _extract_article_length(*, doc: Any, metadata: Mapping[str, Any]) -> int | None:
    """Extract known typed article length if available."""
    for key in ('article_length', 'articleLength', 'length'):
        value = _mapping_value(item=metadata, keys=(key,))
        converted = _as_int(value=value)
        if converted is not None:
            return converted

    text = getattr(doc, 'text', None)
    if isinstance(text, str) and text:
        return len(text)
    return None


def _extract_entities(*, doc: Any, metadata: Mapping[str, Any]) -> tuple[ArticleEntity, ...]:
    """Extract normalized entities from doc attributes or metadata."""
    raw_entities = getattr(doc, 'entities', None)
    if raw_entities is None:
        raw_entities = _mapping_value(item=metadata, keys=('entities', 'article_entities', 'entity_list'))
    if not isinstance(raw_entities, Sequence) or isinstance(raw_entities, (str, bytes)):
        return ()
    entities = [_parse_entity(item=item) for item in raw_entities]
    return tuple(entity for entity in entities if entity is not None)


def _parse_entity(*, item: Any) -> ArticleEntity | None:
    """Normalize entity from known source shapes (raw CSV dict or LinkedEntity)."""
    raw_payload: Mapping[str, Any] | None = None

    if isinstance(item, Mapping):
        # Raw entities from CSV use keys like: gkbId, stdForm, type, mentions, feats.relevance.
        raw_payload = item
        gkb_raw = raw_payload.get('gkbId')
        wdids_raw = raw_payload.get('wdid') or raw_payload.get('wdids')
        entity_type_raw = raw_payload.get('type')
        std_form_raw = raw_payload.get('stdForm')
        relevance_raw = raw_payload.get('relevance')
        mentions_raw = raw_payload.get('mentions')
    else:
        # Linked entities attached in data_loading.py expose normalized attrs + raw_entity payload.
        gkb_raw = getattr(item, 'gkb_id', None)
        wdids_raw = getattr(item, 'wd_ids', None)
        relevance_raw = getattr(item, 'relevance', None)
        mentions_raw = getattr(item, 'mention_count', None)
        entity_type_raw = getattr(item, 'entity_type', None)
        raw_maybe = getattr(item, 'raw_entity', None)
        if isinstance(raw_maybe, Mapping):
            raw_payload = raw_maybe
        if entity_type_raw is None and raw_payload is not None:
            entity_type_raw = raw_payload.get('type')
        std_form_raw = raw_payload.get('stdForm') if raw_payload is not None else None

    if relevance_raw is None and raw_payload is not None:
        feats_raw = raw_payload.get('feats')
        if isinstance(feats_raw, Mapping):
            relevance_raw = feats_raw.get('relevance')
    if mentions_raw is None and raw_payload is not None:
        mentions_raw = raw_payload.get('mentions')
    if raw_payload is not None:
        raw_type = _mapping_value(item=raw_payload, keys=('type', 'entityType'))
        if _normalize_entity_type(value=entity_type_raw) in {None, 'other'}:
            entity_type_raw = raw_type

    gkb_id = _as_str(value=gkb_raw)
    wdids = _as_wdid_tuple(value=wdids_raw)
    entity_type = _normalize_entity_type(value=entity_type_raw)
    std_form = _as_str(value=std_form_raw)
    relevance = _as_float(value=relevance_raw)
    mention_count = _as_mentions_count(value=mentions_raw)

    if (
        gkb_id is None
        and not wdids
        and entity_type is None
        and std_form is None
        and relevance is None
        and mention_count is None
    ):
        return None
    return ArticleEntity(
        gkb_id=gkb_id,
        wdids=wdids,
        entity_type=entity_type,
        std_form=std_form,
        relevance=relevance,
        mention_count=mention_count,
    )


def _mapping_value(*, item: Mapping[str, Any], keys: Sequence[str]) -> Any:
    """Return first present mapping key value."""
    for key in keys:
        if key in item:
            return item[key]
    return None


def _normalize_entity_type(*, value: Any) -> str | None:
    """Normalize entity type to canonical lower-case labels."""
    if value is None:
        return None
    if isinstance(value, Enum):
        enum_value = getattr(value, 'value', None)
        if enum_value is not None:
            return _normalize_entity_type(value=enum_value)
        return _normalize_entity_type(value=value.name)
    value_str = _as_str(value=value)
    if value_str is None:
        return None
    normalized = value_str.strip()
    if normalized.startswith('EntityType.'):
        normalized = normalized.split('.', 1)[1]
    normalized = normalized.lower()
    return normalized or None


def _as_wdid_tuple(*, value: Any) -> tuple[str, ...]:
    """Normalize one-or-many Wikidata ids to an immutable tuple."""
    if value is None:
        return ()

    wdids: list[str] = []
    if isinstance(value, str):
        normalized = value.replace('|', ',')
        chunks = [chunk.strip() for chunk in normalized.split(',')]
        wdids.extend(chunk for chunk in chunks if chunk)
    elif isinstance(value, Sequence):
        for item in value:
            normalized = _as_str(value=item)
            if normalized:
                wdids.append(normalized)
    else:
        normalized = _as_str(value=value)
        if normalized:
            wdids.append(normalized)

    seen: set[str] = set()
    unique_wdids = []
    for item in wdids:
        if item in seen:
            continue
        seen.add(item)
        unique_wdids.append(item)
    return tuple(unique_wdids)


def _as_float(*, value: Any) -> float | None:
    """Best-effort cast to float."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(*, value: Any) -> int | None:
    """Best-effort cast to int."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_mentions_count(*, value: Any) -> int | None:
    """Normalize mention count from integer-like value or mentions list."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value)
    return _as_int(value=value)


def _as_str(*, value: Any) -> str | None:
    """Best-effort cast to non-empty string."""
    if value is None:
        return None
    value_str = str(value).strip()
    return value_str if value_str else None


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
    """Join current/base metric tables and compute current-base deltas."""
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


def build_language_cmp_df(
    *,
    corpora_cmp: pd.DataFrame,
    key_col: str = 'Language',
) -> pd.DataFrame:
    """Aggregate per-corpus comparison rows into per-language rows (macro over corpora)."""
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
            row[f'{metric_col}_current'] = _safe_mean_series(group[f'{metric_col}_current'])
            row[f'{metric_col}_base'] = _safe_mean_series(group[f'{metric_col}_base'])
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
    """Build corpus-level comparison table with macro-head metrics (support >= ``min_support``)."""
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


def _macro_head_corpora_row(
    *,
    corpus_name: str,
    current_stats: Mapping[str, float],
    base_stats: Mapping[str, float],
    info_row: pd.Series | None,
) -> dict[str, Any]:
    """Build one corpus row with macro-head metrics and info columns."""
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
    """Read integer column from optional row, else return default."""
    if row is None:
        return default
    value = row.get(col, np.nan)
    return _safe_int(value, default=default)


def _safe_int(value: Any, default: int = 0) -> int:
    """Convert numeric value to ``int`` with fallback for missing values."""
    if value is None:
        return default
    try:
        if np.isnan(value):
            return default
    except TypeError:
        pass
    return int(value)


def _safe_mean_series(values: pd.Series) -> float:
    """Mean over finite values; returns ``NaN`` when no finite values exist."""
    numeric = pd.to_numeric(values, errors='coerce')
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return float('nan')
    return float(finite.mean())


def _language_cmp_columns(*, key_col: str) -> list[str]:
    """Column layout shared by language comparison tables."""
    columns = [key_col]
    for info_col in ('Data Count', 'Docs No Labels', 'Decent Labels'):
        columns.extend([f'{info_col}_current', f'{info_col}_base'])
    for _, metric_col in _CMP_METRICS_CORPORA:
        columns.extend([f'{metric_col}_current', f'{metric_col}_base', f'{metric_col}_diff'])
    return columns


def language_from_corpus_name(corpus_name: Any) -> str | None:
    """Extract language prefix from ``<lang>_...`` corpus names."""
    name = str(corpus_name).strip()
    if '_' not in name:
        return None
    prefix = name.split('_', 1)[0].lower()
    return prefix if prefix in LANG_PREFIXES else None


def _macro_head_metrics(*, gold: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Compute macro-head metrics for one document subset and class subset."""
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

    tp = np.logical_and(pred, gold).sum(axis=0, dtype=np.int64)
    fp = np.logical_and(pred, np.logical_not(gold)).sum(axis=0, dtype=np.int64)
    fn = np.logical_and(np.logical_not(pred), gold).sum(axis=0, dtype=np.int64)
    tn = np.logical_and(np.logical_not(pred), np.logical_not(gold)).sum(axis=0, dtype=np.int64)
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


def label_from_cat_id(*, cat_id: str) -> str:
    """Return class label format used in per-class tables for one cat id."""
    return '"' + get_cat_name(cat_id) + '"'


def labels_for_cat_ids(*, cat_ids: set[str]) -> dict[str, str]:
    """Map class table label to category id for requested ids."""
    return {label_from_cat_id(cat_id=cat_id): cat_id for cat_id in cat_ids}


def validate_subset_ids_in_corpora(
    *,
    current_corpus: Any,
    base_corpus: Any,
    subset_ids: set[str],
    subset_name: str,
    require_all: bool = True,
) -> None:
    """Validate subset ids against corpus cat lists as an early-fast check."""
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
    """Filter non-aggregate class rows to a category-id subset."""
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
    """Rebuild one run's macro aggregate row as the mean of its per-class metrics.

    The rebuilt row uses standard macro averaging (mean of per-class Precision,
    Recall, F1 and False Positive Count over ``class_ids``). When ``match_label``
    is given, an existing row with that label is located and relabeled to
    ``row_label`` in place; this maps ``evaluate.py``'s ``All_macro_relevant``
    row onto the thesis ``All_macro_head`` label. When no matching row exists the
    row is inserted (optionally after ``insert_after_label``).
    """
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


def diff_only_df(
    *,
    df: pd.DataFrame,
    key_col: str,
    cmp_metrics: tuple[tuple[str, str], ...] = _CMP_METRICS_CLASSES,
) -> pd.DataFrame:
    """Drop base metric columns while preserving current values and diffs."""
    drop_cols = {f'{col}_base' for _, col in cmp_metrics}
    cols = [key_col]
    cols.extend(col for col in df.columns if col != key_col and col not in drop_cols)
    return df.loc[:, cols]


def build_label_to_cat_id_map(*, cat_ids: Sequence[str]) -> dict[str, str]:
    """Build mapping from class label to raw class id for stable joins."""
    return {safe_cat_label(cat_id=cat_id): cat_id for cat_id in cat_ids}


def format_class_id(cat_id: Any) -> str:
    """Format a class id as plain ``<id>`` or empty string for missing values."""
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
    """Attach ``class_id`` column to one class-level dataframe."""
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


def build_class_confusion_counts_df(
    *,
    current_df: pd.DataFrame,
    base_df: pd.DataFrame,
    gold_map: GoldLabelMap,
    cat_ids: Sequence[str],
    current_thr_vec: np.ndarray,
    base_thr_vec: np.ndarray,
) -> pd.DataFrame:
    """Build per-class confusion-count comparison table for current vs base."""
    article_ids = list(current_df['article_id'])
    gold_matrix = gold_map.gold_matrix(article_ids=article_ids, cat_ids=cat_ids).astype(bool)
    not_gold_matrix = np.logical_not(gold_matrix)
    current_pred = build_pred_matrix(df=current_df, cat_ids=cat_ids, thr_vec=current_thr_vec)
    base_pred = build_pred_matrix(df=base_df, cat_ids=cat_ids, thr_vec=base_thr_vec)

    fp_current = np.logical_and(current_pred, not_gold_matrix).sum(axis=0, dtype=np.int64)
    fp_base = np.logical_and(base_pred, not_gold_matrix).sum(axis=0, dtype=np.int64)
    fn_current = np.logical_and(np.logical_not(current_pred), gold_matrix).sum(axis=0, dtype=np.int64)
    fn_base = np.logical_and(np.logical_not(base_pred), gold_matrix).sum(axis=0, dtype=np.int64)
    tp_current = np.logical_and(current_pred, gold_matrix).sum(axis=0, dtype=np.int64)
    tp_base = np.logical_and(base_pred, gold_matrix).sum(axis=0, dtype=np.int64)
    tn_current = np.logical_and(np.logical_not(current_pred), not_gold_matrix).sum(axis=0, dtype=np.int64)
    tn_base = np.logical_and(np.logical_not(base_pred), not_gold_matrix).sum(axis=0, dtype=np.int64)

    rows: list[dict[str, Any]] = []
    for idx, cat_id in enumerate(cat_ids):
        rows.append(
            {
                'IPTC Category': safe_cat_label(cat_id=cat_id),
                'class_id': format_class_id(cat_id=cat_id),
                'Fp_current': int(fp_current[idx]),
                'Fp_base': int(fp_base[idx]),
                'Fp_diff': int(fp_current[idx] - fp_base[idx]),
                'Fn_current': int(fn_current[idx]),
                'Fn_base': int(fn_base[idx]),
                'Fn_diff': int(fn_current[idx] - fn_base[idx]),
                'Tp_current': int(tp_current[idx]),
                'Tp_base': int(tp_base[idx]),
                'Tp_diff': int(tp_current[idx] - tp_base[idx]),
                'Tn_current': int(tn_current[idx]),
                'Tn_base': int(tn_base[idx]),
                'Tn_diff': int(tn_current[idx] - tn_base[idx]),
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
    """Build per-class threshold table for current/base runs."""
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
    """Build one summary row from current/base metric series."""
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
    """Macro-average precision/recall/f1 over the rows of ``sub_df``."""
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


# ---------------------------------------------------------------------------
# Top-change analysis
# ---------------------------------------------------------------------------

def build_top_change_dfs(*, classes_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build ranked improved and degraded category tables."""
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
    """Run per-class McNemar tests on paired current/base predictions.

    ``n10`` counts articles where the current model is correct and the base
    model is wrong. ``n01`` counts the opposite. Rows with too few
    disagreements do not pass significance and receive ``NaN`` p-values.

    Raw ``mcnemar_p_value`` entries are adjusted **across classes** with
    Benjamini-Hochberg FDR (``mcnemar_p_value_fdr``). Pass flags use the FDR
    column with ``alpha``, same pattern as Brier and bootstrap PR-AUC.

    :param current_thr_vec: Per-class threshold vector for the current run
        (aligned with ``cat_ids``).
    :param base_thr_vec: Per-class threshold vector for the base run.
    """
    article_ids = list(current_df['article_id'])
    gold_matrix = gold_map.gold_matrix(article_ids=article_ids, cat_ids=cat_ids).astype(bool)
    current_pred = build_pred_matrix(df=current_df, cat_ids=cat_ids, thr_vec=current_thr_vec)
    base_pred = build_pred_matrix(df=base_df, cat_ids=cat_ids, thr_vec=base_thr_vec)
    current_correct = current_pred == gold_matrix
    base_correct = base_pred == gold_matrix
    not_gold_matrix = np.logical_not(gold_matrix)

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
    """Return asymptotic McNemar p-value with continuity correction."""
    # this test does not care about the samples where models disagree
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
    """Attach McNemar significance columns to top improved/degraded tables."""
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
    """Merge McNemar rows and expose a direction-aware pass flag."""
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
    """Apply Benjamini-Hochberg FDR correction preserving input order."""
    adjusted = multipletests(p_values, method='fdr_bh')[1]
    return np.asarray(adjusted, dtype=float)


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


# ---------------------------------------------------------------------------
# Article-level alignment helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Article-level F1 deltas and entity impact
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
    """Compute per-article F1 for current/base and their delta.

    :param current_thr_vec: Per-class threshold vector for the current run.
    :param base_thr_vec: Per-class threshold vector for the base run.
    """
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
    """Compute per-article F1 scores from prediction and gold matrices.

    :param pred_matrix: Boolean ancestor-normalized prediction matrix.
    """
    gold_bool = gold_matrix.astype(bool)
    tp = np.logical_and(pred_matrix, gold_bool).sum(axis=1, dtype=np.int32)
    fp = np.logical_and(pred_matrix, np.logical_not(gold_bool)).sum(axis=1, dtype=np.int32)
    fn = np.logical_and(np.logical_not(pred_matrix), gold_bool).sum(axis=1, dtype=np.int32)
    denom = 2 * tp + fp + fn
    f1 = np.zeros_like(denom, dtype=float)
    valid = denom > 0
    f1[valid] = (2 * tp[valid]) / denom[valid]
    return f1


def build_article_confusion_diff_df(
    *,
    current_df: pd.DataFrame,
    base_df: pd.DataFrame,
    gold_map: GoldLabelMap,
    cat_ids: Sequence[str],
    current_thr_vec: np.ndarray,
    base_thr_vec: np.ndarray,
) -> pd.DataFrame:
    """Build per-article TP/TN/FP/FN diff (current minus base).

    :param current_thr_vec: Per-class threshold vector for the current run.
    :param base_thr_vec: Per-class threshold vector for the base run.
    """
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
    *, pred_matrix: np.ndarray, gold_matrix: np.ndarray
) -> Mapping[str, np.ndarray]:
    """Compute per-article TP/TN/FP/FN counts from prediction and gold matrices.

    :param pred_matrix: Boolean ancestor-normalized prediction matrix.
    """
    gold_bool = gold_matrix.astype(bool)
    tp = np.logical_and(pred_matrix, gold_bool).sum(axis=1, dtype=np.int32)
    fp = np.logical_and(pred_matrix, np.logical_not(gold_bool)).sum(axis=1, dtype=np.int32)
    fn = np.logical_and(np.logical_not(pred_matrix), gold_bool).sum(axis=1, dtype=np.int32)
    tn = np.int32(gold_matrix.shape[1]) - tp - fp - fn
    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}


def build_entity_impact_all_df(
    *,
    current_df: pd.DataFrame,
    article_f1_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the complete, un-thresholded entity impact table sorted by entity score.

    :return: All entities ranked by ``entity_score`` with an ``AVG`` footer row.
    """
    footer_ids = {'gkbid': 'AVG', 'stdform': 'AVG'}
    entity_df = build_entity_impact_df(current_df=current_df, article_f1_df=article_f1_df)
    if entity_df.empty:
        empty = entity_df.reindex(columns=entity_impact_columns())
        return append_avg_footer_row(df=empty, id_col_values=footer_ids)

    all_entities = entity_df.sort_values(by='entity_score', ascending=False)
    return append_avg_footer_row(df=all_entities, id_col_values=footer_ids)


def build_entity_impact_df(*, current_df: pd.DataFrame, article_f1_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate entity impact from article F1 deltas."""
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
    """Explode article entities into one row per entity occurrence."""
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
    """Select representative value per gkbid by most frequent non-empty entry.

    :param value_col: Column whose most frequent non-empty value is picked per gkbid.
    """
    names = df[['gkbid', value_col]].dropna(subset=['gkbid']).copy()
    names[value_col] = names[value_col].fillna('').astype(str).str.strip()
    names = names[names[value_col] != '']
    if names.empty:
        return pd.DataFrame(columns=['gkbid', value_col])
    counts = names.groupby(['gkbid', value_col], as_index=False).size()
    counts = counts.sort_values(by=['gkbid', 'size', value_col], ascending=[True, False, True])
    return counts.drop_duplicates(subset=['gkbid'], keep='first')[['gkbid', value_col]]


def append_avg_footer_row(*, df: pd.DataFrame, id_col_values: Mapping[str, str]) -> pd.DataFrame:
    """Append one footer row with numeric means and fixed identifier labels."""
    footer: dict[str, Any] = {**id_col_values}
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    for col in numeric_cols:
        footer[col] = float(df[col].mean()) if not df.empty else float('nan')
    return pd.concat([df, pd.DataFrame([footer], columns=df.columns)], ignore_index=True)


def entity_impact_columns() -> list[str]:
    """Return output column order for entity impact tables."""
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
# PR-AUC
# ---------------------------------------------------------------------------

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
            aggregation='macro_all',
            current=_safe_mean(values=pr_auc_df['pr_auc_current']),
            base=_safe_mean(values=pr_auc_df['pr_auc_base']),
        )
    )
    tail_mask = pr_auc_df['positive_support'] < MACRO_HEAD_MIN_SUPPORT
    tail_sub = pr_auc_df[tail_mask]
    rows.append(
        _pr_auc_row(
            aggregation='macro_tail',
            current=_safe_mean(values=tail_sub['pr_auc_current']),
            base=_safe_mean(values=tail_sub['pr_auc_base']),
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


def _avg_filtered(*, values_dict: Mapping[str, float], predicate: Callable[[str], bool]) -> float:
    """Mean of dict values whose key matches ``predicate`` and is not NaN."""
    matched = [v for k, v in values_dict.items() if predicate(k) and not np.isnan(v)]
    return float(np.mean(matched)) if matched else float('nan')


def _prefix_predicate(prefix: str) -> Callable[[str], bool]:
    """Predicate matching corpus names starting with ``<prefix>_``."""
    return lambda name: name.startswith(f'{prefix}_')


def _contains_predicate(token: str) -> Callable[[str], bool]:
    """Predicate matching corpus names containing ``token`` (case-insensitive)."""
    lower = token.lower()
    return lambda name: lower in name.lower()


# ---------------------------------------------------------------------------
# Score matrix and average precision
# ---------------------------------------------------------------------------

def micro_pr_auc(*, gold_matrix: np.ndarray, scores: np.ndarray) -> float:
    """Compute micro PR-AUC by flattening across all classes."""
    if gold_matrix.size == 0 or int(gold_matrix.sum()) == 0:
        return float('nan')
    return float(average_precision_score(gold_matrix, scores, average='micro'))


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


def build_pred_matrix(*, df: pd.DataFrame, cat_ids: Sequence[str], thr_vec: np.ndarray) -> np.ndarray:
    """Build ancestor-normalized boolean prediction matrix aligned with ``cat_ids``.

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
    """
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


def average_precision(*, y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute discrete PR-AUC as average precision."""
    if int(y_true.sum()) == 0:
        return np.nan
    return float(average_precision_score(y_true, y_score))


# ---------------------------------------------------------------------------
# Serialization (CSV / Excel)
# ---------------------------------------------------------------------------

def write_csv(
    *,
    result: ComparisonResult,
    output_path: Path,
    result_sheets: Sequence[tuple[str, str, bool]] = _RESULT_SHEETS,
) -> None:
    """Persist comparison outputs into CSV files."""
    output_path.mkdir(parents=True, exist_ok=True)
    for attr, filename, _ in result_sheets:
        getattr(result, attr).to_csv(output_path / f'{filename}.csv', index=False)


def write_excel(
    *,
    result: ComparisonResult,
    output_path: Path,
    result_sheets: Sequence[tuple[str, str, bool]] = _RESULT_SHEETS,
) -> None:
    """Persist comparison outputs into one Excel workbook."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path) as writer:
        for attr, sheet_name, include_index in result_sheets:
            getattr(result, attr).to_excel(writer, sheet_name=sheet_name, index=include_index)


def log_top_changes(*, result: ComparisonResult, top_n: int) -> None:
    """Log concise previews of top improved and degraded categories."""
    if not result.top_improved_categories.empty:
        LOG.info('Top improved categories:\n%s', result.top_improved_categories.head(top_n).to_string(index=False))
    if not result.top_degraded_categories.empty:
        LOG.info('Top degraded categories:\n%s', result.top_degraded_categories.head(top_n).to_string(index=False))
    if result.excel_path is not None:
        LOG.info('Saved comparison report to %s', result.excel_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
    parser.add_argument('--averaging-type', default='micro', choices=['datapoint', 'micro', 'macro'])
    parser.add_argument('--top-n', type=int, default=20, help='Preview size for improved/degraded categories.')
    parser.add_argument('--only-diff', action='store_true', help='Drop base metric columns from comparison sheets.')
    parser.add_argument(
        '--top-changes-only',
        action='store_true',
        help='Write only top_improved and top_degraded category tables.',
    )
    parser.add_argument(
        '--ignore-saved-thresholds',
        action='store_true',
        help=(
            'Disable per-class threshold autoload from each run dir. By default the script '
            f'tries {THRESHOLD_FILENAMES} and applies the loaded thresholds; pass this flag '
            'to compare both runs under a uniform --threshold-eval.'
        ),
    )
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
        top_changes_only=args.top_changes_only,
        output_path=output_path,
        use_saved_thresholds=not args.ignore_saved_thresholds,
    )


if __name__ == '__main__':
    main()
