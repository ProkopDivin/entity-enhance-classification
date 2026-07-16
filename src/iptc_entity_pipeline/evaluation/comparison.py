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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from iptc_entity_pipeline.category_sets import load_relevant_cat_ids, load_tail_cat_ids
from iptc_entity_pipeline.config import EvaluationCnf
from iptc_entity_pipeline.data_loading import sanitize_name

# ---------------------------------------------------------------------------
# Sub-module imports (used by compare_runs and re-exported for backward compat)
# ---------------------------------------------------------------------------
from iptc_entity_pipeline.evaluation._confusion import (  # noqa: F401 -- re-export
    build_paired_matrices,
    build_pred_matrix,
    build_score_matrix,
    confusion_counts,
    safe_mean,
)
from iptc_entity_pipeline.evaluation.run_loading import (  # noqa: F401 -- re-export
    THRESHOLD_FILENAMES,
    ArticleEntity,
    ArticleEvalRecord,
    GoldArticle,
    GoldLabelMap,
    RunEval,
    add_prob_columns,
    build_aligned_df,
    build_article_records,
    build_run,
    gold_df_from_corpus,
    load_custom_thresholds,
    load_run,
    norm_cat_ids,
    records_to_df,
    thresholds_vector,
)
from iptc_entity_pipeline.evaluation.comparison_tables import (  # noqa: F401 -- re-export
    AGG_CLASS_ROWS,
    AGG_CORPUS_ROWS,
    CLASS_MACRO_ROW,
    CLASS_TAIL_ROW,
    LANG_PREFIXES,
    MACRO_HEAD_MIN_SUPPORT,
    MACRO_HEAD_ROW,
    MACRO_TAIL_ROW,
    MATCH_HEAD_ROW,
    SUMMARY_ROWS,
    SUPPORT_BUCKETS,
    _CLASSES_COMPARISON_DROP_COLS,
    _CMP_METRICS_CLASSES,
    _CMP_METRICS_CORE,
    _CMP_METRICS_CORPORA,
    _CMP_METRICS_REPORT,
    apply_macro_rows,
    build_class_confusion_counts_df,
    build_class_thresholds_df,
    build_cmp_df,
    build_corpora_macro_head_cmp_df,
    build_label_to_cat_id_map,
    build_language_cmp_df,
    build_summary_df,
    class_subset_by_ids,
    diff_only_df,
    format_class_id,
    label_from_cat_id,
    labels_for_cat_ids,
    language_from_corpus_name,
    replace_macro_row_in_run_classes_df,
    safe_cat_label,
    validate_subset_ids_in_corpora,
    with_class_id_column,
)
from iptc_entity_pipeline.evaluation.significance import (  # noqa: F401 -- re-export
    EUROSPORT_TOKEN,
    MCNEMAR_ALPHA,
    MCNEMAR_MIN_DISAGREEMENTS,
    add_mcnemar_to_top_change_dfs,
    average_precision,
    benjamini_hochberg,
    build_mcnemar_significance_df,
    build_pr_auc_dfs,
    build_pr_auc_summary_df,
    mcnemar_p_value,
    micro_pr_auc,
    per_corpus_pr_auc,
)
from iptc_entity_pipeline.evaluation.article_analysis import (  # noqa: F401 -- re-export
    CHANGE_THRESHOLDS,
    TOP_CHANGE_N,
    append_avg_footer_row,
    build_article_confusion_diff_df,
    build_article_f1_diff_df,
    build_entity_impact_all_df,
    build_entity_impact_df,
    build_top_change_dfs,
    build_top_change_stats_dfs,
    choose_mode_by_gkbid,
    compute_article_confusion,
    compute_article_f1,
    entity_impact_columns,
    explode_entities,
    shared_article_ids,
    subset_by_ids,
    top_level_from_label,
)

# Keep the statsmodels mcnemar import accessible on this module for test monkeypatching.
from statsmodels.stats.contingency_tables import mcnemar  # noqa: F401 -- re-export for tests

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result sheets configuration
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Comparison orchestration
# ---------------------------------------------------------------------------

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

    current_classes_df = apply_macro_rows(
        classes_df=current_run.classes_df,
        relevant_cat_ids=relevant_cat_ids,
        tail_cat_ids=tail_cat_ids,
    )
    base_classes_df = apply_macro_rows(
        classes_df=base_run.classes_df,
        relevant_cat_ids=relevant_cat_ids,
        tail_cat_ids=tail_cat_ids,
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

    def add_class_id(df: pd.DataFrame) -> pd.DataFrame:
        return with_class_id_column(df=df, key_col='IPTC Category', label_to_cat_id=label_to_cat_id)

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
            top_improved_categories=add_class_id(top_improved_df),
            top_degraded_categories=add_class_id(top_degraded_df),
            top_improved_stats=empty_df,
            top_degraded_stats=empty_df,
            pr_auc_per_class=empty_df,
            pr_auc_summary=empty_df,
            entity_impact_all=empty_df,
            article_confusion_diff=empty_df,
            current_corpora=current_run.corpora_df,
            current_classes=add_class_id(current_classes_df),
            base_corpora=base_run.corpora_df,
            base_classes=add_class_id(base_classes_df),
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
        classes_comparison=add_class_id(classes_cmp_df),
        class_confusion_counts=class_confusion_df,
        class_thresholds=class_thresholds_df,
        summary_comparison=summary_df,
        top_improved_categories=add_class_id(top_improved_df),
        top_degraded_categories=add_class_id(top_degraded_df),
        top_improved_stats=top_improved_stats_df,
        top_degraded_stats=top_degraded_stats_df,
        pr_auc_per_class=pr_auc_df,
        pr_auc_summary=pr_auc_summary_df,
        entity_impact_all=entity_all_df,
        article_confusion_diff=article_confusion_diff_df,
        current_corpora=current_run.corpora_df,
        current_classes=add_class_id(current_classes_df),
        base_corpora=base_run.corpora_df,
        base_classes=add_class_id(base_classes_df),
        excel_path=excel_path,
    )
    if excel_path is not None:
        result_sheets = _TOP_CHANGE_CATEGORY_SHEETS if top_changes_only else _RESULT_SHEETS
        write_csv(result=result, output_path=excel_path.parent, result_sheets=result_sheets)
        write_excel(result=result, output_path=excel_path, result_sheets=result_sheets)
        log_top_changes(result=result, top_n=top_n)
    return result


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
