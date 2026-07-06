"""Evaluation metrics, run comparison, reporting, and LaTeX table helpers.

The public API is split across submodules; commonly used symbols are
re-exported here for convenience.
"""

from __future__ import annotations

from iptc_entity_pipeline.evaluation.comparison import build_path, compare_runs, load_custom_thresholds, load_run
from iptc_entity_pipeline.evaluation.evaluate import (
    CLASS_DATAPOINT_ROW,
    CLASS_MICRO_ROW,
    CLASS_RELEVANT_MACRO_ROW,
    CORPORA_DATAPOINT_ROW,
    CORPORA_MACRO_ROW,
    CORPORA_MICRO_ROW,
    REMOVED_CAT_IDS,
    aggregate_fold_dfs,
    evaluate_classes,
    evaluate_corpora,
    evaluate_predictions,
    filter_and_normalize,
    get_cat_name,
    get_iptc_topics,
    normalize_pred_cats,
    pred_cats_from_matrix,
)
from iptc_entity_pipeline.evaluation.reporting import (
    METRIC_SERIES,
    STD_METRIC_SERIES,
    ChartSpec,
    build_test_scalar_metrics,
    conf_logging,
    log_stage,
    objective_suffix,
    report_cv,
    report_cv_result_tables,
    report_cv_std,
    report_eval,
    report_test_curve,
    report_test_eval_scalars,
    report_test_eval_tables,
)

__all__ = [
    'ChartSpec',
    'CLASS_DATAPOINT_ROW',
    'CLASS_MICRO_ROW',
    'CLASS_RELEVANT_MACRO_ROW',
    'CORPORA_DATAPOINT_ROW',
    'CORPORA_MACRO_ROW',
    'CORPORA_MICRO_ROW',
    'METRIC_SERIES',
    'REMOVED_CAT_IDS',
    'STD_METRIC_SERIES',
    'aggregate_fold_dfs',
    'build_path',
    'build_test_scalar_metrics',
    'compare_runs',
    'conf_logging',
    'evaluate_classes',
    'evaluate_corpora',
    'evaluate_predictions',
    'filter_and_normalize',
    'get_cat_name',
    'get_iptc_topics',
    'load_custom_thresholds',
    'load_run',
    'log_stage',
    'normalize_pred_cats',
    'objective_suffix',
    'pred_cats_from_matrix',
    'report_cv',
    'report_cv_result_tables',
    'report_cv_std',
    'report_eval',
    'report_test_curve',
    'report_test_eval_scalars',
    'report_test_eval_tables',
]
