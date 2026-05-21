"""ClearML reporting helpers for scalar metrics, tables, and curve charts."""

from __future__ import annotations

import logging
from typing import Any, Mapping, NamedTuple, Sequence

import numpy as np
from clearml import Task

from iptc_entity_pipeline.training import CvFoldCurves, TrainingResult


class ChartSpec(NamedTuple):
    """Specification for a single final-model scatter chart."""

    title: str
    yaxis: str
    train_curve: Sequence[float]
    dev_curve: Sequence[float]

LOGGER = logging.getLogger(__name__)
STD_METRIC_SERIES = (
    'Precision_std',
    'Recall_std',
    'F1_std',
    'Precision_macro_relevant_std',
    'Recall_macro_relevant_std',
    'F1_macro_relevant_std',
)
METRIC_SERIES = (
    'Precision',
    'Recall',
    'F1',
    'Precision_macro_relevant',
    'Recall_macro_relevant',
    'F1_macro_relevant',
)

def _relevant_macro_scalar_row(macro_row: Any) -> dict[str, float]:
    """Map classes ``All-relevant-macro`` row to :data:`METRIC_SERIES` macro-relevant keys."""
    return {
        'Precision_macro_relevant': float(macro_row['Precision']),
        'Recall_macro_relevant': float(macro_row['Recall']),
        'F1_macro_relevant': float(macro_row['F1']),
    }


def build_test_scalar_metrics(
    *,
    df_corpora_test: Any,
    df_classes_test: Any,
    objective_row: str,
) -> dict[str, float]:
    """Build combined test scalar dict (objective corpora row + ``All-relevant-macro``).

    Corpora ``All-macro`` / ``All-micro`` and classes ``All-relevant-macro`` are different
    metrics and must not share one row label.

    :param df_corpora_test: Corpora evaluation table.
    :param df_classes_test: Classes evaluation table.
    :param objective_row: Corpora row for objective metrics (e.g. ``All-micro``).
    :return: Keys aligned with :data:`METRIC_SERIES` for artifacts and pipeline transport.
    """
    from iptc_entity_pipeline.evaluate import CLASS_RELEVANT_MACRO_ROW

    corpora_row = df_corpora_test.loc[objective_row]
    macro_row = df_classes_test.loc[CLASS_RELEVANT_MACRO_ROW]
    objective_metrics = {
        'Precision': float(corpora_row['Precision']),
        'Recall': float(corpora_row['Recall']),
        'F1': float(corpora_row['F1']),
    }
    relevant_macro_metrics = _relevant_macro_scalar_row(macro_row)
    metrics = {**objective_metrics, **relevant_macro_metrics}
    LOGGER.info(
        f'Test scalar metrics: objective_row={objective_row}, '
        f'F1={objective_metrics["F1"]:.4f}, '
        f'F1_macro_relevant={relevant_macro_metrics["F1_macro_relevant"]:.4f}'
    )
    return metrics


def report_eval(*, logger: Any, title: str, row: Mapping[str, Any], iteration: int = 0) -> None:
    """Report shared eval metrics to ClearML scalar charts.

    Expects ``row`` keys from evaluation tables or :func:`cross_validation._build_cv_dev_row`
    (``METRIC_SERIES`` names: micro as ``Precision``/``Recall``/``F1``, macro-relevant
    as ``*_macro_relevant``).
    """
    for series in METRIC_SERIES:
        if series in row:
            logger.report_scalar(title=title, series=series, value=float(row[series]), iteration=iteration)



def report_cv_std(
    *,
    logger: Any,
    row: Mapping[str, Any],
    title: str = 'Cross Validation Results',
    iteration: int = 0,
) -> None:
    """Report standard deviations from CV objective metrics."""
    for series in STD_METRIC_SERIES:
        if series in row:
            logger.report_scalar(
                title=title,
                series=series,
                value=float(row[series]),
                iteration=iteration,
            )


def report_cv_curve(
    *,
    task: Task,
    logger: Any,
    trials_df: Any,
    folds_df: Any,
    cv_dev_df: Any,
    upload_artifacts: bool = False,
) -> None:
    """Report CV tables and upload all frames as artifacts."""
    logger.report_table(title='Cross Validation', series='Folds', iteration=0, table_plot=folds_df)
    logger.report_table(title='Cross Validation', series='Cross Validation Results', iteration=0, table_plot=trials_df)
    if upload_artifacts:
        task.upload_artifact('cv_trials_dataframe', artifact_object=trials_df.copy(deep=True))
        task.upload_artifact('cv_folds_dataframe', artifact_object=folds_df.copy(deep=True))
        task.upload_artifact('cv_dev_summary_dataframe', artifact_object=cv_dev_df.copy(deep=True))


def report_cv_result_tables(
    *,
    task: Task,
    clearml_logger: Any,
    cv_result: Any,
    threshold_aggregation: str,
    upload_artifacts: bool = False,
) -> None:
    """Report CV result tables and optional artifacts."""
    if cv_result.threshold_report_df is not None:
        clearml_logger.report_table(
            title='Threshold Tuning',
            series='Per-class thresholds (CV folds)',
            iteration=0,
            table_plot=cv_result.threshold_report_df,
        )
        n_classes = int(cv_result.threshold_report_df['n_folds'].gt(0).sum())
        LOGGER.info(
            f'Threshold tuning: aggregation={threshold_aggregation}, '
            f'classes_with_tuned_threshold={n_classes}/{len(cv_result.threshold_report_df)}'
        )
        if upload_artifacts:
            task.upload_artifact(
                'threshold_tuning_report',
                artifact_object=cv_result.threshold_report_df.copy(deep=True),
            )
            task.upload_artifact(
                'threshold_tuning_thresholds',
                artifact_object={
                    str(k): float(v) for k, v in (cv_result.tuned_thresholds or {}).items()
                },
            )

    if cv_result.cv_per_corpora_df is not None:
        clearml_logger.report_table(
            title='Cross Validation',
            series='Per-corpora (mean+std)',
            iteration=0,
            table_plot=cv_result.cv_per_corpora_df,
        )
        if upload_artifacts:
            task.upload_artifact(
                'cv_per_corpora_dataframe',
                artifact_object=cv_result.cv_per_corpora_df.copy(deep=True),
            )
    if cv_result.cv_per_class_df is not None:
        clearml_logger.report_table(
            title='Cross Validation',
            series='Per-class (mean+std)',
            iteration=0,
            table_plot=cv_result.cv_per_class_df,
        )
        if upload_artifacts:
            task.upload_artifact(
                'cv_per_class_dataframe',
                artifact_object=cv_result.cv_per_class_df.copy(deep=True),
            )


def report_cv(
    *,
    task: Task,
    logger: Any,
    report: Any,
    upload_artifacts: bool = False,
) -> None:
    """Report all CV results to ClearML in a single call.

    :param report: :class:`CvReport` carrying all data needed for
        trial/fold tables, per-fold loss/F1 curves, and
        threshold/per-class/per-corpora result tables.
    """
    report_cv_curve(
        task=task,
        logger=logger,
        trials_df=report.trials_df,
        folds_df=report.folds_df,
        cv_dev_df=report.cv_dev_df,
        upload_artifacts=upload_artifacts,
    )
    report_cv_fold(logger=logger, fold_curves=report.fold_curves)
    report_cv_result_tables(
        task=task,
        clearml_logger=logger,
        cv_result=report,
        threshold_aggregation=report.threshold_aggregation,
        upload_artifacts=upload_artifacts,
    )


def log_stage(*, task: Task, message: str, print_logs: bool) -> None:
    """Log pipeline stage both to logger and ClearML task text output."""
    LOGGER.info(message)
    task.get_logger().report_text(message, print_console=print_logs)


def _scatter_xy(data: Sequence[float]) -> Any:
    """Convert per-epoch values to scatter plot coordinates."""
    epochs = np.arange(1, len(data) + 1)
    return np.column_stack((epochs, np.asarray(list(data), dtype=float)))


def report_cv_fold(
    *,
    logger: Any,
    fold_curves: Sequence[CvFoldCurves],
) -> None:
    """Report loss and F1 curves for folds of the best CV configuration."""
    if not fold_curves:
        return

    for fold_curve in fold_curves:
        iteration = fold_curve.fold_id
        logger.report_scatter2d(
            title='Cross Validation Fold Loss',
            series=f'train fold {fold_curve.fold_id}',
            iteration=iteration,
            scatter=_scatter_xy(fold_curve.train_loss_per_epoch),
            xaxis='epoch',
            yaxis='loss',
            mode='lines+markers',
        )
        logger.report_scatter2d(
            title='Cross Validation Fold Loss',
            series=f'dev fold {fold_curve.fold_id}',
            iteration=iteration,
            scatter=_scatter_xy(fold_curve.dev_loss_per_epoch),
            xaxis='epoch',
            yaxis='loss',
            mode='lines+markers',
        )
        logger.report_scatter2d(
            title='Cross Validation Fold F1',
            series=f'train fold {fold_curve.fold_id}',
            iteration=iteration,
            scatter=_scatter_xy(fold_curve.train_f1_per_epoch),
            xaxis='epoch',
            yaxis='f1',
            mode='lines+markers',
        )
        logger.report_scatter2d(
            title='Cross Validation Fold F1',
            series=f'dev fold {fold_curve.fold_id}',
            iteration=iteration,
            scatter=_scatter_xy(fold_curve.dev_f1_per_epoch),
            xaxis='epoch',
            yaxis='f1',
            mode='lines+markers',
        )
        logger.report_scatter2d(
            title='Cross Validation Fold F1 Macro Relevant',
            series=f'train fold {fold_curve.fold_id}',
            iteration=iteration,
            scatter=_scatter_xy(fold_curve.train_macro_relevant_f1_per_epoch),
            xaxis='epoch',
            yaxis='f1',
            mode='lines+markers',
        )
        logger.report_scatter2d(
            title='Cross Validation Fold F1 Macro Relevant',
            series=f'dev fold {fold_curve.fold_id}',
            iteration=iteration,
            scatter=_scatter_xy(fold_curve.dev_macro_relevant_f1_per_epoch),
            xaxis='epoch',
            yaxis='f1',
            mode='lines+markers',
        )


def report_test_curve(*, logger: Any, result: TrainingResult, dev_series: str = 'test') -> None:
    """Report final-model train vs validation-like curves across epochs."""
    charts = (
        ChartSpec('Final Model Loss', 'loss', result.train_loss_per_epoch, result.dev_loss_per_epoch),
        ChartSpec('Final Model F1', 'f1', result.train_f1_per_epoch, result.dev_f1_per_epoch),
        ChartSpec(
            'Final Model F1 Macro Relevant',
            'f1',
            result.train_macro_relevant_f1_per_epoch,
            result.dev_macro_relevant_f1_per_epoch,
        ),
        ChartSpec('Final Model Precision', 'precision', result.train_precision_per_epoch, result.dev_precision_per_epoch),
        ChartSpec('Final Model Recall', 'recall', result.train_recall_per_epoch, result.dev_recall_per_epoch),
    )
    for chart in charts:
        if not chart.train_curve and not chart.dev_curve:
            continue
        logger.report_scatter2d(
            title=chart.title,
            series='train',
            iteration=0,
            scatter=_scatter_xy(chart.train_curve),
            xaxis='epoch',
            yaxis=chart.yaxis,
            mode='lines+markers',
        )
        logger.report_scatter2d(
            title=chart.title,
            series=dev_series,
            iteration=0,
            scatter=_scatter_xy(chart.dev_curve),
            xaxis='epoch',
            yaxis=chart.yaxis,
            mode='lines+markers',
        )


def report_test_eval_scalars(
    *,
    clearml_logger: Any,
    df_corpora_test: Any,
    df_classes_test: Any,
    objective_row: str,
) -> None:
    """Report final-test scalars under ``Test Evaluation Results``.

    Logs in order:

    1. Objective metrics from the corpora table row (``objective_row``, e.g. ``All-micro``).
    2. Relevant-class macro from the classes table row ``All-relevant-macro`` (not corpora ``All-macro``).
    3. Optional extra corpora ``All-micro`` row at iteration 1 when ``objective_row`` is not ``All-micro``.

    :param clearml_logger: ClearML logger instance.
    :param df_corpora_test: Test corpora evaluation DataFrame.
    :param df_classes_test: Test classes evaluation DataFrame.
    :param objective_row: Corpora row used as the pipeline objective (e.g. ``All-micro``).
    """
    from iptc_entity_pipeline.evaluate import CLASS_RELEVANT_MACRO_ROW

    title = 'Test Evaluation Results'
    objective_metrics = df_corpora_test.loc[objective_row].to_dict()
    report_eval(logger=clearml_logger, title=title, row=objective_metrics, iteration=0)

    relevant_macro_metrics = _relevant_macro_scalar_row(df_classes_test.loc[CLASS_RELEVANT_MACRO_ROW])
    report_eval(logger=clearml_logger, title=title, row=relevant_macro_metrics, iteration=0)

    LOGGER.info(
        f'Test Evaluation Results reported: objective_row={objective_row}, '
        f'F1={float(objective_metrics["F1"]):.4f}, '
        f'F1_macro_relevant={relevant_macro_metrics["F1_macro_relevant"]:.4f}'
    )

    if objective_row != 'All-micro':
        row_micro = df_corpora_test.loc['All-micro'].to_dict()
        report_eval(logger=clearml_logger, title=title, row=row_micro, iteration=1)


def report_test_eval_tables(
    *,
    clearml_logger: Any,
    df_corpora_test: Any,
    df_classes_test: Any,
) -> None:
    """Report final-test evaluation tables to ClearML.

    CV summary tables are reported in the ``run_cv`` step only.
    """
    clearml_logger.report_table(
        title='Test Evaluation', series='Corpora Dataframe', iteration=0, table_plot=df_corpora_test,
    )
    clearml_logger.report_table(
        title='Test Evaluation', series='Classes Dataframe', iteration=0, table_plot=df_classes_test,
    )


def conf_logging(*, level: int = logging.INFO) -> None:
    """Ensure component worker processes emit INFO logs to console."""
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=level, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
        return

    if root_logger.level > level:
        root_logger.setLevel(level)
    for handler in root_logger.handlers:
        if handler.level > level:
            handler.setLevel(level)