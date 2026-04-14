"""ClearML reporting helpers for scalar metrics, tables, and curve charts."""

from __future__ import annotations

import logging
from typing import Any, Mapping, NamedTuple, Sequence

import numpy as np
from clearml import Task

from iptc_entity_pipeline.training import CvFoldCurves, TrainingResult


class ChartSpec(NamedTuple):
    """Specification for a single train-vs-test scatter chart."""

    title: str
    yaxis: str
    train_curve: Sequence[float]
    dev_curve: Sequence[float]

LOGGER = logging.getLogger(__name__)


def report_eval_scalars(*, logger: Any, title: str, row: Mapping[str, Any], iteration: int = 0) -> None:
    """Report shared eval metrics to ClearML scalar charts."""
    logger.report_scalar(title=title, series='Precision', value=row['Precision'], iteration=iteration)
    logger.report_scalar(title=title, series='Recall', value=row['Recall'], iteration=iteration)
    logger.report_scalar(title=title, series='F1', value=row['F1'], iteration=iteration)


def report_cv_std_scalars(
    *,
    logger: Any,
    row: Mapping[str, Any],
    title: str = 'Cross Validation Results',
    iteration: int = 0,
) -> None:
    """Report standard deviations from CV objective metrics."""
    logger.report_scalar(
        title=title,
        series='Precision_std',
        value=float(row['Precision_std']),
        iteration=iteration,
    )
    logger.report_scalar(
        title=title,
        series='Recall_std',
        value=float(row['Recall_std']),
        iteration=iteration,
    )
    logger.report_scalar(
        title=title,
        series='F1_std',
        value=float(row['F1_std']),
        iteration=iteration,
    )


def report_cv_outputs(*, task: Task, logger: Any, trials_df: Any, folds_df: Any, cv_dev_df: Any) -> None:
    """Report CV tables and upload all frames as artifacts."""
    logger.report_table(title='Cross Validation', series='Folds', iteration=0, table_plot=folds_df)
    logger.report_table(title='Cross Validation', series='Cross Validation Results', iteration=0, table_plot=trials_df)
    task.upload_artifact('cv_trials_dataframe', artifact_object=trials_df)
    task.upload_artifact('cv_folds_dataframe', artifact_object=folds_df)
    task.upload_artifact('cv_dev_summary_dataframe', artifact_object=cv_dev_df)


def log_stage(*, task: Task, message: str, print_logs: bool) -> None:
    """Log pipeline stage both to logger and ClearML task text output."""
    LOGGER.info(message)
    task.get_logger().report_text(message, print_console=print_logs)


def _scatter_xy(data: Sequence[float]) -> Any:
    """Convert per-epoch values to scatter plot coordinates."""
    epochs = np.arange(1, len(data) + 1)
    return np.column_stack((epochs, np.asarray(list(data), dtype=float)))


def report_cv_fold_curve_charts(
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


def report_train_test_curve_charts(*, logger: Any, result: TrainingResult) -> None:
    """Report final-model train vs test curves across epochs."""
    charts = (
        ChartSpec('Final Model Loss', 'loss', result.train_loss_per_epoch, result.dev_loss_per_epoch),
        ChartSpec('Final Model F1', 'f1', result.train_f1_per_epoch, result.dev_f1_per_epoch),
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
            series='test',
            iteration=0,
            scatter=_scatter_xy(chart.dev_curve),
            xaxis='epoch',
            yaxis=chart.yaxis,
            mode='lines+markers',
        )
