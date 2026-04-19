"""Cross-validation pipeline step: hyperparameter sweep with k-fold CV."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from iptc_entity_pipeline.config import (
    CvCnf,
    EvaluationCnf,
    ModelCnf,
    TrainingCnf,
)
from iptc_entity_pipeline.dataset_builder import (
    build_multilabel_targets,
    slice_dataset,
    to_numpy_array,
)
from iptc_entity_pipeline.legacy_reuse import evaluateModel
from iptc_entity_pipeline.training import (
    CvFoldCurves,
    combo_params_json,
    get_obj_row,
    train_model,
)


@dataclass
class FoldScores:
    """Per-fold metric accumulator for cross-validation."""

    loss: list[float] = field(default_factory=list)
    epochs: list[float] = field(default_factory=list)
    precision: list[float] = field(default_factory=list)
    recall: list[float] = field(default_factory=list)
    f1: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class CvResult:
    """Outputs of the cross-validation pipeline step."""

    cv_dev_df: Any
    best_model_config: dict[str, Any]
    best_training_config: dict[str, Any]
    objective_metrics: dict[str, Any]


@dataclass(frozen=True)
class CombinationResult:
    """Aggregated outputs for one hyperparameter combination across all folds."""

    trial_row: dict[str, Any]
    fold_rows: list[dict[str, Any]]
    fold_curves: tuple[CvFoldCurves, ...]
    model_config: ModelCnf
    training_config: TrainingCnf


@dataclass(frozen=True)
class BestSelection:
    """Best-trial selection across all hyperparameter combinations."""

    best_trial: dict[str, Any]
    best_model_config: ModelCnf
    best_training_config: TrainingCnf
    best_fold_curves: tuple[CvFoldCurves, ...]
    trial_rows: list[dict[str, Any]]
    fold_rows: list[dict[str, Any]]


def prepare_cv_arrays(*, train_data: Any) -> tuple[np.ndarray, np.ndarray]:
    """Extract X/Y NumPy arrays for stratified split planning."""
    x_full = to_numpy_array(matrix_like=train_data.X)
    y_full = (
        to_numpy_array(matrix_like=train_data.Y)
        if hasattr(train_data, 'Y')
        else build_multilabel_targets(corpus=train_data.corpus)
    )
    return x_full, y_full


def log_cv_plan(
    *,
    total_combinations: int,
    folds_per_combo: int,
    total_trainings: int,
    py_logger: logging.Logger,
    clearml_logger: Any,
) -> None:
    """Emit the CV hyperparameter search plan to both Python and ClearML loggers."""
    cv_plan_message = (
        'CV hyperparameter search plan: '
        f'combinations={total_combinations} '
        f'folds_per_combination={folds_per_combo} '
        f'total_model_trains={total_trainings}'
    )
    py_logger.info(cv_plan_message)
    clearml_logger.report_text(cv_plan_message, print_console=True)


def log_training_progress(
    *,
    completed_trainings: int,
    total_trainings: int,
    combo_idx: int,
    total_combinations: int,
    fold_idx: int,
    folds_per_combo: int,
    py_logger: logging.Logger,
    clearml_logger: Any,
) -> None:
    """Emit per-fold training progress to both Python and ClearML loggers."""
    training_progress_message = (
        'CV training progress: '
        f'trained_models={completed_trainings}/{total_trainings} '
        f'combination={combo_idx}/{total_combinations} '
        f'fold={fold_idx}/{folds_per_combo}'
    )
    py_logger.info(training_progress_message)
    clearml_logger.report_text(training_progress_message, print_console=True)


def extract_micro_row(
    *,
    df_corpora_fold: Any,
    objective_corpora: str,
    averaging_type: str,
) -> Mapping[str, Any]:
    """Pick the per-fold metric row, preferring 'All-micro' and falling back to objective."""
    if 'All-micro' in df_corpora_fold.index:
        return df_corpora_fold.loc['All-micro'].to_dict()
    return get_obj_row(
        df_corpora=df_corpora_fold,
        objective_corpora=objective_corpora,
        averaging_type=averaging_type,
    )


def evaluate_fold(
    *,
    fit_data: Any,
    val_data: Any,
    feature_dim: int,
    model_config: ModelCnf,
    training_config: TrainingCnf,
    eval_cfg: EvaluationCnf,
    objective_corpora: str,
    fold_idx: int,
    print_logs: bool,
) -> tuple[Mapping[str, Any], float, int, CvFoldCurves]:
    """Train and evaluate one CV fold, returning per-fold metrics and curves."""
    train_result = train_model(
        train_data=fit_data,
        dev_data=val_data,
        feature_dim=feature_dim,
        model_config=model_config,
        training_config=training_config,
        print_logs=print_logs,
    )
    df_corpora_fold, _ = evaluateModel(
        model=train_result.model,
        evalData=val_data,
        evaluation_config=eval_cfg,
        customThresholds=None,
    )
    micro_row = extract_micro_row(
        df_corpora_fold=df_corpora_fold,
        objective_corpora=objective_corpora,
        averaging_type=eval_cfg.averaging_type,
    )
    fold_curve = CvFoldCurves(
        fold_id=fold_idx,
        train_loss_per_epoch=train_result.train_loss_per_epoch,
        dev_loss_per_epoch=train_result.dev_loss_per_epoch,
        train_f1_per_epoch=train_result.train_f1_per_epoch,
        dev_f1_per_epoch=train_result.dev_f1_per_epoch,
    )
    return micro_row, float(train_result.final_dev_loss), int(train_result.epochs_run), fold_curve


def summarize_combination(
    *,
    combo_idx: int,
    params_json: str,
    fold_scores: FoldScores,
) -> dict[str, Any]:
    """Aggregate per-fold scores for one combination into a single trial row."""
    return {
        'trial_id': combo_idx,
        'params': params_json,
        'epochs': float(np.mean(fold_scores.epochs)),
        'Loss_mean': float(np.mean(fold_scores.loss)),
        'Loss_std': float(np.std(fold_scores.loss)),
        'F1_mean': float(np.mean(fold_scores.f1)),
        'F1_std': float(np.std(fold_scores.f1)),
        'Precision_mean': float(np.mean(fold_scores.precision)),
        'Precision_std': float(np.std(fold_scores.precision)),
        'Recall_mean': float(np.mean(fold_scores.recall)),
        'Recall_std': float(np.std(fold_scores.recall)),
    }


def run_combination(
    *,
    combo_idx: int,
    total_combinations: int,
    model_config: ModelCnf,
    training_config: TrainingCnf,
    train_data: Any,
    x_full: np.ndarray,
    y_full: np.ndarray,
    feature_dim: int,
    eval_cfg: EvaluationCnf,
    cv_cfg: CvCnf,
    folds_per_combo: int,
    total_trainings: int,
    completed_trainings: int,
    objective_corpora: str,
    debug: bool,
    print_logs: bool,
    py_logger: logging.Logger,
    clearml_logger: Any,
) -> tuple[CombinationResult, int]:
    """Run k-fold CV for one hyperparameter combination."""
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

    params_json = combo_params_json(
        model_config=model_config,
        training_config=training_config,
    )
    cv_splitter = MultilabelStratifiedKFold(
        n_splits=cv_cfg.folds,
        shuffle=True,
        random_state=cv_cfg.random_seed,
    )
    fold_scores = FoldScores()
    fold_curves: list[CvFoldCurves] = []
    fold_rows: list[dict[str, Any]] = []

    for fold_idx, (fit_indices, val_indices) in enumerate(cv_splitter.split(x_full, y_full), start=1):
        fit_data = slice_dataset(dataset=train_data, indices=fit_indices.tolist())
        val_data = slice_dataset(dataset=train_data, indices=val_indices.tolist())
        micro_row, dev_loss, epochs_run, fold_curve = evaluate_fold(
            fit_data=fit_data,
            val_data=val_data,
            feature_dim=feature_dim,
            model_config=model_config,
            training_config=training_config,
            eval_cfg=eval_cfg,
            objective_corpora=objective_corpora,
            fold_idx=fold_idx,
            print_logs=print_logs,
        )
        completed_trainings += 1
        log_training_progress(
            completed_trainings=completed_trainings,
            total_trainings=total_trainings,
            combo_idx=combo_idx,
            total_combinations=total_combinations,
            fold_idx=fold_idx,
            folds_per_combo=folds_per_combo,
            py_logger=py_logger,
            clearml_logger=clearml_logger,
        )
        fold_scores.loss.append(dev_loss)
        fold_scores.epochs.append(float(epochs_run))
        fold_scores.precision.append(float(micro_row['Precision']))
        fold_scores.recall.append(float(micro_row['Recall']))
        fold_scores.f1.append(float(micro_row['F1']))
        fold_curves.append(fold_curve)
        fold_rows.append(
            {
                'trial_id': combo_idx,
                'fold_id': fold_idx,
                'params': params_json,
                'epochs': float(epochs_run),
                'Loss': dev_loss,
                'Precision': float(micro_row['Precision']),
                'Recall': float(micro_row['Recall']),
                'F1': float(micro_row['F1']),
            }
        )
        if debug:
            break

    trial_row = summarize_combination(
        combo_idx=combo_idx,
        params_json=params_json,
        fold_scores=fold_scores,
    )
    return (
        CombinationResult(
            trial_row=trial_row,
            fold_rows=fold_rows,
            fold_curves=tuple(fold_curves),
            model_config=model_config,
            training_config=training_config,
        ),
        completed_trainings,
    )


def select_best_combination(
    *,
    combinations: Sequence[tuple[ModelCnf, TrainingCnf]],
    train_data: Any,
    x_full: np.ndarray,
    y_full: np.ndarray,
    feature_dim: int,
    eval_cfg: EvaluationCnf,
    cv_cfg: CvCnf,
    objective_corpora: str,
    debug: bool,
    print_logs: bool,
    py_logger: logging.Logger,
    clearml_logger: Any,
) -> BestSelection:
    """Iterate hyperparameter combinations, run CV per combo, and select the best by F1_mean."""
    total_combinations = len(combinations)
    folds_per_combo = 1 if debug else cv_cfg.folds
    total_trainings = total_combinations * folds_per_combo
    log_cv_plan(
        total_combinations=total_combinations,
        folds_per_combo=folds_per_combo,
        total_trainings=total_trainings,
        py_logger=py_logger,
        clearml_logger=clearml_logger,
    )

    completed_trainings = 0
    trial_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    best_trial: dict[str, Any] | None = None
    best_model_cfg: ModelCnf | None = None
    best_training_cfg: TrainingCnf | None = None
    best_fold_curves: tuple[CvFoldCurves, ...] = ()

    for combo_idx, (combo_model_cfg, combo_train_cfg) in enumerate(combinations, start=1):
        combo_result, completed_trainings = run_combination(
            combo_idx=combo_idx,
            total_combinations=total_combinations,
            model_config=combo_model_cfg,
            training_config=combo_train_cfg,
            train_data=train_data,
            x_full=x_full,
            y_full=y_full,
            feature_dim=feature_dim,
            eval_cfg=eval_cfg,
            cv_cfg=cv_cfg,
            folds_per_combo=folds_per_combo,
            total_trainings=total_trainings,
            completed_trainings=completed_trainings,
            objective_corpora=objective_corpora,
            debug=debug,
            print_logs=print_logs,
            py_logger=py_logger,
            clearml_logger=clearml_logger,
        )
        trial_rows.append(combo_result.trial_row)
        fold_rows.extend(combo_result.fold_rows)
        if best_trial is None or combo_result.trial_row['F1_mean'] > best_trial['F1_mean']:
            best_trial = combo_result.trial_row
            best_model_cfg = combo_result.model_config
            best_training_cfg = combo_result.training_config
            best_fold_curves = combo_result.fold_curves

    if best_trial is None or best_model_cfg is None or best_training_cfg is None:
        raise ValueError('No CV trial results were produced.')

    return BestSelection(
        best_trial=best_trial,
        best_model_config=best_model_cfg,
        best_training_config=best_training_cfg,
        best_fold_curves=best_fold_curves,
        trial_rows=trial_rows,
        fold_rows=fold_rows,
    )


def build_cv_dataframes(
    *,
    trial_rows: Sequence[Mapping[str, Any]],
    fold_rows: Sequence[Mapping[str, Any]],
    best_trial: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assemble trials, folds, and best-trial summary dataframes."""
    trials_df = pd.DataFrame(trial_rows).sort_values(by='F1_mean', ascending=False).reset_index(drop=True)
    folds_df = pd.DataFrame(fold_rows)
    cv_dev_df = pd.DataFrame(
        [
            {
                'params': best_trial['params'],
                'epochs': best_trial['epochs'],
                'Precision': best_trial['Precision_mean'],
                'Recall': best_trial['Recall_mean'],
                'F1': best_trial['F1_mean'],
                'Loss': best_trial['Loss_mean'],
                'Precision_std': best_trial['Precision_std'],
                'Recall_std': best_trial['Recall_std'],
                'F1_std': best_trial['F1_std'],
                'Loss_std': best_trial['Loss_std'],
            }
        ],
    )
    return trials_df, folds_df, cv_dev_df


def build_objective_metrics(*, best_trial: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the objective metrics dict from the best trial summary."""
    return {
        'Loss_mean': float(best_trial['Loss_mean']),
        'Loss_std': float(best_trial['Loss_std']),
        'Precision_mean': float(best_trial['Precision_mean']),
        'Precision_std': float(best_trial['Precision_std']),
        'Recall_mean': float(best_trial['Recall_mean']),
        'Recall_std': float(best_trial['Recall_std']),
        'F1_mean': float(best_trial['F1_mean']),
        'F1_std': float(best_trial['F1_std']),
        'epochs': float(best_trial['epochs']),
    }
