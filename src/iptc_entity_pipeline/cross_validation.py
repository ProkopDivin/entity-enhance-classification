"""Cross-validation pipeline step: hyperparameter sweep with k-fold CV."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
import pandas as pd

from iptc_entity_pipeline.config import (
    CvCnf,
    EvaluationCnf,
    HyperparamSpace,
    ModelCnf,
    OptunaCnf,
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

if TYPE_CHECKING:
    from geneea.catlib.vec.dataset import EmbeddingDataset

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CvResult:
    """Outputs of the cross-validation pipeline step.

    Returned by ``run_cv`` as a typed replacement for the previously-used raw dict.
    Config fields are serialized dicts because ClearML transports them across the
    serialization boundary to downstream pipeline components.
    """

    cv_dev_df: pd.DataFrame
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
    pruned: bool = False


@dataclass(frozen=True)
class BestSelection:
    """Best-trial selection across all hyperparameter combinations."""

    best_trial: dict[str, Any]
    best_model_config: ModelCnf
    best_training_config: TrainingCnf
    best_fold_curves: tuple[CvFoldCurves, ...]
    trial_rows: list[dict[str, Any]]
    fold_rows: list[dict[str, Any]]


def prepare_cv(*, train_data: EmbeddingDataset) -> tuple[np.ndarray, np.ndarray]:
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
    clearml_logger: Any,
) -> None:
    """Emit the CV hyperparameter search plan to ClearML logger."""
    cv_plan_message = (
        'CV hyperparameter search plan: '
        f'combinations={total_combinations} '
        f'folds_per_combination={folds_per_combo} '
        f'total_model_trains={total_trainings}'
    )
    clearml_logger.report_text(cv_plan_message, print_console=True)


def log_training_progress(
    *,
    completed_trainings: int,
    total_trainings: int,
    combo_idx: int,
    total_combinations: int,
    fold_idx: int,
    folds_per_combo: int,
    clearml_logger: Any,
) -> None:
    """Emit per-fold training progress to ClearML logger."""
    training_progress_message = (
        'CV training progress: '
        f'trained_models={completed_trainings}/{total_trainings} '
        f'combination={combo_idx}/{total_combinations} '
        f'fold={fold_idx}/{folds_per_combo}'
    )
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
    fit_data: EmbeddingDataset,
    val_data: EmbeddingDataset,
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
        connect_config=False,
    )
    df_corpora_fold, _ = evaluateModel(
        model=train_result.model,
        evalData=val_data,
        evaluation_config=eval_cfg,
        customThresholds=None,
        connect_config=False,
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
    fold_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-fold rows for one combination into a single trial row."""
    df = pd.DataFrame(fold_rows)
    return {
        'trial_id': combo_idx,
        'params': params_json,
        'epochs': float(df['epochs'].mean()),
        'Loss_mean': float(df['Loss'].mean()),
        'Loss_std': float(df['Loss'].std(ddof=0)),
        'F1_mean': float(df['F1'].mean()),
        'F1_std': float(df['F1'].std(ddof=0)),
        'Precision_mean': float(df['Precision'].mean()),
        'Precision_std': float(df['Precision'].std(ddof=0)),
        'Recall_mean': float(df['Recall'].mean()),
        'Recall_std': float(df['Recall'].std(ddof=0)),
    }


def _build_configs_from_trial(
    *,
    trial: Any,
    space: HyperparamSpace,
    base_training: TrainingCnf,
) -> tuple[ModelCnf, TrainingCnf]:
    """Build model/training configs from one Optuna trial suggestion."""
    model_config = ModelCnf(
        hidden_dim=trial.suggest_categorical('hidden_dim', list(space.hidden_dims)),
        dropouts1=trial.suggest_categorical('dropouts1', list(space.dropouts1)),
        dropouts2=trial.suggest_categorical('dropouts2', list(space.dropouts2)),
    )
    training_config = replace(
        base_training,
        batch_size=trial.suggest_categorical('batch_size', list(space.batch_sizes)),
        learning_rate=trial.suggest_categorical('learning_rate', list(space.learning_rates)),
    )
    return model_config, training_config


def _build_search_space(*, space: HyperparamSpace) -> dict[str, list[Any]]:
    """Build Optuna GridSampler search space from HyperparamSpace."""
    return {
        'hidden_dim': list(space.hidden_dims),
        'dropouts1': list(space.dropouts1),
        'dropouts2': list(space.dropouts2),
        'batch_size': list(space.batch_sizes),
        'learning_rate': list(space.learning_rates),
    }


def _count_total_combinations(*, space: HyperparamSpace) -> int:
    """Return number of hyperparameter combinations in the Optuna grid."""
    return (
        len(space.hidden_dims)
        * len(space.dropouts1)
        * len(space.dropouts2)
        * len(space.batch_sizes)
        * len(space.learning_rates)
    )


def _resolve_n_trials(*, optuna_cfg: OptunaCnf, total_combinations: int) -> int:
    """Resolve effective number of trials for Optuna study."""
    sampler_name = optuna_cfg.sampler.strip().lower()
    if optuna_cfg.n_trials > 0:
        if sampler_name == 'grid':
            return min(optuna_cfg.n_trials, total_combinations)
        return optuna_cfg.n_trials
    return total_combinations


def _build_pruner(*, optuna_module: Any, optuna_cfg: OptunaCnf) -> Any:
    """Create Optuna pruner based on configuration."""
    pruner_name = optuna_cfg.pruner.strip().lower()
    if pruner_name == 'none':
        return optuna_module.pruners.NopPruner()
    if pruner_name == 'median':
        return optuna_module.pruners.MedianPruner(
            n_startup_trials=max(0, int(optuna_cfg.startup_trials)),
            n_warmup_steps=max(0, int(optuna_cfg.warmup_steps)),
        )
    if pruner_name in {'successive_halving', 'asha'}:
        return optuna_module.pruners.SuccessiveHalvingPruner(
            min_resource=max(1, int(optuna_cfg.warmup_steps) + 1),
        )
    raise ValueError(f'Unsupported Optuna pruner: {optuna_cfg.pruner}')


def _build_sampler(*, optuna_module: Any, optuna_cfg: OptunaCnf, space: HyperparamSpace) -> Any:
    """Create Optuna sampler based on configuration."""
    sampler_name = optuna_cfg.sampler.strip().lower()
    if sampler_name == 'grid':
        return optuna_module.samplers.GridSampler(search_space=_build_search_space(space=space))
    if sampler_name == 'tpe':
        return optuna_module.samplers.TPESampler(seed=int(optuna_cfg.seed))
    if sampler_name == 'random':
        return optuna_module.samplers.RandomSampler(seed=int(optuna_cfg.seed))
    raise ValueError(f'Unsupported Optuna sampler: {optuna_cfg.sampler}')


def run_combination(
    *,
    combo_idx: int,
    total_combinations: int,
    model_config: ModelCnf,
    training_config: TrainingCnf,
    train_data: EmbeddingDataset,
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
    clearml_logger: Any,
    trial: Any = None,
) -> tuple[CombinationResult, int]:
    """Run k-fold CV for one hyperparameter combination.

    :param trial: Optuna trial for per-fold intermediate reporting and pruning.
    """
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
    fold_curves: list[CvFoldCurves] = []
    fold_rows: list[dict[str, Any]] = []
    pruned = False

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
            clearml_logger=clearml_logger,
        )
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
        if trial is not None:
            running_f1 = sum(r['F1'] for r in fold_rows) / len(fold_rows)
            trial.report(running_f1, step=fold_idx)
            if trial.should_prune():
                LOGGER.info(f'Trial {combo_idx} pruned after fold {fold_idx}/{folds_per_combo}')
                pruned = True
                break
        if debug:
            break

    trial_row = summarize_combination(
        combo_idx=combo_idx,
        params_json=params_json,
        fold_rows=fold_rows,
    )
    return (
        CombinationResult(
            trial_row=trial_row,
            fold_rows=fold_rows,
            fold_curves=tuple(fold_curves),
            model_config=model_config,
            training_config=training_config,
            pruned=pruned,
        ),
        completed_trainings,
    )


def select_best(
    *,
    space: HyperparamSpace,
    base_training: TrainingCnf,
    train_data: EmbeddingDataset,
    x_full: np.ndarray,
    y_full: np.ndarray,
    feature_dim: int,
    eval_cfg: EvaluationCnf,
    cv_cfg: CvCnf,
    optuna_cfg: OptunaCnf,
    objective_corpora: str,
    debug: bool,
    print_logs: bool,
    clearml_logger: Any,
) -> BestSelection:
    """Run Optuna-managed search and select the best trial by mean F1."""
    from importlib import import_module

    optuna = import_module('optuna')

    total_combinations = _count_total_combinations(space=space)
    n_trials = _resolve_n_trials(optuna_cfg=optuna_cfg, total_combinations=total_combinations)
    if debug:
        n_trials = 2
    folds_per_combo = 1 if debug else cv_cfg.folds
    total_trainings = n_trials * folds_per_combo
    log_cv_plan(
        total_combinations=n_trials,
        folds_per_combo=folds_per_combo,
        total_trainings=total_trainings,
        clearml_logger=clearml_logger,
    )
    clearml_logger.report_text(
        (
            'Optuna config: '
            f'sampler={optuna_cfg.sampler} '
            f'pruner={optuna_cfg.pruner} '
            f'direction={optuna_cfg.direction} '
            f'n_trials={n_trials} '
            f'grid_size={total_combinations}'
        ),
        print_console=True,
    )

    completed_trainings = 0
    trial_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    trial_results: dict[int, CombinationResult] = {}

    study = optuna.create_study(
        direction=optuna_cfg.direction,
        sampler=_build_sampler(optuna_module=optuna, optuna_cfg=optuna_cfg, space=space),
        pruner=_build_pruner(optuna_module=optuna, optuna_cfg=optuna_cfg),
    )

    def objective(trial: Any) -> float:
        nonlocal completed_trainings
        combo_model_cfg, combo_train_cfg = _build_configs_from_trial(
            trial=trial,
            space=space,
            base_training=base_training,
        )
        combo_result, completed_after_trial = run_combination(
            combo_idx=trial.number + 1,
            total_combinations=n_trials,
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
            clearml_logger=clearml_logger,
            trial=trial,
        )
        completed_trainings = completed_after_trial
        trial_rows.append(combo_result.trial_row)
        fold_rows.extend(combo_result.fold_rows)
        trial_results[trial.number] = combo_result
        if combo_result.pruned:
            raise optuna.TrialPruned()
        return float(combo_result.trial_row['F1_mean'])

    study.optimize(func=objective, n_trials=n_trials)

    if study.best_trial.number not in trial_results:
        raise ValueError('No CV trial results were produced.')
    best_result = trial_results[study.best_trial.number]

    return BestSelection(
        best_trial=best_result.trial_row,
        best_model_config=best_result.model_config,
        best_training_config=best_result.training_config,
        best_fold_curves=best_result.fold_curves,
        trial_rows=trial_rows,
        fold_rows=fold_rows,
    )


_CV_DEV_RENAME: Mapping[str, str] = {
    'Precision_mean': 'Precision',
    'Recall_mean': 'Recall',
    'F1_mean': 'F1',
    'Loss_mean': 'Loss',
}

_CV_DEV_PASSTHROUGH: tuple[str, ...] = (
    'params', 'epochs',
    'Precision_std', 'Recall_std', 'F1_std', 'Loss_std',
)


def build_cv_df(
    *,
    trial_rows: Sequence[Mapping[str, Any]],
    fold_rows: Sequence[Mapping[str, Any]],
    best_trial: Mapping[str, Any],
    objective_corpora: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assemble trials, folds, and best-trial summary dataframes.

    :param objective_corpora: Used as the cv_dev_df index label so downstream
        consumers (``eval_final``) can look it up by the same key they use for
        the test evaluation table.
    """
    trials_df = pd.DataFrame(trial_rows).sort_values(by='F1_mean', ascending=False).reset_index(drop=True)
    folds_df = pd.DataFrame(fold_rows)
    row: dict[str, Any] = {k: best_trial[k] for k in _CV_DEV_PASSTHROUGH}
    row.update({new: best_trial[old] for old, new in _CV_DEV_RENAME.items()})
    cv_dev_df = pd.DataFrame([row], index=pd.Index([objective_corpora], name='Corpus Name'))
    return trials_df, folds_df, cv_dev_df


_METRIC_KEYS: tuple[str, ...] = (
    'Loss_mean', 'Loss_std',
    'Precision_mean', 'Precision_std',
    'Recall_mean', 'Recall_std',
    'F1_mean', 'F1_std',
    'epochs',
)


def build_cv_result(
    *,
    cv_dev_df: pd.DataFrame,
    selection: BestSelection,
) -> CvResult:
    """Assemble the typed CV result from the best-trial selection."""
    objective_metrics = {k: float(selection.best_trial[k]) for k in _METRIC_KEYS}
    return CvResult(
        cv_dev_df=cv_dev_df,
        best_model_config=asdict(selection.best_model_config),
        best_training_config=asdict(selection.best_training_config),
        objective_metrics=objective_metrics,
    )
