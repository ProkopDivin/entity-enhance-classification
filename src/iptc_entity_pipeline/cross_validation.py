"""Cross-validation pipeline step: hyperparameter sweep with k-fold CV."""

from __future__ import annotations

import gc
import json
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
    ThresholdTuningCnf,
    TrainingCnf,
)
from iptc_entity_pipeline.dataset_builder import (
    build_multilabel_targets,
    slice_dataset,
    to_numpy_array,
)
from iptc_entity_pipeline.evaluate import CLASS_RELEVANT_MACRO_ROW, aggregate_fold_dfs
from iptc_entity_pipeline.legacy_reuse import evaluateModel
from iptc_entity_pipeline.seeding import fold_seed, set_global_seed
from iptc_entity_pipeline.threshold_tuning import (
    ThresholdTuningResult,
    aggregate_fold_thresholds,
    tune_thresholds_dense,
)
from iptc_entity_pipeline.training import (
    CvFoldCurves,
    combo_params_json,
    train_model,
)

if TYPE_CHECKING:
    from geneea.catlib.vec.dataset import EmbeddingDataset

LOGGER = logging.getLogger(__name__)


def _is_better_score(*, candidate: float, best: float, direction: str) -> bool:
    '''Return whether ``candidate`` improves on ``best`` for Optuna direction.'''
    if direction == 'maximize':
        return candidate > best
    if direction == 'minimize':
        return candidate < best
    raise ValueError(f'Unsupported Optuna direction: {direction}')


def _release_training_memory(*, model: Any) -> None:
    '''Drop a trained model and encourage freeing GPU/CPU memory between CV folds.'''
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FoldMetrics:
    """Per-fold scalar metrics for one CV fold.

    Typed replacement for the previously-used raw dict, so typos are caught
    at definition time and IDE autocompletion works.
    """

    trial_id: int
    fold_id: int
    params: str
    epochs: float
    loss: float
    precision_macro_relevant: float
    recall_macro_relevant: float
    f1_macro_relevant: float
    precision_micro: float
    recall_micro: float
    f1_micro: float

    def to_row(self) -> dict[str, Any]:
        '''Serialize to the dict format used by trial/fold DataFrames.'''
        return {
            'trial_id': self.trial_id,
            'fold_id': self.fold_id,
            'params': self.params,
            'epochs': self.epochs,
            'Loss': self.loss,
            'Precision_macro_relevant': self.precision_macro_relevant,
            'Recall_macro_relevant': self.recall_macro_relevant,
            'F1_macro_relevant': self.f1_macro_relevant,
            'Precision_micro': self.precision_micro,
            'Recall_micro': self.recall_micro,
            'F1_micro': self.f1_micro,
        }


@dataclass
class ProgressCounter:
    """Mutable training progress counter shared across Optuna trials."""

    value: int = 0


@dataclass(frozen=True)
class TrialResult:
    """Aggregated outputs for one hyperparameter combination across all folds."""

    trial_row: dict[str, Any]
    fold_metrics: tuple[FoldMetrics, ...]
    fold_curves: tuple[CvFoldCurves, ...]
    fold_thresholds: tuple[dict[str, float], ...]
    fold_corpora_dfs: tuple[pd.DataFrame, ...]
    fold_classes_dfs: tuple[pd.DataFrame, ...]
    model_config: ModelCnf
    training_config: TrainingCnf
    pruned: bool = False


@dataclass(frozen=True)
class SearchOutcome:
    """Best-trial selection plus accumulated rows from all Optuna trials."""

    best: TrialResult
    trial_rows: list[dict[str, Any]]
    fold_rows: list[dict[str, Any]]


@dataclass(frozen=True)
class CvOutputs:
    """Pickle-safe CV results for ClearML pipeline step return values.

    Mirrors the public output attributes populated on :class:`CV` after
    :meth:`CV.fit`, without training data or other non-serializable state.
    """

    best_params: dict[str, Any]
    best_model_config: ModelCnf
    best_training_config: TrainingCnf
    tuned_thresholds: dict[str, float] | None
    threshold_report: pd.DataFrame | None
    trials: pd.DataFrame
    folds: pd.DataFrame
    best_trial_stats: dict[str, Any]
    cv_dev_df: pd.DataFrame
    per_class_df: pd.DataFrame | None
    per_corpora_df: pd.DataFrame | None
    per_class_fold_dfs: tuple[pd.DataFrame, ...] | None
    fold_curves: tuple[CvFoldCurves, ...] | None


@dataclass(frozen=True)
class CvReport:
    """All data needed by reporting.py to render CV results to ClearML.

    Field names on the threshold/per-class/per-corpora slots match the
    attribute names expected by :func:`reporting.report_cv_result_tables`
    so CvReport duck-types as a drop-in ``cv_result`` argument.
    """

    trials_df: pd.DataFrame
    folds_df: pd.DataFrame
    cv_dev_df: pd.DataFrame
    fold_curves: tuple[CvFoldCurves, ...]
    threshold_aggregation: str
    tuned_thresholds: dict[str, float] | None = None
    threshold_report_df: pd.DataFrame | None = None
    cv_per_corpora_df: pd.DataFrame | None = None
    cv_per_class_df: pd.DataFrame | None = None


@dataclass(frozen=True)
class FoldEvalOutput:
    """Per-fold evaluation outputs collected by :meth:`CV._evaluate_fold`.

    :param objective_row: Objective corpora metric row (micro, datapoint, or named corpus).
    :param macro_relevant_row: Relevant-class macro row from classes table.
    :param dev_loss: Final dev loss for the fold.
    :param epochs_run: Number of epochs the fold actually ran (early stop).
    :param fold_curve: Per-epoch train/dev loss + F1 curves.
    :param fold_thresholds: Per-class threshold map when tuning is enabled,
        empty dict otherwise.
    :param df_corpora: Per-corpus evaluation table (at ``eval_thresholds``
        if provided, else at the global ``eval_cfg.threshold_eval``).
    :param df_classes: Per-class evaluation table at the same thresholds.
    """

    objective_row: Mapping[str, Any]
    macro_relevant_row: Mapping[str, Any]
    dev_loss: float
    epochs_run: int
    fold_curve: CvFoldCurves
    fold_thresholds: dict[str, float]
    df_corpora: pd.DataFrame
    df_classes: pd.DataFrame


# ---------------------------------------------------------------------------
# Pure helper functions (no config dependencies)
# ---------------------------------------------------------------------------

def extract_metric_rows(
    *,
    df_corpora_fold: Any,
    df_classes_fold: Any,
    objective_row: str,
    averaging_type: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Pick per-fold objective and relevant-macro metric rows."""
    row = df_corpora_fold.loc[objective_row].to_dict()
    macro_relevant_row = df_classes_fold.loc[CLASS_RELEVANT_MACRO_ROW].to_dict()
    return row, macro_relevant_row


def summarize_combination(
    *,
    combo_idx: int,
    params_json: str,
    fold_metrics: Sequence[FoldMetrics],
) -> dict[str, Any]:
    """Aggregate per-fold metrics for one combination into a single trial row."""
    df = pd.DataFrame([fm.to_row() for fm in fold_metrics])
    return {
        'trial_id': combo_idx,
        'params': params_json,
        'epochs': float(df['epochs'].mean()),
        'Loss_mean': float(df['Loss'].mean()),
        'Loss_std': float(df['Loss'].std(ddof=0)),
        'F1_macro_relevant_mean': float(df['F1_macro_relevant'].mean()),
        'F1_macro_relevant_std': float(df['F1_macro_relevant'].std(ddof=0)),
        'Precision_macro_relevant_mean': float(df['Precision_macro_relevant'].mean()),
        'Precision_macro_relevant_std': float(df['Precision_macro_relevant'].std(ddof=0)),
        'Recall_macro_relevant_mean': float(df['Recall_macro_relevant'].mean()),
        'Recall_macro_relevant_std': float(df['Recall_macro_relevant'].std(ddof=0)),
        'F1_micro_mean': float(df['F1_micro'].mean()),
        'F1_micro_std': float(df['F1_micro'].std(ddof=0)),
        'Precision_micro_mean': float(df['Precision_micro'].mean()),
        'Precision_micro_std': float(df['Precision_micro'].std(ddof=0)),
        'Recall_micro_mean': float(df['Recall_micro'].mean()),
        'Recall_micro_std': float(df['Recall_micro'].std(ddof=0)),
    }


def _mean_eval_tables(*, first_df: pd.DataFrame, second_df: pd.DataFrame) -> pd.DataFrame:
    """Average two eval tables and align rows/columns when subsets differ."""
    row_index = first_df.index.union(second_df.index, sort=False)
    col_index = first_df.columns.union(second_df.columns, sort=False)
    first_aligned = first_df.reindex(index=row_index, columns=col_index)
    second_aligned = second_df.reindex(index=row_index, columns=col_index)

    out_df = first_aligned.combine_first(second_aligned)
    numeric_cols_first = set(first_df.select_dtypes(include='number').columns)
    numeric_cols_second = set(second_df.select_dtypes(include='number').columns)
    numeric_cols = [col for col in col_index if col in numeric_cols_first or col in numeric_cols_second]
    for col in numeric_cols:
        first_num = pd.to_numeric(first_aligned[col], errors='coerce')
        second_num = pd.to_numeric(second_aligned[col], errors='coerce')
        out_df[col] = pd.concat([first_num, second_num], axis=1).mean(axis=1, skipna=True)
    return out_df


def _subset_predictions(*, pred_scores: Any, indices: Sequence[int]) -> Any:
    """Select prediction rows by positional indices.

    Supports both the dense ``np.ndarray`` produced by the current
    matrix-based evaluation path and the legacy ``list[list[...]]``
    representation (kept for any out-of-tree callers).
    """
    if isinstance(pred_scores, np.ndarray):
        return pred_scores[np.asarray(indices, dtype=np.int64)]
    return [pred_scores[int(idx)] for idx in indices]


def _build_cv_dev_row(best_trial: Mapping[str, Any]) -> dict[str, Any]:
    """Build CV summary metrics from a best-trial row.

    Keys match :data:`iptc_entity_pipeline.reporting.METRIC_SERIES` and
    :data:`iptc_entity_pipeline.reporting.STD_METRIC_SERIES` so
    :func:`~iptc_entity_pipeline.reporting.report_eval` can log scalars
    without renaming.
    """
    return {
        'params': best_trial['params'],
        'epochs': float(best_trial['epochs']),
        'Loss': float(best_trial['Loss_mean']),
        'Loss_std': float(best_trial['Loss_std']),
        'Precision_micro': float(best_trial['Precision_micro_mean']),
        'Precision_micro_std': float(best_trial['Precision_micro_std']),
        'Recall_micro': float(best_trial['Recall_micro_mean']),
        'Recall_micro_std': float(best_trial['Recall_micro_std']),
        'F1_micro': float(best_trial['F1_micro_mean']),
        'F1_micro_std': float(best_trial['F1_micro_std']),
        'Precision_macro_relevant': float(best_trial['Precision_macro_relevant_mean']),
        'Precision_macro_relevant_std': float(best_trial['Precision_macro_relevant_std']),
        'Recall_macro_relevant': float(best_trial['Recall_macro_relevant_mean']),
        'Recall_macro_relevant_std': float(best_trial['Recall_macro_relevant_std']),
        'F1_macro_relevant': float(best_trial['F1_macro_relevant_mean']),
        'F1_macro_relevant_std': float(best_trial['F1_macro_relevant_std']),
    }


# ---------------------------------------------------------------------------
# CV orchestrator
# ---------------------------------------------------------------------------

class CV:
    """Cross-validation orchestrator.

    Encapsulates config, Optuna hyperparameter search, and k-fold CV
    results.  Initialize with typed configs, call :meth:`fit` to run the
    search, then access outputs via public attributes or build a
    :class:`CvReport` for ClearML reporting via :meth:`prepare_report`.
    """

    def __init__(
        self,
        *,
        model_cnf: ModelCnf,
        hparam_cnf: HyperparamSpace,
        train_cnf: TrainingCnf,
        eval_cnf: EvaluationCnf,
        cv_cnf: CvCnf,
        optuna_cnf: OptunaCnf,
        tuning_cnf: ThresholdTuningCnf,
        objective_row: str,
        random_seed: int,
        debug: bool = False,
        eval_thresholds: Mapping[str, float] | None = None,
    ) -> None:
        self._model_cnf = model_cnf
        self._hparam_cnf = hparam_cnf
        self._train_cnf = train_cnf
        self._eval_cnf = eval_cnf
        self._cv_cnf = cv_cnf
        self._optuna_cnf = optuna_cnf
        self._objective_row = objective_row
        self._random_seed = random_seed
        self._debug = debug
        self._eval_thresholds = eval_thresholds

        if eval_thresholds is not None and tuning_cnf.enabled:
            LOGGER.info(
                'Assembly mode: external eval_thresholds provided, '
                'forcing tuning_cnf.enabled=False'
            )
            self._tuning_cnf = replace(tuning_cnf, enabled=False)
        else:
            self._tuning_cnf = tuning_cnf
        self._selection_metric = str(self._tuning_cnf.selection_metric)
        if self._selection_metric not in {'F1_micro', 'F1_macro_relevant'}:
            raise ValueError(
                'Unsupported cv_hparam_selection_metric: '
                f'{self._selection_metric}'
            )

        # Fit-time state (populated by fit(), used across private methods)
        self._train_data: EmbeddingDataset | None = None
        self._y_full: np.ndarray | None = None
        self._feature_dim: int = 0
        self._print_logs: bool = True
        self._clearml_logger: Any = None
        self._progress: ProgressCounter = ProgressCounter()
        self._n_trials: int = 0
        self._folds_per_combo: int = 0
        self._total_trainings: int = 0

        # Output attributes (populated by _finalize())
        self.best_params: dict[str, Any] | None = None
        self.best_model_config: ModelCnf | None = None
        self.best_training_config: TrainingCnf | None = None
        self.tuned_thresholds: dict[str, float] | None = None
        self.threshold_report: pd.DataFrame | None = None
        self.trials: pd.DataFrame | None = None
        self.folds: pd.DataFrame | None = None
        self.best_trial_stats: dict[str, Any] | None = None
        self.per_class_df: pd.DataFrame | None = None
        self.per_corpora_df: pd.DataFrame | None = None
        self.per_class_fold_dfs: tuple[pd.DataFrame, ...] | None = None
        self.fold_curves: tuple[CvFoldCurves, ...] | None = None
        self.cv_dev_df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        *,
        train_data: EmbeddingDataset,
        feature_dim: int,
        print_logs: bool = True,
        clearml_logger: Any,
    ) -> CV:
        """Run cross-validation and populate output attributes.

        :param train_data: full training dataset to split into folds
        :param feature_dim: feature dimensionality
        :param print_logs: forwarded to per-fold training
        :param clearml_logger: ClearML logger for progress reporting
        :return: self for chaining
        """
        self._train_data = train_data
        self._feature_dim = feature_dim
        self._print_logs = print_logs
        self._clearml_logger = clearml_logger
        self._y_full = self._prepare_labels()
        LOGGER.info(
            f'CV data prepared: n_samples={self._y_full.shape[0]}, '
            f'y_shape={self._y_full.shape}, feature_dim={feature_dim}'
        )

        outcome = self._select_best()
        selected_metric_col = f'{self._selection_metric}_mean'
        selected_score = float(outcome.best.trial_row[selected_metric_col])
        LOGGER.info(
            'CV complete: '
            f'selected_metric={self._selection_metric}, '
            f'selected_score={selected_score:.4f}, '
            f'F1_micro_mean={outcome.best.trial_row["F1_micro_mean"]:.4f}, '
            f'F1_macro_relevant_mean={outcome.best.trial_row["F1_macro_relevant_mean"]:.4f}'
        )

        self._finalize(outcome=outcome)
        return self

    def export_outputs(self) -> CvOutputs:
        """Return pickle-safe outputs for ClearML pipeline transport."""
        if self.trials is None or self.best_params is None:
            raise RuntimeError('CV not fitted yet; call fit() first')
        if self.best_model_config is None or self.best_training_config is None:
            raise RuntimeError('CV best configs missing after fit()')
        if self.cv_dev_df is None or self.best_trial_stats is None:
            raise RuntimeError('CV summary outputs missing after fit()')
        return CvOutputs(
            best_params=self.best_params,
            best_model_config=self.best_model_config,
            best_training_config=self.best_training_config,
            tuned_thresholds=self.tuned_thresholds,
            threshold_report=self.threshold_report,
            trials=self.trials,
            folds=self.folds,
            best_trial_stats=self.best_trial_stats,
            cv_dev_df=self.cv_dev_df,
            per_class_df=self.per_class_df,
            per_corpora_df=self.per_corpora_df,
            per_class_fold_dfs=self.per_class_fold_dfs,
            fold_curves=self.fold_curves,
        )

    def prepare_report(self) -> CvReport:
        """Build a report object for ClearML reporting.

        :return: CvReport with all data needed by reporting functions
        """
        if self.trials is None:
            raise RuntimeError('CV not fitted yet; call fit() first')
        return CvReport(
            trials_df=self.trials,
            folds_df=self.folds,
            cv_dev_df=self.cv_dev_df,
            fold_curves=self.fold_curves,
            threshold_aggregation=self._tuning_cnf.aggregation,
            tuned_thresholds=self.tuned_thresholds,
            threshold_report_df=self.threshold_report,
            cv_per_corpora_df=self.per_corpora_df,
            cv_per_class_df=self.per_class_df,
        )

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _prepare_labels(self) -> np.ndarray:
        """Extract label matrix for stratified CV splits (no embedding copy)."""
        if hasattr(self._train_data, 'Y'):
            return to_numpy_array(matrix_like=self._train_data.Y)
        return build_multilabel_targets(corpus=self._train_data.corpus)

    # ------------------------------------------------------------------
    # Optuna infrastructure
    # ------------------------------------------------------------------

    def _count_combinations(self) -> int:
        """Return number of hyperparameter combinations in the grid."""
        s = self._hparam_cnf
        return (
            len(s.hidden_dims)
            * len(s.dropouts1)
            * len(s.dropouts2)
            * len(s.attention_hidden_dims)
            * len(s.attention_dropouts)
            * len(s.batch_sizes)
            * len(s.learning_rates)
        )

    def _resolve_n_trials(self, *, total_combinations: int) -> int:
        """Resolve effective number of trials for Optuna study."""
        sampler_name = self._optuna_cnf.sampler.strip().lower()
        if self._optuna_cnf.n_trials > 0:
            if sampler_name == 'grid':
                return min(self._optuna_cnf.n_trials, total_combinations)
            return self._optuna_cnf.n_trials
        return total_combinations

    def _build_search_space(self) -> dict[str, list[Any]]:
        """Build Optuna GridSampler search space from HyperparamSpace."""
        s = self._hparam_cnf
        return {
            'hidden_dim': list(s.hidden_dims),
            'dropouts1': list(s.dropouts1),
            'dropouts2': list(s.dropouts2),
            'attention_hidden_dim': list(s.attention_hidden_dims),
            'attention_dropout': list(s.attention_dropouts),
            'batch_size': list(s.batch_sizes),
            'learning_rate': list(s.learning_rates),
        }

    def _build_sampler(self, *, optuna_module: Any) -> Any:
        """Create Optuna sampler based on configuration."""
        sampler_name = self._optuna_cnf.sampler.strip().lower()
        if sampler_name == 'grid':
            return optuna_module.samplers.GridSampler(search_space=self._build_search_space())
        if sampler_name == 'tpe':
            return optuna_module.samplers.TPESampler(seed=int(self._random_seed))
        if sampler_name == 'random':
            return optuna_module.samplers.RandomSampler(seed=int(self._random_seed))
        raise ValueError(f'Unsupported Optuna sampler: {self._optuna_cnf.sampler}')

    def _build_pruner(self, *, optuna_module: Any) -> Any:
        """Create Optuna pruner based on configuration."""
        pruner_name = self._optuna_cnf.pruner.strip().lower()
        if pruner_name == 'none':
            return optuna_module.pruners.NopPruner()
        if pruner_name == 'median':
            return optuna_module.pruners.MedianPruner(
                n_startup_trials=max(0, int(self._optuna_cnf.startup_trials)),
                n_warmup_steps=max(0, int(self._optuna_cnf.warmup_steps)),
            )
        if pruner_name in {'successive_halving', 'asha'}:
            return optuna_module.pruners.SuccessiveHalvingPruner(
                min_resource=max(1, int(self._optuna_cnf.warmup_steps) + 1),
            )
        raise ValueError(f'Unsupported Optuna pruner: {self._optuna_cnf.pruner}')

    def _build_configs_from_trial(self, *, trial: Any) -> tuple[ModelCnf, TrainingCnf]:
        """Build model/training configs from one Optuna trial suggestion."""
        s = self._hparam_cnf
        model_config = replace(
            self._model_cnf,
            hidden_dim=trial.suggest_categorical('hidden_dim', list(s.hidden_dims)),
            dropouts1=trial.suggest_categorical('dropouts1', list(s.dropouts1)),
            dropouts2=trial.suggest_categorical('dropouts2', list(s.dropouts2)),
            attention_hidden_dim=trial.suggest_categorical(
                'attention_hidden_dim', list(s.attention_hidden_dims),
            ),
            attention_dropout=trial.suggest_categorical(
                'attention_dropout', list(s.attention_dropouts),
            ),
        )
        training_config = replace(
            self._train_cnf,
            batch_size=trial.suggest_categorical('batch_size', list(s.batch_sizes)),
            learning_rate=trial.suggest_categorical('learning_rate', list(s.learning_rates)),
        )
        return model_config, training_config

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_cv_plan(self) -> None:
        """Emit the CV hyperparameter search plan to ClearML logger."""
        msg = (
            'CV hyperparameter search plan: '
            f'combinations={self._n_trials} '
            f'folds_per_combination={self._folds_per_combo} '
            f'total_model_trains={self._total_trainings}'
        )
        self._clearml_logger.report_text(msg, print_console=True)

    def _log_progress(self, *, combo_idx: int, fold_idx: int) -> None:
        """Emit per-fold training progress to ClearML logger."""
        msg = (
            'CV training progress: '
            f'trained_models={self._progress.value}/{self._total_trainings} '
            f'combination={combo_idx}/{self._n_trials} '
            f'fold={fold_idx}/{self._folds_per_combo}'
        )
        self._clearml_logger.report_text(msg, print_console=True)


    def _fold_metric_for_selection(self, *, fold_metric: FoldMetrics) -> float:
        """Return per-fold metric value used for running Optuna pruning reports."""
        if self._selection_metric == 'F1_micro':
            return float(fold_metric.f1_micro)
        if self._selection_metric == 'F1_macro_relevant':
            return float(fold_metric.f1_macro_relevant)
        raise ValueError(f'Unsupported selection_metric: {self._selection_metric}')

    # ------------------------------------------------------------------
    # Fold evaluation
    # ------------------------------------------------------------------

    def _split_oof_indices(
        self,
        *,
        val_data: EmbeddingDataset,
        fold_idx: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split one fold's OOF validation rows into two stratified halves."""
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

        y_val = (
            to_numpy_array(matrix_like=val_data.Y)
            if hasattr(val_data, 'Y')
            else build_multilabel_targets(corpus=val_data.corpus)
        )
        splitter = MultilabelStratifiedKFold(
            n_splits=2,
            shuffle=True,
            random_state=fold_seed(base_seed=self._random_seed, fold_idx=fold_idx + 1000),
        )
        x_dummy = np.zeros((int(y_val.shape[0]), 1), dtype=np.float32)
        idx_a, idx_b = next(splitter.split(x_dummy, y_val))
        return np.asarray(idx_a, dtype=np.int64), np.asarray(idx_b, dtype=np.int64)
        
    def _evaluate_fold(
        self,
        *,
        fit_data: EmbeddingDataset,
        val_data: EmbeddingDataset,
        model_config: ModelCnf,
        training_config: TrainingCnf,
        fold_idx: int,
    ) -> FoldEvalOutput:
        """Train and evaluate one CV fold.

        Uses ``self._feature_dim``, ``self._print_logs``, ``self._eval_cnf``,
        ``self._tuning_cnf``, ``self._objective_row``, ``self._random_seed``,
        and ``self._eval_thresholds`` from the instance.
        """
        set_global_seed(seed=fold_seed(base_seed=self._random_seed, fold_idx=fold_idx))
        train_result = train_model(
            train_data=fit_data,
            dev_data=val_data,
            feature_dim=self._feature_dim,
            model_config=model_config,
            training_config=training_config,
            print_logs=self._print_logs,
            connect_config=False,
        )
        model = train_result.model

        custom_thresholds = dict(self._eval_thresholds) if self._eval_thresholds else None
        df_corpora_fold, df_classes_fold, pred_scores = evaluateModel(
            model=model,
            evalData=val_data,
            evaluation_config=self._eval_cnf,
            customThresholds=custom_thresholds,
            connect_config=False,
            returnPredictions=True,
        )

        if self._tuning_cnf.enabled:
            # it is this complicated to avoid data leakege - do not want to tune on the same data as we evaluate on
            if custom_thresholds:
                raise ValueError('custom tresholds would be overridden by threshold tuning, do not provide custom thresholds or disable threshold tuning')
            cat_list = list(model.catList)
            idx_a, idx_b = self._split_oof_indices(val_data=val_data, fold_idx=fold_idx)
            val_data_a = slice_dataset(dataset=val_data, indices=idx_a.tolist())
            val_data_b = slice_dataset(dataset=val_data, indices=idx_b.tolist())

            pred_scores_a = _subset_predictions(pred_scores=pred_scores, indices=idx_a.tolist())
            pred_scores_b = _subset_predictions(pred_scores=pred_scores, indices=idx_b.tolist())

            thresholds_a = tune_thresholds_dense(
                score_matrix=pred_scores_a,
                cat_list=cat_list,
                eval_corpus=val_data_a.corpus,
                tuning_cfg=self._tuning_cnf,
            )
            del pred_scores_a
            thresholds_b = tune_thresholds_dense(
                score_matrix=pred_scores_b,
                cat_list=cat_list,
                eval_corpus=val_data_b.corpus,
                tuning_cfg=self._tuning_cnf,
            )
            del pred_scores_b

            df_corpora_b, df_classes_b = evaluateModel(
                model=model,
                evalData=val_data_b,
                evaluation_config=self._eval_cnf,
                customThresholds=thresholds_a,
                connect_config=False,
                returnPredictions=False,
            )
            df_corpora_a, df_classes_a = evaluateModel(
                model=model,
                evalData=val_data_a,
                evaluation_config=self._eval_cnf,
                customThresholds=thresholds_b,
                connect_config=False,
                returnPredictions=False,
            )
            df_corpora_fold = _mean_eval_tables(first_df=df_corpora_a, second_df=df_corpora_b)
            df_classes_fold = _mean_eval_tables(first_df=df_classes_a, second_df=df_classes_b)

            # retune the tresholds - more data for tuning = more stability
            fold_thresholds = tune_thresholds_dense(
                score_matrix=pred_scores,
                cat_list=cat_list,
                eval_corpus=val_data.corpus,
                tuning_cfg=self._tuning_cnf,
            )
        else:
            fold_thresholds = {}

        objective_row, macro_relevant_row = extract_metric_rows(
            df_corpora_fold=df_corpora_fold,
            df_classes_fold=df_classes_fold,
            objective_row=self._objective_row,
            averaging_type=self._eval_cnf.averaging_type,
        )
        fold_curve = CvFoldCurves(
            fold_id=fold_idx,
            train_loss_per_epoch=train_result.train_loss_per_epoch,
            dev_loss_per_epoch=train_result.dev_loss_per_epoch,
            train_f1_micro_per_epoch=train_result.train_f1_micro_per_epoch,
            dev_f1_micro_per_epoch=train_result.dev_f1_micro_per_epoch,
            train_f1_macro_relevant_per_epoch=train_result.train_f1_macro_relevant_per_epoch,
            dev_f1_macro_relevant_per_epoch=train_result.dev_f1_macro_relevant_per_epoch,
        )
        _release_training_memory(model=model)
        return FoldEvalOutput(
            objective_row=objective_row,
            macro_relevant_row=macro_relevant_row,
            dev_loss=float(train_result.final_dev_loss),
            epochs_run=int(train_result.epochs_run),
            fold_curve=fold_curve,
            fold_thresholds=fold_thresholds,
            df_corpora=df_corpora_fold,
            df_classes=df_classes_fold,
        )

    # ------------------------------------------------------------------
    # Combination (k-fold for one param set)
    # ------------------------------------------------------------------

    def _run_combination(
        self,
        *,
        combo_idx: int,
        model_config: ModelCnf,
        training_config: TrainingCnf,
        trial: Any = None,
    ) -> TrialResult:
        """Run k-fold CV for one hyperparameter combination.

        :param trial: Optuna trial for per-fold intermediate reporting and pruning.
        """
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

        params_json = combo_params_json(
            model_config=model_config,
            training_config=training_config,
        )
        cv_splitter = MultilabelStratifiedKFold(
            n_splits=self._cv_cnf.folds,
            shuffle=True,
            random_state=int(self._random_seed),
        )
        fold_curves: list[CvFoldCurves] = []
        fold_metrics_list: list[FoldMetrics] = []
        fold_thresholds: list[dict[str, float]] = []
        fold_corpora_dfs: list[pd.DataFrame] = []
        fold_classes_dfs: list[pd.DataFrame] = []
        pruned = False
        x_dummy = np.zeros((int(self._y_full.shape[0]), 1), dtype=np.float32)

        for fold_idx, (fit_indices, val_indices) in enumerate(
            cv_splitter.split(x_dummy, self._y_full), start=1,
        ):
            fit_data = slice_dataset(dataset=self._train_data, indices=fit_indices.tolist())
            val_data = slice_dataset(dataset=self._train_data, indices=val_indices.tolist())
            fold_out = self._evaluate_fold(
                fit_data=fit_data,
                val_data=val_data,
                model_config=model_config,
                training_config=training_config,
                fold_idx=fold_idx,
            )
            self._progress.value += 1
            self._log_progress(combo_idx=combo_idx, fold_idx=fold_idx)
            fold_curves.append(fold_out.fold_curve)
            fold_thresholds.append(fold_out.fold_thresholds)
            fold_corpora_dfs.append(fold_out.df_corpora)
            fold_classes_dfs.append(fold_out.df_classes)
            fm = FoldMetrics(
                trial_id=combo_idx,
                fold_id=fold_idx,
                params=params_json,
                epochs=float(fold_out.epochs_run),
                loss=fold_out.dev_loss,
                precision_macro_relevant=float(fold_out.macro_relevant_row['Precision']),
                recall_macro_relevant=float(fold_out.macro_relevant_row['Recall']),
                f1_macro_relevant=float(fold_out.macro_relevant_row['F1']),
                precision_micro=float(fold_out.objective_row['Precision']),
                recall_micro=float(fold_out.objective_row['Recall']),
                f1_micro=float(fold_out.objective_row['F1']),
            )
            fold_metrics_list.append(fm)
            if trial is not None:
                running_score = (
                    sum(
                        self._fold_metric_for_selection(fold_metric=m)
                        for m in fold_metrics_list
                    ) / len(fold_metrics_list)
                )
                trial.report(running_score, step=fold_idx)
                if trial.should_prune():
                    LOGGER.info(f'Trial {combo_idx} pruned after fold {fold_idx}/{self._folds_per_combo}')
                    pruned = True
                    break
            if self._debug:
                break

        trial_row = summarize_combination(
            combo_idx=combo_idx,
            params_json=params_json,
            fold_metrics=fold_metrics_list,
        )
        return TrialResult(
            trial_row=trial_row,
            fold_metrics=tuple(fold_metrics_list),
            fold_curves=tuple(fold_curves),
            fold_thresholds=tuple(fold_thresholds),
            fold_corpora_dfs=tuple(fold_corpora_dfs),
            fold_classes_dfs=tuple(fold_classes_dfs),
            model_config=model_config,
            training_config=training_config,
            pruned=pruned,
        )

    # ------------------------------------------------------------------
    # Optuna search
    # ------------------------------------------------------------------

    def _select_best(self) -> SearchOutcome:
        """Run Optuna-managed search and select the best trial by configured CV metric."""
        from importlib import import_module
        optuna = import_module('optuna')

        total_combinations = self._count_combinations()
        self._n_trials = self._resolve_n_trials(total_combinations=total_combinations)
        if self._debug:
            self._n_trials = 2
        self._folds_per_combo = 1 if self._debug else self._cv_cnf.folds
        self._total_trainings = self._n_trials * self._folds_per_combo
        self._log_cv_plan()
        self._clearml_logger.report_text(
            (
                'Optuna config: '
                f'sampler={self._optuna_cnf.sampler} '
                f'pruner={self._optuna_cnf.pruner} '
                f'direction={self._optuna_cnf.direction} '
                f'selection_metric={self._selection_metric} '
                f'n_trials={self._n_trials} '
                f'grid_size={total_combinations}'
            ),
            print_console=True,
        )

        self._progress = ProgressCounter()
        trial_rows: list[dict[str, Any]] = []
        fold_rows: list[dict[str, Any]] = []
        best_trial_result: TrialResult | None = None
        best_trial_score: float | None = None
        direction = self._optuna_cnf.direction

        study = optuna.create_study(
            direction=direction,
            sampler=self._build_sampler(optuna_module=optuna),
            pruner=self._build_pruner(optuna_module=optuna),
        )

        def objective(trial: Any) -> float:
            nonlocal best_trial_result, best_trial_score
            combo_model_cfg, combo_train_cfg = self._build_configs_from_trial(trial=trial)
            combo_result = self._run_combination(
                combo_idx=trial.number + 1,
                model_config=combo_model_cfg,
                training_config=combo_train_cfg,
                trial=trial,
            )
            trial_rows.append(combo_result.trial_row)
            fold_rows.extend(fm.to_row() for fm in combo_result.fold_metrics)
            objective_value = float(combo_result.trial_row[f'{self._selection_metric}_mean'])
            if (
                best_trial_result is None
                or _is_better_score(candidate=objective_value, best=best_trial_score, direction=direction)
            ):
                best_trial_result = combo_result
                best_trial_score = objective_value
            if combo_result.pruned:
                raise optuna.TrialPruned()
            return objective_value

        study.optimize(func=objective, n_trials=self._n_trials)

        if best_trial_result is None:
            raise ValueError('No CV trial results were produced.')
        if study.best_trial.number + 1 != best_trial_result.trial_row['trial_id']:
            LOGGER.warning(
                'CV best-trial id mismatch between Optuna and retained heavy result: '
                f'optuna_trial={study.best_trial.number + 1}, '
                f'retained_trial_id={best_trial_result.trial_row["trial_id"]}'
            )

        return SearchOutcome(
            best=best_trial_result,
            trial_rows=trial_rows,
            fold_rows=fold_rows,
        )

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def _finalize(self, *, outcome: SearchOutcome) -> None:
        """Populate output attributes from completed search."""
        best = outcome.best
        best_trial = best.trial_row

        self.trials = (
            pd.DataFrame(outcome.trial_rows)
            .sort_values(by=f'{self._selection_metric}_mean', ascending=False)
            .reset_index(drop=True)
        )
        self.folds = pd.DataFrame(outcome.fold_rows)

        self.best_params = json.loads(best_trial['params'])
        self.best_model_config = best.model_config

        fixed_epochs = max(
            1,
            int(round(float(best_trial['epochs']))) - self._train_cnf.early_stopping_patience,
        )
        self.best_training_config = replace(
            best.training_config,
            epochs=fixed_epochs,
            early_stopping_patience=0,
        )

        cv_dev_row = _build_cv_dev_row(best_trial)
        self.best_trial_stats = cv_dev_row
        self.cv_dev_df = pd.DataFrame(
            [cv_dev_row],
            index=pd.Index([self._objective_row], name='objective_row'),
        )

        if self._eval_thresholds is not None:
            self.tuned_thresholds = {str(k): float(v) for k, v in self._eval_thresholds.items()}
        elif self._tuning_cnf.enabled and best.fold_thresholds:
            cat_list = list(self._train_data.corpus.catList)
            tuning_result: ThresholdTuningResult = aggregate_fold_thresholds(
                fold_thresholds=best.fold_thresholds,
                cat_list=cat_list,
                default_threshold=self._eval_cnf.threshold_eval,
                aggregation=self._tuning_cnf.aggregation,
                min_folds_for_tuning=self._tuning_cnf.min_folds_for_tuning,
            )
            self.tuned_thresholds = tuning_result.cat_to_threshold
            self.threshold_report = tuning_result.report_df

        self.per_corpora_df = (
            aggregate_fold_dfs(fold_dfs=best.fold_corpora_dfs)
            if best.fold_corpora_dfs else None
        )
        self.per_class_df = (
            aggregate_fold_dfs(fold_dfs=best.fold_classes_dfs)
            if best.fold_classes_dfs else None
        )
        self.per_class_fold_dfs = (
            tuple(best.fold_classes_dfs) if best.fold_classes_dfs else None
        )
        self.fold_curves = best.fold_curves
