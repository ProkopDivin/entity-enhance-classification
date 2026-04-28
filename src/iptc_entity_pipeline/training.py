"""Model creation, training loop wrapper, and hyperparameter display helpers."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from iptc_entity_pipeline.config import ModelCnf, TrainingCnf
from iptc_entity_pipeline.dataset_builder import to_numpy_array
from iptc_entity_pipeline.legacy_reuse import createClassificationModel, trainClassificationModel

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingResult:
    """Outputs from one classification training run."""

    model: Any
    final_dev_loss: float
    epochs_run: int
    train_precision_per_epoch: tuple[float, ...]
    dev_precision_per_epoch: tuple[float, ...]
    train_recall_per_epoch: tuple[float, ...]
    dev_recall_per_epoch: tuple[float, ...]
    train_loss_per_epoch: tuple[float, ...]
    dev_loss_per_epoch: tuple[float, ...]
    train_f1_per_epoch: tuple[float, ...]
    dev_f1_per_epoch: tuple[float, ...]


@dataclass(frozen=True)
class CvFoldCurves:
    """Per-epoch train/dev curves for one CV fold."""

    fold_id: int
    train_loss_per_epoch: tuple[float, ...]
    dev_loss_per_epoch: tuple[float, ...]
    train_f1_per_epoch: tuple[float, ...]
    dev_f1_per_epoch: tuple[float, ...]


def combo_params_json(*, model_config: ModelCnf, training_config: TrainingCnf) -> str:
    """Serialize selected hyperparameters as JSON for tabular display."""
    payload = {
        'hidden_dim': model_config.hidden_dim,
        'batch_size': training_config.batch_size,
        'dropouts1': model_config.dropouts1,
        'dropouts2': model_config.dropouts2,
        'learning_rate': training_config.learning_rate,
    }
    return json.dumps(payload, sort_keys=True)


def model_payload(*, model_config: ModelCnf) -> dict[str, Any]:
    """Build model payload expected by legacy model factory."""
    return {
        'hiddenDim': model_config.hidden_dim,
        'dropouts1': model_config.dropouts1,
        'dropouts2': model_config.dropouts2,
    }


def train_payload(*, training_config: TrainingCnf) -> dict[str, Any]:
    """Build training payload expected by legacy training entrypoint."""
    return {
        'epochs': training_config.epochs,
        'batchSize': training_config.batch_size,
        'validationSplitName': 'dev',
        'earlyStoppingPatience': training_config.early_stopping_patience,
        'earlyStoppingMinDelta': training_config.early_stopping_min_delta,
        'optimizerConfig': {
            'name': training_config.optimizer_name,
            'adamConfig': {'lr': training_config.learning_rate},
        },
        'lrSchedulerConfig': {
            'name': training_config.lr_scheduler_name,
            'stepLRConfig': {
                'stepSize': training_config.step_size,
                'gamma': training_config.gamma,
            },
            'cosineAnnealingLRConfig': {'T_max': max(training_config.epochs, 1)},
        },
        'lossConfig': {
            'name': training_config.loss_name,
            'focalLossConfig': {'alpha': 0.25, 'gamma': 2.0},
        },
    }


def train_model(
    *,
    train_data: Any,
    dev_data: Any,
    feature_dim: int,
    model_config: ModelCnf,
    training_config: TrainingCnf,
    print_logs: bool = True,
    connect_config: bool = True,
) -> TrainingResult:
    """
    Create and train the classification model on given fit/dev datasets.

    :param train_data: Training ``EmbeddingDataset``.
    :param dev_data: Dev/validation ``EmbeddingDataset``.
    :param feature_dim: Input embedding dimensionality.
    :param model_config: Scalar model hyperparameters.
    :param training_config: Training loop parameters.
    :param print_logs: Whether to print ClearML log messages to console.
    :param connect_config: Register config with ClearML task (disable during CV to avoid flooding).
    :return: Training result with model, loss, and per-epoch curves.
    """
    def calc_f1_curve(*, precision_curve: Sequence[float], recall_curve: Sequence[float]) -> tuple[float, ...]:
        f1_values: list[float] = []
        for precision, recall in zip(precision_curve, recall_curve):
            denom = precision + recall
            f1_values.append(0.0 if denom == 0.0 else float((2.0 * precision * recall) / denom))
        return tuple(f1_values)

    log_config = {'PRINT_LOGS': print_logs}

    if hasattr(train_data, 'Y'):
        out_dim = int(to_numpy_array(matrix_like=train_data.Y).shape[1])
    else:
        out_dim = int(train_data.corpus.catCnt)
    model = createClassificationModel(
        embDim=int(feature_dim),
        outDim=out_dim,
        modelConfig=model_payload(model_config=model_config),
        textVectorizer='not None',
        logConfig=log_config,
        connect_config=connect_config,
    )
    (
        model,
        final_dev_loss,
        epochs_run,
        train_precisions,
        train_recalls,
        train_losses,
        dev_precisions,
        dev_recalls,
        dev_losses,
    ) = trainClassificationModel(
        model=model,
        trainData=train_data,
        devData=dev_data,
        trainingConfig=train_payload(training_config=training_config),
        logConfig=log_config,
        connect_config=connect_config,
    )
    return TrainingResult(
        model=model,
        final_dev_loss=final_dev_loss,
        epochs_run=epochs_run,
        train_precision_per_epoch=tuple(train_precisions),
        dev_precision_per_epoch=tuple(dev_precisions),
        train_recall_per_epoch=tuple(train_recalls),
        dev_recall_per_epoch=tuple(dev_recalls),
        train_loss_per_epoch=tuple(train_losses),
        dev_loss_per_epoch=tuple(dev_losses),
        train_f1_per_epoch=calc_f1_curve(precision_curve=train_precisions, recall_curve=train_recalls),
        dev_f1_per_epoch=calc_f1_curve(precision_curve=dev_precisions, recall_curve=dev_recalls),
    )


def get_obj_row(*, df_corpora: Any, objective_corpora: str, averaging_type: str) -> Mapping[str, Any]:
    """Return objective corpus row, fallback to all-corpora row for given averaging."""
    objective_row_name = f'All-{averaging_type}'
    if objective_corpora in df_corpora.index:
        return df_corpora.loc[objective_corpora].to_dict()
    return df_corpora.loc[objective_row_name].to_dict()
