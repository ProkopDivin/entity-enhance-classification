"""Legacy model/train/eval helpers ported from original IPTC pipeline.

Source provenance:
- Source file: /home/share/clearml-pipelines/iptc/IPTC-pipeline-GE-2905/iptc_pipeline.py
- Functions ported for behavior preservation: createClassificationModel,
  trainClassificationModel, evaluateModel.
"""

# CamelCase names are preserved intentionally for the three public entry
# points to match the original legacy implementation.

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from clearml import Task
from geneea.catlib.model.nnet import NeuralCategModel
from geneea.catlib.vec.dataset import EmbeddingDataset
from geneea.catlib.vec.vectorizer import Vectorizer

from iptc_entity_pipeline.config import EvaluationCnf

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EpochStats:
    """Per-epoch metric accumulator for a single data split."""

    precision: list[float] = field(default_factory=list)
    recall: list[float] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class LegacyTrainResult:
    """Typed result replacing the raw 9-tuple from trainClassificationModel."""

    model: NeuralCategModel
    final_dev_loss: float
    epochs_run: int
    train_precisions: list[float]
    train_recalls: list[float]
    train_losses: list[float]
    dev_precisions: list[float]
    dev_recalls: list[float]
    dev_losses: list[float]


# ---------------------------------------------------------------------------
# Loss / optimizer / scheduler factories
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal loss for imbalanced multi-label classification."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean') -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets_f = targets.type(torch.float32)
        at = self.alpha * targets_f + (1 - self.alpha) * (1 - targets_f)
        pt = torch.exp(-bce)
        loss = at * (1 - pt) ** self.gamma * bce
        if self.reduction == 'mean':
            return torch.mean(loss)
        if self.reduction == 'sum':
            return torch.sum(loss)
        return loss


def _parse_optimizer(config: dict[str, Any]) -> tuple[Any, float]:
    """Build optimizer factory and learning rate from config dict."""
    if config['name'] == 'adam':
        lr = config['adamConfig']['lr']

        def factory(params: Any, lr: float) -> torch.optim.Optimizer:
            return torch.optim.Adam(params, lr=lr)

        return factory, lr
    raise ValueError(f'Unknown optimizer: {config["name"]}')


def _parse_lr_scheduler(config: dict[str, Any]) -> Any:
    """Build LR scheduler factory from config dict."""
    name = config['name']
    if name == 'stepLR':
        sched_cfg = config['stepLRConfig']

        def factory(opt: torch.optim.Optimizer) -> Any:
            return torch.optim.lr_scheduler.StepLR(
                opt, step_size=sched_cfg['stepSize'], gamma=sched_cfg['gamma'],
            )

        return factory
    if name == 'cosineAnnealingLR':
        sched_cfg = config['cosineAnnealingLRConfig']

        def factory(opt: torch.optim.Optimizer) -> Any:
            return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=sched_cfg['T_max'])

        return factory
    raise ValueError(f'Unknown lr scheduler: {name}')


def _parse_loss(
    config: dict[str, Any],
    train_data: EmbeddingDataset,
    device: torch.device,
) -> nn.Module:
    """Build loss function from config dict."""
    name = config['name']
    if name == 'bceWithLogitsLoss':
        return nn.BCEWithLogitsLoss()
    if name == 'focalLoss':
        return FocalLoss(config['focalLossConfig']['alpha'], config['focalLossConfig']['gamma'])
    if name == 'bceWithLogitsLossWeighted':
        corpus = train_data.corpus
        freqs: dict[str, int] = {cat: 0 for cat in corpus.catList}
        for doc in corpus:
            for cat in doc.cats:
                freqs[cat] += 1
        pos_weights = [
            (len(corpus) - freqs[cat]) / freqs[cat] if freqs[cat] > 0 else 1.0
            for cat in corpus.catList
        ]
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weights).to(device))
    raise ValueError(f'Unknown loss: {name}')


# ---------------------------------------------------------------------------
# Validation / early-stopping helpers
# ---------------------------------------------------------------------------

def _validate_split(
    *,
    model: NeuralCategModel,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    thr: float = 0.0,
) -> tuple[float, float, float]:
    """Compute precision, recall, and mean loss over a dataloader split.

    :return: ``(precision, recall, mean_loss)`` tuple.
    """
    total_gold = 0.0
    total_pred = 0.0
    total_correct = 0.0
    total_loss = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
        model._nn.eval()
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(model._device)
            y_batch = y_batch.to(model._device)
            pred = model._nn(x_batch)
            total_loss += loss_fn(pred, y_batch).item()
            total_pred += (pred > thr).type(torch.float).sum().item()
            total_gold += (y_batch > 0.5).type(torch.float).sum().item()
            total_correct += torch.logical_and(pred > thr, y_batch > 0.5).type(torch.float).sum().item()

    mean_loss = total_loss / num_batches
    precision = total_correct / (total_pred or 1.0)
    recall = total_correct / (total_gold or 1.0)
    return precision, recall, mean_loss


def _clone_state_cpu(model: NeuralCategModel) -> dict[str, Any]:
    """Snapshot model weights to CPU for early-stopping checkpoint."""
    return {k: v.detach().cpu().clone() for k, v in model._nn.state_dict().items()}


def _restore_state_cpu(model: NeuralCategModel, state: dict[str, Any]) -> None:
    """Restore model weights from a CPU snapshot."""
    model._nn.load_state_dict({k: v.to(model._device) for k, v in state.items()})


def _log_validation(
    *,
    clearml_logger: Any,
    precision: float,
    recall: float,
    loss: float,
    print_logs: bool,
) -> None:
    """Emit human-readable validation results to ClearML logger."""
    clearml_logger.report_text(
        f'Test Error: \n Prec: {100 * precision:>0.2f}%,\n'
        f' Recall: {100 * recall:>0.2f}%,\n'
        f' Avg loss: {loss:>8f} \n',
        level=logging.INFO,
        print_console=print_logs,
    )


# ---------------------------------------------------------------------------
# Public entry points (camelCase preserved for legacy compatibility)
# ---------------------------------------------------------------------------

def createClassificationModel(
    embDim: int,
    outDim: int,
    modelConfig: dict[str, int | str | float],
    textVectorizer: Vectorizer[str] | Literal['not None'],
    logConfig: dict[str, Any],
    connect_config: bool = True,
) -> NeuralCategModel:
    """
    Legacy model constructor kept identical to original behavior.

    :param embDim: Input embedding size.
    :param outDim: Number of output categories.
    :param modelConfig: Model configuration dictionary.
    :param textVectorizer: Text vectorizer or legacy marker.
    :param logConfig: Logging configuration.
    :param connect_config: Register config with ClearML task (disable during CV).
    :return: Initialized neural classifier model.
    """
    task = Task.current_task()
    if task is not None:
        logger = task.get_logger()
        logger.report_text('Creating classification model', level=logging.INFO, print_console=logConfig['PRINT_LOGS'])
        if connect_config:
            task.connect(modelConfig, name='modelConfig')

    return NeuralCategModel.create(
        embDim=embDim,
        outDim=outDim,
        hiddenDim=modelConfig['hiddenDim'],
        dropouts=[modelConfig['dropouts1'], modelConfig['dropouts2']],
        textVectorizer=textVectorizer,
    )


def trainClassificationModel(
    model: NeuralCategModel,
    trainData: EmbeddingDataset,
    devData: EmbeddingDataset,
    trainingConfig: dict[str, Any],
    logConfig: dict[str, Any],
    connect_config: bool = True,
) -> LegacyTrainResult:
    """
    Legacy training loop kept equivalent to original pipeline.

    :param model: Model to train.
    :param trainData: Training dataset.
    :param devData: Development dataset.
    :param trainingConfig: Training configuration.
    :param logConfig: Logging configuration.
    :param connect_config: Register config with ClearML task (disable during CV).
    :return: Typed training result with model, loss, and per-epoch curves.
    """
    task = Task.current_task()
    clearml_logger = task.get_logger()
    print_logs = logConfig['PRINT_LOGS']
    clearml_logger.report_text('Training classification model', level=logging.INFO, print_console=print_logs)
    if connect_config:
        task.connect(trainingConfig, name='trainingConfig')

    validation_split_name = str(trainingConfig.get('validationSplitName', 'dev')).strip().lower() or 'dev'
    validation_title = validation_split_name.capitalize()

    opt_factory, lr = _parse_optimizer(trainingConfig['optimizerConfig'])
    sched_factory = _parse_lr_scheduler(trainingConfig['lrSchedulerConfig'])
    loss_fn = _parse_loss(trainingConfig['lossConfig'], trainData, model._device)

    optimizer = opt_factory(model._nn.parameters(), lr)
    scheduler = sched_factory(optimizer)
    model.catList = list(trainData.catList)
    clearml_logger.report_text(
        f'Training model with {len(model.catList)} categories', level=logging.INFO, print_console=print_logs,
    )

    train_loader = torch.utils.data.DataLoader(
        trainData, batch_size=trainingConfig['batchSize'], shuffle=True, drop_last=True,
    )
    dev_loader = torch.utils.data.DataLoader(devData, batch_size=trainingConfig['batchSize'])

    train_stats = EpochStats()
    dev_stats = EpochStats()
    es_patience = int(trainingConfig.get('earlyStoppingPatience', 0))
    es_min_delta = float(trainingConfig.get('earlyStoppingMinDelta', 0.0))

    best_state_cpu: dict[str, Any] | None = None
    best_dev_loss: float | None = None
    epochs_without_improvement = 0

    start_time = time.time()
    for epoch in range(trainingConfig['epochs']):
        last_time = time.time()
        clearml_logger.report_text(
            f'Epoch {epoch + 1}\n-------------------------------time: {time.time() - last_time}',
            level=logging.INFO, print_console=print_logs,
        )
        model._trainEpoch(train_loader, loss_fn, optimizer, scheduler)
        clearml_logger.report_text(
            f'time make train epoch: {time.time() - last_time}', level=logging.INFO, print_console=print_logs,
        )

        last_time = time.time()
        dev_prec, dev_rec, dev_loss = _validate_split(model=model, dataloader=dev_loader, loss_fn=loss_fn)
        _log_validation(
            clearml_logger=clearml_logger, precision=dev_prec, recall=dev_rec, loss=dev_loss, print_logs=print_logs,
        )
        clearml_logger.report_text(
            f'time {validation_split_name} validation done: {time.time() - last_time}',
            level=logging.INFO, print_console=print_logs,
        )

        last_time = time.time()
        dev_stats.precision.append(dev_prec)
        dev_stats.recall.append(dev_rec)
        dev_stats.loss.append(dev_loss)
        clearml_logger.report_scalar(
            title=f'{validation_title} Training Stats', series='Precision', value=dev_prec, iteration=epoch,
        )
        clearml_logger.report_scalar(
            title=f'{validation_title} Training Stats', series='Recall', value=dev_rec, iteration=epoch,
        )
        clearml_logger.report_scalar(
            title=f'{validation_title} Training Stats', series='Loss', value=dev_loss, iteration=epoch,
        )

        if es_patience > 0:
            if best_dev_loss is None or dev_loss < best_dev_loss - es_min_delta:
                best_dev_loss = dev_loss
                best_state_cpu = _clone_state_cpu(model)
                epochs_without_improvement = 0
                clearml_logger.report_text(
                    f'Early stopping: new best {validation_split_name} loss={dev_loss:.6f} at epoch {epoch + 1}',
                    level=logging.INFO, print_console=print_logs,
                )
            else:
                epochs_without_improvement += 1
                clearml_logger.report_text(
                    f'Early stopping: epochs without improvement: {epochs_without_improvement}',
                    level=logging.INFO, print_console=print_logs,
                )

        clearml_logger.report_text(
            f'time done: {time.time() - last_time}', level=logging.INFO, print_console=print_logs,
        )

        last_time = time.time()
        train_prec, train_rec, train_loss = _validate_split(model=model, dataloader=train_loader, loss_fn=loss_fn)
        clearml_logger.report_text(
            f'time train validation done: {time.time() - last_time}', level=logging.INFO, print_console=print_logs,
        )

        last_time = time.time()
        train_stats.precision.append(train_prec)
        train_stats.recall.append(train_rec)
        train_stats.loss.append(train_loss)
        clearml_logger.report_scalar(title='Train Training Stats', series='Precision', value=train_prec, iteration=epoch)
        clearml_logger.report_scalar(title='Train Training Stats', series='Recall', value=train_rec, iteration=epoch)
        clearml_logger.report_scalar(title='Train Training Stats', series='Loss', value=train_loss, iteration=epoch)

        if es_patience > 0 and epochs_without_improvement >= es_patience:
            clearml_logger.report_text(
                f'Early stopping: no improvement in {validation_split_name} loss for {es_patience} epoch(s); '
                f'stopping after epoch {epoch + 1} (best loss={best_dev_loss:.6f})',
                level=logging.INFO, print_console=print_logs,
            )
            break

    if es_patience > 0 and best_state_cpu is not None:
        _restore_state_cpu(model, best_state_cpu)
        clearml_logger.report_text(
            f'Early stopping: restored weights from best {validation_split_name} epoch',
            level=logging.INFO, print_console=print_logs,
        )

    clearml_logger.report_text(
        f'time: {time.time() - start_time}', level=logging.INFO, print_console=print_logs,
    )
    model._nn.eval()
    final_dev_loss = best_dev_loss if best_dev_loss is not None else (dev_stats.loss[-1] if dev_stats.loss else 0.0)

    return LegacyTrainResult(
        model=model,
        final_dev_loss=final_dev_loss,
        epochs_run=len(dev_stats.loss),
        train_precisions=list(train_stats.precision),
        train_recalls=list(train_stats.recall),
        train_losses=list(train_stats.loss),
        dev_precisions=list(dev_stats.precision),
        dev_recalls=list(dev_stats.recall),
        dev_losses=list(dev_stats.loss),
    )


def evaluateModel(
    model: NeuralCategModel,
    evalData: EmbeddingDataset,
    evaluation_config: EvaluationCnf,
    customThresholds: Mapping[str, float] | None = None,
    *,
    returnPredictions: bool = False,
    connect_config: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, Any]:
    """
    Legacy evaluation entry point.

    Delegates to :func:`evaluate.evaluate_predictions` for the actual metric
    computation while preserving the original function signature.

    :param model: Trained model.
    :param evalData: Evaluation dataset.
    :param evaluation_config: Typed evaluation settings.
    :param customThresholds: Optional per-label thresholds.
    :param returnPredictions: Whether to also return raw prediction scores.
    :param connect_config: Register config with ClearML task (disable during CV).
    :return: Corpora/class tables, optionally with raw prediction scores.
    """
    from dataclasses import asdict

    from iptc_entity_pipeline.evaluate import evaluate_predictions

    task = Task.current_task()
    if task is not None and connect_config:
        task.connect(asdict(evaluation_config), name='evaluationConfig')

    predictions = model.classifyDataset(evalData, thr=evaluation_config.threshold_predict, returnScores=True)
    df_corpora, df_classes = evaluate_predictions(
        pred_wgh_cats=predictions,
        eval_corpus=evalData.corpus,
        thr=evaluation_config.threshold_eval,
        cat_to_thr=customThresholds,
        per_corpus=evaluation_config.per_corpus,
        per_class=evaluation_config.per_class,
        averaging_type=evaluation_config.averaging_type,
    )
    if returnPredictions:
        return df_corpora, df_classes, predictions
    return df_corpora, df_classes
