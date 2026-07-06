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
from typing import Any, Iterator, Literal, Mapping, Sequence, Type

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from clearml import Task
from geneea.catlib.model.nnet import (
    EntityAttention3MLP,
    EntityAttention2MLP,
    EntityAttentionMLP,
    EntityMhaAttention2MLP,
    EntityMhaAttentionMLP,
    EntityNoPoolingMLP,
    LeakyMLP,
    MLP,
    MLPGELU,
    MLPNN,
    NeuralCategModel,
    SkipMLP,
)
from geneea.catlib.vec.dataset import EmbeddingDataset
from geneea.catlib.vec.vectorizer import Vectorizer

from iptc_entity_pipeline.category_sets import load_relevant_cat_ids
from iptc_entity_pipeline.config import EvaluationCnf
from iptc_entity_pipeline.dataset_builder import RaggedEmbeddingDataset, ragged_collate_fn

LOGGER = logging.getLogger(__name__)


def _batch_to_device(batch: Any, *, device: torch.device) -> Any:
    """Move tensor containers to target device."""
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: _batch_to_device(value, device=device) for key, value in batch.items()}
    if isinstance(batch, tuple):
        return tuple(_batch_to_device(item, device=device) for item in batch)
    if isinstance(batch, list):
        return [_batch_to_device(item, device=device) for item in batch]
    return batch


def _resolve_nn_type(model_config: Mapping[str, Any]) -> Type[MLPNN]:
    """Resolve NN architecture from legacy model config payload."""
    nn_name = str(model_config.get('nnType', 'mlp')).strip().lower()
    if nn_name == 'mlp':
        return MLP
    if nn_name in {'mlp_gelu', 'gelu_mlp'}:
        return MLPGELU
    if nn_name in {'skip_mlp', 'mlp_skip'}:
        return SkipMLP
    if nn_name in {'leaky_mlp', 'mlp_leaky'}:
        return LeakyMLP
    if nn_name == 'entity_no_pooling':
        return EntityNoPoolingMLP
    if nn_name == 'entity_attention_mlp':
        return EntityAttentionMLP
    if nn_name in {'entity_attention3_mlp', 'entity_attention_3_mlp'}:
        return EntityAttention3MLP
    if nn_name in {'entity_attention2_mlp', 'entity_attention_2_mlp'}:
        return EntityAttention2MLP
    if nn_name in {'entity_mha_attention_mlp', 'entity_multihead_attention_mlp'}:
        return EntityMhaAttentionMLP
    if nn_name in {'entity_mha_attention2_mlp', 'entity_multihead_attention2_mlp'}:
        return EntityMhaAttention2MLP
    raise ValueError(f'Unknown nnType: {nn_name}')


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EpochStats:
    """Per-epoch metric accumulator for a single data split."""

    precision: list[float] = field(default_factory=list)
    recall: list[float] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    macro_relevant_f1: list[float] = field(default_factory=list)


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
    train_macro_relevant_f1s: list[float]
    dev_macro_relevant_f1s: list[float]


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
    relevant_indices: list[int] | None = None,
) -> tuple[float, float, float, float]:
    """Compute precision, recall, and mean loss over a dataloader split.

    :return: ``(precision, recall, mean_loss, macro_relevant_f1)`` tuple.
    """
    total_gold = 0.0
    total_pred = 0.0
    total_correct = 0.0
    total_loss = 0.0
    num_batches = len(dataloader)
    relevant_idx_tensor = (
        torch.as_tensor(relevant_indices, dtype=torch.long, device=model._device)
        if relevant_indices else None
    )
    tp_rel = np.zeros(len(relevant_indices), dtype=np.int64) if relevant_indices else None
    fp_rel = np.zeros(len(relevant_indices), dtype=np.int64) if relevant_indices else None
    fn_rel = np.zeros(len(relevant_indices), dtype=np.int64) if relevant_indices else None

    with torch.no_grad():
        model._nn.eval()
        for x_batch, y_batch in dataloader:
            x_batch = _batch_to_device(x_batch, device=model._device)
            y_batch = y_batch.to(model._device)
            pred = model._nn(x_batch)
            pred_bin = pred > thr
            gold_bin = y_batch > 0.5
            total_loss += loss_fn(pred, y_batch).item()
            total_pred += pred_bin.type(torch.float).sum().item()
            total_gold += gold_bin.type(torch.float).sum().item()
            total_correct += torch.logical_and(pred_bin, gold_bin).type(torch.float).sum().item()
            if relevant_idx_tensor is not None and tp_rel is not None and fp_rel is not None and fn_rel is not None:
                pred_rel = pred_bin.index_select(dim=1, index=relevant_idx_tensor)
                gold_rel = gold_bin.index_select(dim=1, index=relevant_idx_tensor)
                tp_rel += torch.logical_and(pred_rel, gold_rel).sum(dim=0).detach().cpu().numpy().astype(np.int64)
                fp_rel += (
                    torch.logical_and(pred_rel, torch.logical_not(gold_rel))
                    .sum(dim=0).detach().cpu().numpy().astype(np.int64)
                )
                fn_rel += (
                    torch.logical_and(torch.logical_not(pred_rel), gold_rel)
                    .sum(dim=0).detach().cpu().numpy().astype(np.int64)
                )

    mean_loss = total_loss / num_batches
    precision = total_correct / (total_pred or 1.0)
    recall = total_correct / (total_gold or 1.0)
    macro_relevant_f1 = float('nan')
    if tp_rel is not None and fp_rel is not None and fn_rel is not None and len(tp_rel) > 0:
        denom = (2 * tp_rel) + fp_rel + fn_rel
        per_class_f1 = np.divide(
            2 * tp_rel,
            denom,
            out=np.zeros(len(tp_rel), dtype=float),
            where=denom > 0,
        )
        macro_relevant_f1 = float(np.mean(per_class_f1))
    return precision, recall, mean_loss, macro_relevant_f1


def _calc_f1(*, precision: float, recall: float) -> float:
    """Compute F1 from precision and recall, returning 0 for empty denominators."""
    denom = precision + recall
    return 0.0 if denom == 0.0 else float((2.0 * precision * recall) / denom)


def _clone_state_cpu(model: NeuralCategModel) -> dict[str, Any]:
    """Snapshot model weights to CPU for early-stopping checkpoint."""
    return {k: v.detach().cpu().clone() for k, v in model._nn.state_dict().items()}


def _restore_state_cpu(model: NeuralCategModel, state: dict[str, Any]) -> None:
    """Restore model weights from a CPU snapshot."""
    model._nn.load_state_dict({k: v.to(model._device) for k, v in state.items()})


def _log_validation(
    *,
    c_log: Any,
    precision: float,
    recall: float,
    loss: float,
    print_logs: bool,
) -> None:
    """Emit human-readable validation results to ClearML logger."""
    c_log.report_text(
        f'Test Error: \n Prec: {100 * precision:>0.2f}%,\n'
        f' Recall: {100 * recall:>0.2f}%,\n'
        f' Avg loss: {loss:>8f} \n',
        level=logging.INFO,
        print_console=print_logs,
    )


def _log_info(*, logger: Any, message: str, print_logs: bool) -> None:
    """Log one INFO message to ClearML text logs."""
    logger.report_text(message, level=logging.INFO, print_console=print_logs)


def _log_elapsed(*, logger: Any, label: str, started_at: float, print_logs: bool) -> None:
    """Log elapsed seconds since ``started_at`` to ClearML text logs."""
    _log_info(logger=logger, message=f'{label}: {time.time() - started_at}', print_logs=print_logs)


def _report_stats_scalars(
    *,
    logger: Any,
    title: str,
    precision: float,
    recall: float,
    loss: float,
    iteration: int,
    macro_relevant_f1: float = float('nan'),
) -> None:
    """Report precision/recall/loss series for one iteration."""
    logger.report_scalar(title=title, series='Precision', value=precision, iteration=iteration)
    logger.report_scalar(title=title, series='Recall', value=recall, iteration=iteration)
    logger.report_scalar(
        title=title,
        series='F1',
        value=_calc_f1(precision=precision, recall=recall),
        iteration=iteration,
    )
    logger.report_scalar(title=title, series='Loss', value=loss, iteration=iteration)


# ---------------------------------------------------------------------------
# Public entry points (camelCase preserved for legacy compatibility)
# ---------------------------------------------------------------------------

def createClassificationModel(
    embDim: int,
    outDim: int,
    modelConfig: dict[str, Any],
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
        _log_info(logger=logger, message='Creating classification model', print_logs=logConfig['PRINT_LOGS'])
        if connect_config:
            task.connect(modelConfig, name='modelConfig')

    nn_type = _resolve_nn_type(model_config=modelConfig)
    nn_kwargs: dict[str, Any] = {}
    if nn_type in {
        EntityNoPoolingMLP,
        EntityAttention3MLP,
        EntityAttentionMLP,
        EntityAttention2MLP,
        EntityMhaAttentionMLP,
        EntityMhaAttention2MLP,
    }:
        nn_kwargs['entityDim'] = int(modelConfig['entityDim'])
    if nn_type in {EntityAttentionMLP, EntityAttention2MLP}:
        nn_kwargs.update({
            'attentionHiddenDim': int(modelConfig['attentionHiddenDim']),
            'attentionDropout': float(modelConfig.get('attentionDropout', 0.0)),
        })
    if nn_type in {EntityMhaAttentionMLP, EntityMhaAttention2MLP}:
        nn_kwargs['attentionNumHeads'] = int(modelConfig.get('attentionNumHeads', 1))
    bias_from_prior_logits = modelConfig.get('biasFromPriorLogits')

    return NeuralCategModel.create(
        embDim=embDim,
        outDim=outDim,
        hiddenDim=modelConfig['hiddenDim'],
        dropouts=[modelConfig['dropouts1'], modelConfig['dropouts2']],
        nnType=nn_type,
        textVectorizer=textVectorizer,
        nnKwargs=nn_kwargs,
        biasFromPriorLogits=bias_from_prior_logits,
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
    :param trainingConfig: Training configuration. May include
        ``trainValidation`` (default True); set False to skip
        the per-epoch full training-set validation pass and train-only ClearML scalars.
    :param logConfig: Logging configuration.
    :param connect_config: Register config with ClearML task (disable during CV).
    :return: Typed training result with model, loss, and per-epoch curves.
    """
    task = Task.current_task()
    clearml_logger = task.get_logger()
    print_logs = logConfig['PRINT_LOGS']
    _log_info(logger=clearml_logger, message='Training classification model', print_logs=print_logs)
    if connect_config:
        task.connect(trainingConfig, name='trainingConfig')

    validation_split_name = str(trainingConfig.get('validationSplitName', 'dev')).strip().lower() or 'dev'
    train_validation = bool(trainingConfig.get('trainValidation', True))
    validation_title = validation_split_name.capitalize()

    opt_factory, lr = _parse_optimizer(trainingConfig['optimizerConfig'])
    sched_factory = _parse_lr_scheduler(trainingConfig['lrSchedulerConfig'])
    loss_fn = _parse_loss(trainingConfig['lossConfig'], trainData, model._device)

    optimizer = opt_factory(model._nn.parameters(), lr)
    scheduler = sched_factory(optimizer)
    model.catList = list(trainData.catList)
    relevant_cat_ids = load_relevant_cat_ids()
    cat_to_idx = {str(cat_id): idx for idx, cat_id in enumerate(model.catList)}
    relevant_indices = [cat_to_idx[cat_id] for cat_id in sorted(relevant_cat_ids) if cat_id in cat_to_idx]
    _log_info(
        logger=clearml_logger,
        message=f'Relevant macro tracking classes found={len(relevant_indices)}/{len(relevant_cat_ids)}',
        print_logs=print_logs,
    )
    _log_info(
        logger=clearml_logger, message=f'Training model with {len(model.catList)} categories', print_logs=print_logs,
    )

    train_collate = ragged_collate_fn if isinstance(trainData, RaggedEmbeddingDataset) else None
    dev_collate = ragged_collate_fn if isinstance(devData, RaggedEmbeddingDataset) else None
    train_loader = torch.utils.data.DataLoader(
        trainData,
        batch_size=trainingConfig['batchSize'],
        shuffle=True,
        drop_last=True,
        collate_fn=train_collate,
    )
    dev_loader = torch.utils.data.DataLoader(
        devData,
        batch_size=trainingConfig['batchSize'],
        collate_fn=dev_collate,
    )

    train_stats = EpochStats()
    dev_stats = EpochStats()
    es_patience = int(trainingConfig.get('earlyStoppingPatience', 0))
    es_min_delta = float(trainingConfig.get('earlyStoppingMinDelta', 0.0))
    es_metric = str(trainingConfig.get('earlyStoppingMetric', 'loss')).strip().lower() or 'loss'
    if es_metric not in {'loss', 'f1'}:
        raise ValueError(f'Unknown early stopping metric: {es_metric}')

    best_state_cpu: dict[str, Any] | None = None
    best_score: float | None = None
    best_dev_loss: float | None = None
    epochs_without_improvement = 0

    start_time = time.time()
    for epoch in range(trainingConfig['epochs']):
        last_time = time.time()
        epoch_msg = f'Epoch {epoch + 1}\n-------------------------------time: {time.time() - last_time}'
        _log_info(logger=clearml_logger, message=epoch_msg, print_logs=print_logs)
        model._trainEpoch(train_loader, loss_fn, optimizer, scheduler)
        _log_elapsed(logger=clearml_logger, label='time make train epoch', started_at=last_time, print_logs=print_logs)

        last_time = time.time()
        dev_prec, dev_rec, dev_loss, dev_macro_relevant_f1 = _validate_split(
            model=model,
            dataloader=dev_loader,
            loss_fn=loss_fn,
            relevant_indices=relevant_indices,
        )
        _log_validation(c_log=clearml_logger, precision=dev_prec, recall=dev_rec, loss=dev_loss, print_logs=print_logs)
        validation_done_label = f'time {validation_split_name} validation done'
        _log_elapsed(logger=clearml_logger, label=validation_done_label, started_at=last_time, print_logs=print_logs)

        last_time = time.time()
        dev_stats.precision.append(dev_prec)
        dev_stats.recall.append(dev_rec)
        dev_stats.loss.append(dev_loss)
        dev_stats.macro_relevant_f1.append(dev_macro_relevant_f1)
        _report_stats_scalars(
            logger=clearml_logger,
            title=f'{validation_title} Training Stats',
            precision=dev_prec,
            recall=dev_rec,
            loss=dev_loss,
            iteration=epoch,
            macro_relevant_f1=dev_macro_relevant_f1,
        )

        if es_patience > 0:
            dev_f1 = _calc_f1(precision=dev_prec, recall=dev_rec)
            current_score = dev_loss if es_metric == 'loss' else dev_f1
            if best_score is None:
                is_improved = True
            elif es_metric == 'loss':
                is_improved = current_score < best_score - es_min_delta
            else:
                is_improved = current_score > best_score + es_min_delta
            if is_improved:
                best_score = current_score
                best_dev_loss = dev_loss
                best_state_cpu = _clone_state_cpu(model)
                epochs_without_improvement = 0
                _log_info(
                    logger=clearml_logger,
                    message=(
                        f'Early stopping: new best {validation_split_name} {es_metric}={current_score:.6f} '
                        f'at epoch {epoch + 1}'
                    ),
                    print_logs=print_logs,
                )
            else:
                epochs_without_improvement += 1
                _log_info(
                    logger=clearml_logger,
                    message=f'Early stopping: epochs without improvement: {epochs_without_improvement}',
                    print_logs=print_logs,
                )

        _log_elapsed(logger=clearml_logger, label='time done', started_at=last_time, print_logs=print_logs)

        if train_validation:
            last_time = time.time()
            train_prec, train_rec, train_loss, train_macro_relevant_f1 = _validate_split(
                model=model,
                dataloader=train_loader,
                loss_fn=loss_fn,
                relevant_indices=relevant_indices,
            )
            _log_elapsed(
                logger=clearml_logger, label='time train validation done', started_at=last_time,
                print_logs=print_logs,
            )

            last_time = time.time()
            train_stats.precision.append(train_prec)
            train_stats.recall.append(train_rec)
            train_stats.loss.append(train_loss)
            train_stats.macro_relevant_f1.append(train_macro_relevant_f1)
            _report_stats_scalars(
                logger=clearml_logger,
                title='Train Training Stats',
                precision=train_prec,
                recall=train_rec,
                loss=train_loss,
                iteration=epoch,
                macro_relevant_f1=train_macro_relevant_f1,
            )

        if es_patience > 0 and epochs_without_improvement >= es_patience:
            clearml_logger.report_text(
                f'Early stopping: no improvement in {validation_split_name} {es_metric} for {es_patience} epoch(s); '
                f'stopping after epoch {epoch + 1} (best {es_metric}={best_score:.6f})',
                level=logging.INFO, print_console=print_logs,
            )
            break

    if es_patience > 0 and best_state_cpu is not None:
        _restore_state_cpu(model, best_state_cpu)
        clearml_logger.report_text(
            f'Early stopping: restored weights from best {validation_split_name} {es_metric} epoch',
            level=logging.INFO, print_console=print_logs,
        )

    _log_elapsed(logger=clearml_logger, label='time', started_at=start_time, print_logs=print_logs)
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
        train_macro_relevant_f1s=list(train_stats.macro_relevant_f1),
        dev_macro_relevant_f1s=list(dev_stats.macro_relevant_f1),
    )


def predict_score_matrix(
    *,
    model: NeuralCategModel,
    eval_data: Any,
    batch_size: int | None = None,
) -> np.ndarray:
    """Return a dense ``(n_docs, n_classes)`` sigmoid score matrix.

    Iterates over ``eval_data`` with the same ``DataLoader`` behavior as
    :meth:`NeuralCategModel.classifyDataset`, applies ``Sigmoid`` to the
    raw logits, and concatenates batches into a single ``float32`` array.
    The column order is the model's ``catList``.

    Used by :func:`evaluateModel` and by ``cross_validation`` so the
    threshold-tuning path can slice and re-evaluate predictions without
    materializing the ~3 GB Python list of ``(cat_id, score)`` tuples
    that ``classifyDataset(returnScores=True)`` would otherwise produce
    for a 20k-doc dev fold over ~1k classes.

    :param model: Trained classification model.
    :param eval_data: Dataset compatible with :class:`torch.utils.data.DataLoader`.
    :param batch_size: Optional batch size override.
    :return: Sigmoid probability matrix with shape ``(len(eval_data), len(model.catList))``.
    """
    from torch.utils.data import DataLoader

    cat_count = len(model.catList) if model.catList is not None else 0
    bs = int(batch_size) if batch_size is not None else int(model.DEFAULT_BATCH_SIZE)
    collate_fn = getattr(eval_data, 'collate_fn', None)
    sigmoid = nn.Sigmoid()
    batches: list[np.ndarray] = []
    model._nn.eval()
    with torch.no_grad():
        for batch in DataLoader(eval_data, batch_size=bs, collate_fn=collate_fn):
            features = batch[0] if isinstance(batch, (tuple, list)) else batch
            logits = model._nn(_batch_to_device(features, device=model._device))
            probs = sigmoid(logits)
            batches.append(probs.detach().cpu().numpy().astype(np.float32, copy=False))
    if not batches:
        return np.zeros((0, cat_count), dtype=np.float32)
    return np.vstack(batches)


def _iter_wgh_labels_rows(
    score_matrix: np.ndarray,
    cats: list[str],
    thr: float,
) -> Iterator[list[tuple[str, float]]]:
    """Inner generator for :func:`wgh_labels_from_score_matrix`."""
    for row in score_matrix:
        idxs = np.where(row > thr)[0]
        scored = [(cats[int(k)], float(row[int(k)])) for k in idxs]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        yield scored


def wgh_labels_from_score_matrix(
    *,
    score_matrix: np.ndarray,
    cat_list: Sequence[str],
    thr: float = -np.inf,
) -> Iterator[list[tuple[str, float]]]:
    """Yield the legacy ``list[tuple[cat, score]]`` view row-by-row.

    Mirrors the per-row output of
    :func:`geneea.catlib.vec.util.vecToScoredCats` (``score > thr`` strict
    inequality, sorted by score descending) so callers that persist the
    legacy format on disk (e.g. :func:`model_io.save_outputs`) keep
    producing byte-compatible artifacts. The default ``thr=-inf`` matches
    the historical ``EvaluationCnf.threshold_predict = -9999`` setting,
    i.e. retain every ``(cat, score)`` tuple. This is a one-shot
    conversion used only at the end of ``eval_final``; it is *not* used
    inside the CV loop.

    Implemented as a generator so callers that only need rows
    sequentially (e.g. row-by-row serialization) never have to hold the
    full ``list[list[tuple[cat, score]]]`` in memory. Callers that need
    a concrete list should wrap with ``list(...)``. Shape validation is
    eager (a wrapper around the inner generator).

    :param score_matrix: Dense ``(n_docs, n_classes)`` score matrix
        aligned with ``cat_list``.
    :param cat_list: Category ids matching the matrix columns.
    :param thr: Score threshold (strict ``>``), matching legacy behavior.
    :return: Iterator of per-document WghLabels rows, sorted by score
        descending.
    """
    if score_matrix.ndim != 2:
        raise ValueError(f'score_matrix must be 2D, got shape={score_matrix.shape}')
    if score_matrix.shape[1] != len(cat_list):
        raise ValueError(
            f'score_matrix columns ({score_matrix.shape[1]}) do not match cat_list ({len(cat_list)})'
        )
    return _iter_wgh_labels_rows(score_matrix, list(cat_list), float(thr))


def evaluateModel(
    model: NeuralCategModel,
    evalData: EmbeddingDataset,
    evaluation_config: EvaluationCnf,
    customThresholds: Mapping[str, float] | None = None,
    *,
    returnPredictions: bool = False,
    connect_config: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Legacy evaluation entry point.

    Internally predicts a dense ``(n_docs, n_classes)`` sigmoid score
    matrix once, then derives normalized per-doc category lists via
    :func:`evaluate.pred_cats_from_matrix` and produces the per-corpus /
    per-class metric tables. When ``returnPredictions=True`` the dense
    matrix (columns aligned with ``model.catList``) is returned instead
    of the previous Python ``list[list[tuple[cat, score]]]`` so callers
    (notably CV threshold tuning) can slice and re-evaluate without
    paying the ~30x Python-object overhead.

    :param model: Trained model.
    :param evalData: Evaluation dataset.
    :param evaluation_config: Typed evaluation settings.
    :param customThresholds: Optional per-label thresholds.
    :param returnPredictions: Whether to also return the dense score matrix.
    :param connect_config: Register config with ClearML task (disable during CV).
    :return: Corpora/class tables, optionally with the dense score matrix.
    """
    from dataclasses import asdict

    from iptc_entity_pipeline.evaluation.evaluate import (
        evaluate_classes,
        evaluate_corpora,
        pred_cats_from_matrix,
    )

    task = Task.current_task()
    if task is not None and connect_config:
        task.connect(asdict(evaluation_config), name='evaluationConfig')

    cat_list = list(model.catList) if model.catList is not None else list(evalData.corpus.catList)
    score_matrix = predict_score_matrix(model=model, eval_data=evalData)
    pred_cats = pred_cats_from_matrix(
        score_matrix=score_matrix,
        cat_list=cat_list,
        threshold=evaluation_config.threshold_eval,
        cat_to_thr=customThresholds,
    )
    df_corpora = evaluate_corpora(
        pred_cats=pred_cats,
        eval_corpus=evalData.corpus,
        per_corpus=evaluation_config.per_corpus,
        averaging_type=evaluation_config.averaging_type,
    )
    df_classes = evaluate_classes(
        pred_cats=pred_cats,
        eval_corpus=evalData.corpus,
        per_class=evaluation_config.per_class,
    )
    if returnPredictions:
        return df_corpora, df_classes, score_matrix
    return df_corpora, df_classes
