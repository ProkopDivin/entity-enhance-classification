"""Legacy model/train/eval helpers copied from original IPTC pipeline.

Source provenance:
- Source file: /home/share/clearml-pipelines/iptc/IPTC-pipeline-GE-2905/iptc_pipeline.py
- Functions copied for behavior preservation: createClassificationModel,
  trainClassificationModel, evaluateModel.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
from clearml import Task
from geneea.catlib.model.nnet import NeuralCategModel
from geneea.catlib.vec.dataset import EmbeddingDataset
from geneea.catlib.vec.vectorizer import Vectorizer


def createClassificationModel(
    embDim: int,
    outDim: int,
    modelConfig: Dict[str, Union[int, str, float]],
    textVectorizer: Vectorizer[str] | Literal['not None'],
    logConfig: Dict[str, Any],
) -> NeuralCategModel:
    """
    Legacy model constructor kept identical to original behavior.

    :param embDim: Input embedding size.
    :param outDim: Number of output categories.
    :param modelConfig: Model configuration dictionary.
    :param textVectorizer: Text vectorizer or legacy marker.
    :param logConfig: Logging configuration.
    :return: Initialized neural classifier model.
    """
    import logging

    logger = Task.current_task().get_logger()
    logger.report_text('Creating classification model', level=logging.INFO, print_console=logConfig['PRINT_LOGS'])

    Task.current_task().connect(modelConfig, name='modelConfig')
    model = NeuralCategModel.create(
        embDim=embDim,
        outDim=outDim,
        hiddenDim=modelConfig['hiddenDim'],
        dropouts=[modelConfig['dropouts1'], modelConfig['dropouts2']],
        textVectorizer=textVectorizer,
    )

    return model


def trainClassificationModel(
    model: NeuralCategModel,
    trainData: EmbeddingDataset,
    devData: EmbeddingDataset,
    trainingConfig: Dict[str, Any],
    logConfig: Dict[str, Any],
) -> Tuple[
    NeuralCategModel,
    float,
    int,
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
]:
    """
    Legacy training loop kept equivalent to original pipeline.

    :param model: Model to train.
    :param trainData: Training dataset.
    :param devData: Development dataset.
    :param trainingConfig: Training configuration.
    :param logConfig: Logging configuration.
    :return: Tuple of trained model and final dev loss.
    """
    import logging
    import time

    import torch

    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)

    logger = Task.current_task().get_logger()
    logger.report_text('Training classification model', level=logging.INFO, print_console=logConfig['PRINT_LOGS'])
    Task.current_task().connect(trainingConfig, name='trainingConfig')

    def parseOptimizer(optimizerConfig) -> Tuple[torch.optim.Optimizer, float]:
        if optimizerConfig['name'] == 'adam':

            def optimizerFactory(p, lr):
                return torch.optim.Adam(p, lr=lr)

            lr = optimizerConfig['adamConfig']['lr']
        else:
            raise ValueError('Unknown optimizer')

        return optimizerFactory, lr

    def parseLrScheduler(lrSchedulerConfig):
        if lrSchedulerConfig['name'] == 'stepLR':
            lrSchedConfig = lrSchedulerConfig['stepLRConfig']

            def lrSchedFactory(opt):
                return torch.optim.lr_scheduler.StepLR(
                    opt,
                    step_size=lrSchedConfig['stepSize'],
                    gamma=lrSchedConfig['gamma'],
                )

        elif lrSchedulerConfig['name'] == 'cosineAnnealingLR':
            lrSchedConfig = lrSchedulerConfig['cosineAnnealingLRConfig']

            def lrSchedFactory(opt):
                return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=lrSchedConfig['T_max'])

        else:
            raise ValueError('Unknown lr scheduler')

        return lrSchedFactory

    def parseLoss(lossConfig: Dict[str, Any], trainData: EmbeddingDataset, device: torch.device):
        import torch.nn as nn
        import torch.nn.functional as F

        if lossConfig['name'] == 'bceWithLogitsLoss':
            loss_fn = nn.BCEWithLogitsLoss()
        elif lossConfig['name'] == 'focalLoss':

            class FocalLoss(nn.Module):
                def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
                    super(FocalLoss, self).__init__()
                    self.alpha = alpha
                    self.gamma = gamma
                    self.reduction = reduction

                def forward(self, inputs, targets):
                    bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
                    targets = targets.type(torch.float32)
                    at = self.alpha * targets + (1 - self.alpha) * (1 - targets)
                    pt = torch.exp(-bce_loss)
                    focal_loss = at * (1 - pt) ** self.gamma * bce_loss

                    if self.reduction == 'mean':
                        return torch.mean(focal_loss)
                    if self.reduction == 'sum':
                        return torch.sum(focal_loss)
                    return focal_loss

            loss_fn = FocalLoss(
                lossConfig['focalLossConfig']['alpha'],
                lossConfig['focalLossConfig']['gamma'],
            )
        elif lossConfig['name'] == 'bceWithLogitsLossWeighted':
            trainCorpus = trainData.corpus
            freqs = {cat: 0 for cat in trainCorpus.catList}
            for doc in trainCorpus:
                for cat in doc.cats:
                    freqs[cat] += 1
            pos_weights = [
                (len(trainCorpus) - freqs[cat]) / freqs[cat] if freqs[cat] > 0 else 1.0 for cat in trainCorpus.catList
            ]
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weights).to(device))
        else:
            raise ValueError('Unknown loss')

        return loss_fn

    def validation(
        model: NeuralCategModel,
        dataloader: torch.utils.data.DataLoader,
        lossFn: torch.nn.Module,
        thr: float = 0.0,
        logInfo: bool = True,
    ) -> Tuple[float, float, float]:
        total_gold = 0
        total_pred = 0
        num_batches = len(dataloader)
        test_loss, total_correct = 0, 0

        with torch.no_grad():
            model._nn.eval()
            for X, y in dataloader:
                X = X.to(model._device)
                y = y.to(model._device)
                pred = model._nn(X)
                test_loss += lossFn(pred, y).item()
                pred_cnt = (pred > thr).type(torch.float).sum().item()
                gold_cnt = (y > 0.5).type(torch.float).sum().item()
                correct = torch.logical_and(pred > thr, y > 0.5).type(torch.float).sum().item()
                total_correct += correct
                total_pred += pred_cnt
                total_gold += gold_cnt

        test_loss /= num_batches
        precision = total_correct / (total_pred or 1.0)
        recall = total_correct / (total_gold or 1.0)
        if logInfo:
            logger.report_text(
                f'Test Error: \n Prec: {(100 * precision):>0.2f}% ({total_correct}/{total_pred}),\n exc_info=Recall: {(100 * recall):>0.2f}% ({total_correct}/{total_gold}),\n Avg loss: {test_loss:>8f} \n',
                level=logging.INFO,
                print_console=logConfig['PRINT_LOGS'],
            )
        return precision, recall, test_loss

    validation_split_name = str(trainingConfig.get('validationSplitName', 'dev')).strip().lower() or 'dev'
    validation_title_name = validation_split_name.capitalize()

    optimizerFactory, lr = parseOptimizer(trainingConfig['optimizerConfig'])
    lrSchedFactory = parseLrScheduler(trainingConfig['lrSchedulerConfig'])
    loss_fn = parseLoss(trainingConfig['lossConfig'], trainData, model._device)

    optimizer = optimizerFactory(model._nn.parameters(), lr)
    lrSched = lrSchedFactory(optimizer)
    model.catList = list(trainData.catList)
    logger.report_text(
        f'Training model with {len(model.catList)} categories',
        level=logging.INFO,
        print_console=logConfig['PRINT_LOGS'],
    )

    train_loader = torch.utils.data.DataLoader(
        trainData,
        batch_size=trainingConfig['batchSize'],
        shuffle=True,
        drop_last=True,
    )
    dev_loader = torch.utils.data.DataLoader(devData, batch_size=trainingConfig['batchSize'])
    task = Task.current_task()
    logger = task.get_logger()

    train_stats = [[], [], []]
    dev_stats = [[], [], []]
    es_patience = int(trainingConfig.get('earlyStoppingPatience', 0))
    es_min_delta = float(trainingConfig.get('earlyStoppingMinDelta', 0.0))

    def _is_better_loss(*, current_loss: float, best_loss: float) -> bool:
        return current_loss < best_loss - es_min_delta

    def _clone_state_dict_cpu() -> Dict[str, Any]:
        return {k: v.detach().cpu().clone() for k, v in model._nn.state_dict().items()}

    def _load_state_dict_from_cpu(state: Dict[str, Any]) -> None:
        model._nn.load_state_dict({k: v.to(model._device) for k, v in state.items()})

    best_state_cpu: Optional[Dict[str, Any]] = None
    best_dev_loss: Optional[float] = None
    epochs_without_improvement = 0

    start_training_time = time.time()
    for t in range(trainingConfig['epochs']):
        last_time = time.time()
        logger.report_text(
            f'Epoch {t + 1}\n-------------------------------time: {time.time() - last_time}',
            level=logging.INFO,
            print_console=logConfig['PRINT_LOGS'],
        )
        model._trainEpoch(train_loader, loss_fn, optimizer, lrSched)
        logger.report_text(
            f'time make train epoch: {time.time() - last_time}',
            level=logging.INFO,
            print_console=logConfig['PRINT_LOGS'],
        )
        last_time = time.time()
        dev_precision, dev_recall, dev_loss = validation(model, dev_loader, loss_fn)
        logger.report_text(
            f'time {validation_split_name} validation done: {time.time() - last_time}',
            level=logging.INFO,
            print_console=logConfig['PRINT_LOGS'],
        )
        last_time = time.time()

        for i, st in enumerate([dev_precision, dev_recall, dev_loss]):
            dev_stats[i].append(st)
        logger.report_scalar(
            title=f'{validation_title_name} Training Stats',
            series='Precision',
            value=dev_precision,
            iteration=t,
        )
        logger.report_scalar(
            title=f'{validation_title_name} Training Stats',
            series='Recall',
            value=dev_recall,
            iteration=t,
        )
        logger.report_scalar(
            title=f'{validation_title_name} Training Stats',
            series='Loss',
            value=dev_loss,
            iteration=t,
        )

        if es_patience > 0:
            if best_dev_loss is None or _is_better_loss(current_loss=dev_loss, best_loss=best_dev_loss):
                best_dev_loss = dev_loss
                best_state_cpu = _clone_state_dict_cpu()
                epochs_without_improvement = 0
                logger.report_text(
                    f'Early stopping: new best {validation_split_name} loss={dev_loss:.6f} at epoch {t + 1}',
                    level=logging.INFO,
                    print_console=logConfig['PRINT_LOGS'],
                )
            else:
                epochs_without_improvement += 1
                logger.report_text(
                    f'Early stopping: epochs without improvement: {epochs_without_improvement}',
                    level=logging.INFO,
                    print_console=logConfig['PRINT_LOGS'],
                )

        logger.report_text(
            f'time done: {time.time() - last_time}',
            level=logging.INFO,
            print_console=logConfig['PRINT_LOGS'],
        )
        last_time = time.time()

        train_precision, train_recall, train_loss = validation(model, train_loader, loss_fn, logInfo=False)
        logger.report_text(
            f'time train validation done: {time.time() - last_time}',
            level=logging.INFO,
            print_console=logConfig['PRINT_LOGS'],
        )
        last_time = time.time()
        for i, st in enumerate([train_precision, train_recall, train_loss]):
            train_stats[i].append(st)
        logger.report_scalar(title='Train Training Stats', series='Precision', value=train_precision, iteration=t)
        logger.report_scalar(title='Train Training Stats', series='Recall', value=train_recall, iteration=t)
        logger.report_scalar(title='Train Training Stats', series='Loss', value=train_loss, iteration=t)

        if es_patience > 0 and epochs_without_improvement >= es_patience:
            logger.report_text(
                f'Early stopping: no improvement in {validation_split_name} loss for {es_patience} epoch(s); '
                f'stopping after epoch {t + 1} (best loss={best_dev_loss:.6f})',
                level=logging.INFO,
                print_console=logConfig['PRINT_LOGS'],
            )
            break

    if es_patience > 0 and best_state_cpu is not None:
        _load_state_dict_from_cpu(best_state_cpu)
        logger.report_text(
            f'Early stopping: restored weights from best {validation_split_name} epoch',
            level=logging.INFO,
            print_console=logConfig['PRINT_LOGS'],
        )

    logger.report_text(
        f'time: {time.time() - start_training_time}',
        level=logging.INFO,
        print_console=logConfig['PRINT_LOGS'],
    )
    model._nn.eval()
    final_dev_loss = best_dev_loss if best_dev_loss is not None else (dev_stats[2][-1] if dev_stats[2] else 0.0)
    epochs_run = len(dev_stats[2]) if dev_stats[2] else 0
    train_precisions = list(train_stats[0]) if train_stats[0] else []
    train_recalls = list(train_stats[1]) if train_stats[1] else []
    train_losses = list(train_stats[2]) if train_stats[2] else []
    dev_precisions = list(dev_stats[0]) if dev_stats[0] else []
    dev_recalls = list(dev_stats[1]) if dev_stats[1] else []
    dev_losses = list(dev_stats[2]) if dev_stats[2] else []
    return (
        model,
        final_dev_loss,
        epochs_run,
        train_precisions,
        train_recalls,
        train_losses,
        dev_precisions,
        dev_recalls,
        dev_losses,
    )


def evaluateModel(
    model: NeuralCategModel,
    evalData: EmbeddingDataset,
    evaluationConfig: Mapping[str, Any],
    customThresholds: Optional[Mapping[str, float]] = None,
    *,
    returnPredictions: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame] | Tuple[pd.DataFrame, pd.DataFrame, Any]:
    """
    Legacy evaluation entry point.

    Delegates to :func:`evaluate.evaluate_predictions` for the actual metric
    computation while preserving the original function signature.

    :param model: Trained model.
    :param evalData: Evaluation dataset.
    :param evaluationConfig: Evaluation settings.
    :param customThresholds: Optional per-label thresholds.
    :param returnPredictions: Whether to also return raw prediction scores.
    :return: Corpora/class tables, optionally with raw prediction scores.
    """
    from iptc_entity_pipeline.evaluate import evaluate_predictions

    Task.current_task().connect(evaluationConfig, name='evaluationConfig')

    predictions = model.classifyDataset(evalData, thr=evaluationConfig['thresholdPredict'], returnScores=True)
    df_corpora, df_classes = evaluate_predictions(
        pred_wgh_cats=predictions,
        eval_corpus=evalData.corpus,
        thr=evaluationConfig['thresholdEval'],
        cat_to_thr=customThresholds,
        per_corpus=evaluationConfig['perCorpus'],
        per_class=evaluationConfig['perClass'],
        averaging_type=evaluationConfig['averagingType'],
    )
    if returnPredictions:
        return df_corpora, df_classes, predictions
    return df_corpora, df_classes
