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
from geneea.catlib.data import Corpus
from geneea.catlib.model.nnet import NeuralCategModel
from geneea.catlib.vec.dataset import EmbeddingDataset
from geneea.catlib.vec.vectorizer import Vectorizer
from geneea.evaluation.utils import AvgData, ClassData


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
) -> NeuralCategModel:
    """
    Legacy training loop kept equivalent to original pipeline.

    :param model: Model to train.
    :param trainData: Training dataset.
    :param devData: Development dataset.
    :param trainingConfig: Training configuration.
    :param logConfig: Logging configuration.
    :return: Trained model.
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

    def makePlots(train_stats: list[List[float]], dev_stats: list[List[float]]) -> None:
        def scatter(iterations, data):
            return np.hstack([np.atleast_2d(np.arange(0, iterations)).T, np.atleast_2d(data).T])

        logger = Task.current_task().get_logger()
        iterations = len(dev_stats[2])
        dev_precision, dev_recall, dev_loss = dev_stats[0], dev_stats[1], dev_stats[2]
        train_precision, train_recall, train_loss = train_stats[0], train_stats[1], train_stats[2]

        logger.report_scatter2d(
            title='loss during training',
            series=f'{validation_split_name} loss',
            iteration=t + 1,
            scatter=scatter(iterations=iterations, data=dev_loss),
            xaxis='epoch',
            yaxis='loss',
            mode='lines+markers',
        )
        logger.report_scatter2d(
            title='loss during training',
            series='train loss',
            iteration=t + 1,
            scatter=scatter(iterations=iterations, data=train_loss),
            xaxis='epoch',
            yaxis='loss',
            mode='lines+markers',
        )
        logger.report_scatter2d(
            title='precission during training',
            series=f'{validation_split_name} precission',
            iteration=t + 1,
            scatter=scatter(iterations=iterations, data=dev_precision),
            xaxis='epoch',
            yaxis='precision',
            mode='lines+markers',
        )
        logger.report_scatter2d(
            title='precission during training',
            series='train precission',
            iteration=t + 1,
            scatter=scatter(iterations=iterations, data=train_precision),
            xaxis='epoch',
            yaxis='precision',
            mode='lines+markers',
        )
        logger.report_scatter2d(
            title='recall during training',
            series=f'{validation_split_name} recall',
            iteration=t + 1,
            scatter=scatter(iterations=iterations, data=dev_recall),
            xaxis='epoch',
            yaxis='recall',
            mode='lines+markers',
        )
        logger.report_scatter2d(
            title='recall during training',
            series='train recall',
            iteration=t + 1,
            scatter=scatter(iterations=iterations, data=train_recall),
            xaxis='epoch',
            yaxis='recall',
            mode='lines+markers',
        )

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
    makePlots(train_stats, dev_stats)
    return model


def evaluateModel(
    model: NeuralCategModel,
    evalData: EmbeddingDataset,
    evaluationConfig: Mapping[str, Any],
    customThresholds: Optional[Mapping[str, float]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Legacy evaluation implementation copied from original pipeline.

    :param model: Trained model.
    :param evalData: Evaluation dataset.
    :param evaluationConfig: Evaluation settings.
    :param customThresholds: Optional per-label thresholds.
    :return: Corpora-level and class-level evaluation tables.
    """
    from geneea.catlib.model.model import LabelT, WghLabels, filterLabels
    from geneea.evaluation import utils as evalutil
    from geneea.mediacats import iptc

    Task.current_task().connect(evaluationConfig, name='evaluationConfig')
    iptcCats = iptc.IptcTopics.load()

    def isDecentLabel(stats: ClassData) -> bool:
        return stats.precision >= 0.6 and stats.fmeasure(beta=0.4) >= 0.5 and stats.trueCnt >= 10

    def getCatName(catId: str) -> str:
        return iptcCats.getCategory(catId).getLongName(abbreviate=True, shorten=True)

    def normalizeCategories(predCats: List[List[str]]) -> List[List[str]]:
        GLOB_WARMING_CAT_ID = '20000419'
        REMOVED_CAT_IDS = frozenset([GLOB_WARMING_CAT_ID])
        newPredCats = [iptcCats.normalizeCategories([iptcCats.getCategory(c) for c in cs]) for cs in predCats]
        return [sorted(c.id for c in cs if c.id not in REMOVED_CAT_IDS) for cs in newPredCats]

    def update_statistics(
        statistics: Dict[str, List],
        data_count: int,
        stats: ClassData | AvgData,
        digits: int = 3,
    ) -> None:
        statistics['Data Count'].append(data_count)
        statistics['Precision'].append(round(stats.precision, digits))
        statistics['Recall'].append(round(stats.recall, digits))
        statistics['F1'].append(round(stats.fmeasure(beta=1), digits))

    def update_labels(
        statistics: Dict[str, List],
        zeroLabelDocsSc: int,
        predCatsSc: List[List[str]],
        decentLabelsSc: List[str],
    ) -> None:
        statistics['Docs No Labels'].append(round(zeroLabelDocsSc / len(predCatsSc), 3))
        statistics['Decent Labels'].append(len(decentLabelsSc))

    def evaluateCorpuses(
        predWghCats: Union[List[List[LabelT]], List[WghLabels]],
        evalCorpus: Corpus,
        thr: float,
        catToThr: Optional[Mapping[str, float]] = None,
        perCorpus: bool = True,
        averagingType: Literal['datapoint', 'micro', 'macro'] = 'datapoint',
    ) -> pd.DataFrame:
        def prepare_stats(subData: Corpus, predCatsSc: List[List[str]], averagingType: str):
            avgTypes = ['datapoint', 'macro', 'micro']
            avgStatsSc, microStats, indivStatsSc = evalutil.multiStats(goldVals=[d.cats for d in subData], predVals=predCatsSc)
            stats = None
            if averagingType == 'datapoint':
                stats = avgStatsSc
            elif averagingType == 'micro':
                stats = microStats
            elif averagingType == 'macro':
                macroStats = AvgData.empty()
                for cat in indivStats:
                    classData = indivStats[cat]
                    macroStats.update(prec=classData.precision, recall=classData.recall)
                stats = macroStats
            if averagingType not in avgTypes:
                raise ValueError(f'Incorect averagingType, currenttype: {averagingType}, schould be one of these: {avgTypes}')

            zeroLabelDocsSc = sum(1 for cs in predCatsSc if not cs)
            decentLabelsSc = [name for name, stats in indivStatsSc.items() if isDecentLabel(stats)]
            return (avgStatsSc.cnt, stats, zeroLabelDocsSc, decentLabelsSc)

        predCats = [filterLabels(dc, thr=thr, thrByLabel=catToThr, keepWgh=False) for dc in predWghCats]
        predCats = normalizeCategories(predCats)

        statistics = {
            'Data Count': [],
            'Precision': [],
            'Recall': [],
            'F1': [],
            'Docs No Labels': [],
            'Decent Labels': [],
        }
        corporaNames = []
        if perCorpus:
            corporaNames = sorted({d.metadata['corpusName'] for d in evalCorpus})
            for corpusName in corporaNames:
                mask = [d.metadata['corpusName'] == corpusName for d in evalCorpus]
                subData = evalCorpus.filterByBools(mask)
                predCatsSc = [cs for (inSc, cs) in zip(mask, predCats) if inSc]
                if not len(subData) == len(predCatsSc):
                    raise ValueError(
                        f'length of subData is {len(subData)} but it should be equal to length of predCatsSc {len(predCatsSc)}'
                    )
                count, stats, zeroLabelDocsSc, decentLabelsSc = prepare_stats(subData, predCatsSc, averagingType)
                update_statistics(statistics, count, stats, digits=3)
                update_labels(statistics, zeroLabelDocsSc, predCatsSc, decentLabelsSc)
        
        for stat in statistics.keys():
            statistics[stat].append(pd.Series(statistics[stat], dtype=float).mean())
        corporaNames.append('All-macro')
            
        macroStats = {}
        goldVals = [d.cats for d in evalCorpus]
        avgStats, microStats, indivStats = evalutil.multiStats(goldVals=goldVals, predVals=predCats)
        decentLabels = [name for name, stats in indivStats.items() if isDecentLabel(stats)]
        zeroLabelDocs = sum(1 for cs in predCats if not cs)
        update_statistics(statistics, avgStats.cnt, microStats)
        update_labels(statistics, zeroLabelDocs, predCats, decentLabels)
        corporaNames.append('All-micro')
        macroStats = AvgData.empty()
        for cat in indivStats:
            classData = indivStats[cat]
            macroStats.update(prec=classData.precision, recall=classData.recall)
        update_statistics(statistics, avgStats.cnt, avgStats)
        update_labels(statistics, zeroLabelDocs, predCats, decentLabels)
        corporaNames.append('All-datapoint')

        df = pd.DataFrame(data=statistics)
        df.index = corporaNames
        df.index.name = 'Corpus Name'
        return df

    def evaluateClasses(
        predWghCats: List[List[LabelT]] | List[WghLabels],
        evalCorpus: Corpus,
        thr: float,
        catToThr: Mapping[str, float],
        perClass: bool = True,
    ) -> pd.DataFrame:
        predCats = [filterLabels(dc, thr=thr, thrByLabel=catToThr, keepWgh=False) for dc in predWghCats]
        predCats = normalizeCategories(predCats)

        statistics = {'Data Count': [], 'Precision': [], 'Recall': [], 'F1': []}
        categoryNames = []
        if perClass:
            categories = evalCorpus.catList
            categoryNames = ['"' + getCatName(c) + '"' for c in categories]
            for category in categories:
                predCatsSc = [1 if category in cs else 0 for cs in predCats]
                goldValsSc = [1 if category in doc.cats else 0 for doc in evalCorpus]
                classToData, _, _ = evalutil.classStats(trueVals=goldValsSc, predVals=predCatsSc)
                classificationStats = classToData[1]
                update_statistics(statistics, sum(goldValsSc), classificationStats)

        gold_vals = [d.cats for d in evalCorpus]
        avgStats, microStats, classStats = evalutil.multiStats(goldVals=gold_vals, predVals=predCats)
        update_statistics(statistics, avgStats.cnt, microStats)
        categoryNames.append('All - micro avg')
        macroAvg = AvgData.empty()
        for cat in classStats:
            classData = classStats[cat]
            macroAvg.update(prec=classData.precision, recall=classData.recall)
        update_statistics(statistics, avgStats.cnt, macroAvg)
        categoryNames.append('All - macro avg')
        update_statistics(statistics, avgStats.cnt, avgStats)
        categoryNames.append('All - datapoint avg')

        df = pd.DataFrame(data=statistics)
        df.index = categoryNames
        df.index.name = 'IPTC Category'
        return df

    predictions = model.classifyDataset(evalData, thr=evaluationConfig['thresholdPredict'], returnScores=True)
    dfCorpora = evaluateCorpuses(
        predictions,
        evalData.corpus,
        thr=evaluationConfig['thresholdEval'],
        catToThr=customThresholds,
        perCorpus=evaluationConfig['perCorpus'],
        averagingType=evaluationConfig['averagingType'],
    )
    dfClasses = evaluateClasses(
        predictions,
        evalData.corpus,
        thr=evaluationConfig['thresholdEval'],
        catToThr=customThresholds,
        perClass=evaluationConfig['perClass'],
    )
    return dfCorpora, dfClasses

