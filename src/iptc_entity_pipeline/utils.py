"""Utility helpers for the ClearML training pipeline."""

import logging
from itertools import product
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from clearml import Task

from iptc_entity_pipeline.dataset_builder import build_embedding_dataset
from iptc_entity_pipeline.legacy_reuse import createClassificationModel, trainClassificationModel

LOGGER = logging.getLogger(__name__)


def report_eval_scalars(*, logger: Any, title: str, row: Mapping[str, Any], iteration: int = 0) -> None:
    """Report shared eval metrics to ClearML scalar charts."""
    logger.report_scalar(title=title, series='Precision', value=row['Precision'], iteration=iteration)
    logger.report_scalar(title=title, series='Recall', value=row['Recall'], iteration=iteration)
    logger.report_scalar(title=title, series='F04', value=row['F04'], iteration=iteration)
    logger.report_scalar(title=title, series='F1', value=row['F1'], iteration=iteration)


def log_stage(*, task: Task, message: str, logging_config: Mapping[str, Any]) -> None:
    """Log pipeline stage both to logger and ClearML task text output."""
    LOGGER.info(message)
    task.get_logger().report_text(message, print_console=bool(logging_config['print_logs']))


def report_cv_std_scalars(*, logger: Any, row: Mapping[str, Any], iteration: int = 0) -> None:
    """Report standard deviations from CV objective metrics."""
    logger.report_scalar(
        title='Dev Cross Validation Std',
        series='Precision_std',
        value=float(row['Precision_std']),
        iteration=iteration,
    )
    logger.report_scalar(
        title='Dev Cross Validation Std',
        series='Recall_std',
        value=float(row['Recall_std']),
        iteration=iteration,
    )
    logger.report_scalar(
        title='Dev Cross Validation Std',
        series='F04_std',
        value=float(row['F04_std']),
        iteration=iteration,
    )
    logger.report_scalar(
        title='Dev Cross Validation Std',
        series='F1_std',
        value=float(row['F1_std']),
        iteration=iteration,
    )


def report_cv_outputs(*, task: Task, logger: Any, trials_df: Any, folds_df: Any, cv_dev_df: Any) -> None:
    """Report CV tables and upload them as ClearML artifacts."""
    logger.report_table(title='Cross Validation', series='Trials', iteration=0, table_plot=trials_df)
    logger.report_table(title='Cross Validation', series='Folds', iteration=0, table_plot=folds_df)
    logger.report_table(title='Cross Validation', series='Best Dev Mean+Std', iteration=0, table_plot=cv_dev_df)
    task.upload_artifact('cv_trials_dataframe', artifact_object=trials_df)
    task.upload_artifact('cv_folds_dataframe', artifact_object=folds_df)
    task.upload_artifact('cv_dev_summary_dataframe', artifact_object=cv_dev_df)


def to_numpy_array(*, matrix_like: Any) -> np.ndarray:
    """Convert ndarray-like object or tensor to numpy array."""
    if hasattr(matrix_like, 'detach') and hasattr(matrix_like, 'cpu'):
        return matrix_like.detach().cpu().numpy()
    return np.asarray(matrix_like)


def _require_corpus_cls() -> Any:
    from geneea.catlib.data import Corpus

    return Corpus


def _require_embedding_dataset_cls() -> Any:
    from geneea.catlib.vec.dataset import EmbeddingDataset

    return EmbeddingDataset


def _set_corpus_cat_list(*, corpus: Any, cat_list: Sequence[str]) -> None:
    try:
        setattr(corpus, 'catList', list(cat_list))
    except Exception:
        LOGGER.warning('Unable to set corpus.catList explicitly, keeping corpus defaults')


def _build_dataset_with_targets(
    *,
    corpus: Any,
    x_matrix: np.ndarray,
    y_matrix: np.ndarray,
    cat_list: Sequence[str] | None = None,
) -> Any:
    embedding_dataset_cls = _require_embedding_dataset_cls()
    if cat_list is not None:
        _set_corpus_cat_list(corpus=corpus, cat_list=cat_list)
    x_tensor = torch.as_tensor(x_matrix, dtype=torch.float32)
    y_tensor = torch.as_tensor(y_matrix, dtype=torch.float32)
    return embedding_dataset_cls(corpus, x_tensor, y_tensor)


def merge_datasets(*, left_data: Any, right_data: Any) -> Any:
    """Merge two embedding datasets by concatenating corpus docs and feature matrices."""
    corpus_cls = _require_corpus_cls()
    merged_docs = list(left_data.corpus) + list(right_data.corpus)
    merged_corpus = corpus_cls(doc for doc in merged_docs)
    merged_x = np.vstack([to_numpy_array(matrix_like=left_data.X), to_numpy_array(matrix_like=right_data.X)])
    if hasattr(left_data, 'Y') and hasattr(right_data, 'Y'):
        merged_y = np.vstack([to_numpy_array(matrix_like=left_data.Y), to_numpy_array(matrix_like=right_data.Y)])
        merged_cat_list = list(left_data.corpus.catList) if hasattr(left_data.corpus, 'catList') else None
        return _build_dataset_with_targets(
            corpus=merged_corpus,
            x_matrix=merged_x,
            y_matrix=merged_y,
            cat_list=merged_cat_list,
        )
    return build_embedding_dataset(corpus=merged_corpus, x_matrix=merged_x)


def slice_dataset(*, dataset: Any, indices: Sequence[int]) -> Any:
    """Return dataset subset by explicit positional indices."""
    corpus_cls = _require_corpus_cls()
    docs = list(dataset.corpus)
    x_matrix = to_numpy_array(matrix_like=dataset.X)
    selected_docs = [docs[idx] for idx in indices]
    selected_x = x_matrix[np.asarray(indices, dtype=np.int64)]
    selected_corpus = corpus_cls(doc for doc in selected_docs)
    if hasattr(dataset, 'Y'):
        y_matrix = to_numpy_array(matrix_like=dataset.Y)
        selected_y = y_matrix[np.asarray(indices, dtype=np.int64)]
        cat_list = list(dataset.corpus.catList) if hasattr(dataset.corpus, 'catList') else None
        return _build_dataset_with_targets(
            corpus=selected_corpus,
            x_matrix=selected_x,
            y_matrix=selected_y,
            cat_list=cat_list,
        )
    return build_embedding_dataset(corpus=selected_corpus, x_matrix=selected_x)


def model_payload(*, model_config: Mapping[str, Any]) -> dict[str, Any]:
    """Build model payload expected by legacy model factory."""
    return {
        'hiddenDim': int(model_config['hidden_dim']),
        'dropouts1': float(model_config['dropouts1']),
        'dropouts2': float(model_config['dropouts2']),
    }


def train_payload(*, training_config: Mapping[str, Any]) -> dict[str, Any]:
    """Build training payload expected by legacy training entrypoint."""
    return {
        'epochs': int(training_config['epochs']),
        'batchSize': int(training_config['batch_size']),
        'validationSplitName': str(training_config.get('validation_split_name', 'dev')),
        'earlyStoppingPatience': int(training_config.get('early_stopping_patience', 0)),
        'earlyStoppingMinDelta': float(training_config.get('early_stopping_min_delta', 0.0)),
        'optimizerConfig': {
            'name': training_config['optimizer_name'],
            'adamConfig': {'lr': float(training_config['learning_rate'])},
        },
        'lrSchedulerConfig': {
            'name': training_config['lr_scheduler_name'],
            'stepLRConfig': {
                'stepSize': int(training_config['step_size']),
                'gamma': float(training_config['gamma']),
            },
            'cosineAnnealingLRConfig': {'T_max': max(int(training_config['epochs']), 1)},
        },
        'lossConfig': {
            'name': training_config['loss_name'],
            'focalLossConfig': {'alpha': 0.25, 'gamma': 2.0},
        },
    }


def iter_param_grid(
    *,
    model_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Expand list-based model/training config into all trial combinations."""
    model_space = {
        'hidden_dim': [int(value) for value in model_config['hidden_dim']],
        'dropouts1': [float(value) for value in model_config['dropouts1']],
        'dropouts2': [float(value) for value in model_config['dropouts2']],
    }
    training_space = {
        'batch_size': [int(value) for value in training_config['batch_size']],
        'learning_rate': [float(value) for value in training_config['learning_rate']],
    }
    combos: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for model_values in product(
        model_space['hidden_dim'],
        model_space['dropouts1'],
        model_space['dropouts2'],
    ):
        for training_values in product(
            training_space['batch_size'],
            training_space['learning_rate'],
        ):
            combo_model_config = {
                'hidden_dim': model_values[0],
                'dropouts1': model_values[1],
                'dropouts2': model_values[2],
            }
            combo_training_config = dict(training_config)
            combo_training_config.update(
                {
                    'batch_size': training_values[0],
                    'learning_rate': training_values[1],
                }
            )
            combos.append((combo_model_config, combo_training_config))
    return combos


def train_model(
    *,
    train_data: Any,
    dev_data: Any,
    feature_dim: int,
    model_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    logging_config: Mapping[str, Any],
) -> Any:
    """Create and train the classification model on given fit/dev datasets."""
    if hasattr(train_data, 'Y'):
        out_dim = int(to_numpy_array(matrix_like=train_data.Y).shape[1])
    else:
        out_dim = int(train_data.corpus.catCnt)
    model = createClassificationModel(
        embDim=int(feature_dim),
        outDim=out_dim,
        modelConfig=model_payload(model_config=model_config),
        textVectorizer='not None',
        logConfig={'PRINT_LOGS': bool(logging_config['print_logs'])},
    )
    return trainClassificationModel(
        model=model,
        trainData=train_data,
        devData=dev_data,
        trainingConfig=train_payload(training_config=training_config),
        logConfig={'PRINT_LOGS': bool(logging_config['print_logs'])},
    )


def get_eval_config(*, evaluation_config: Mapping[str, Any]) -> dict[str, Any]:
    """Map external evaluation config to evaluator payload format."""
    return {
        'thresholdPredict': float(evaluation_config['threshold_predict']),
        'thresholdEval': float(evaluation_config['threshold_eval']),
        'perCorpus': bool(evaluation_config['per_corpus']),
        'perClass': bool(evaluation_config['per_class']),
        'averagingType': str(evaluation_config['averaging_type']),
    }


def get_obj_row(*, df_corpora: Any, objective_corpora: str, averaging_type: str) -> Mapping[str, Any]:
    """Return objective corpus row, fallback to all-corpora row for given averaging."""
    objective_row_name = f'All-{averaging_type}'
    if objective_corpora in df_corpora.index:
        return df_corpora.loc[objective_corpora].to_dict()
    return df_corpora.loc[objective_row_name].to_dict()
