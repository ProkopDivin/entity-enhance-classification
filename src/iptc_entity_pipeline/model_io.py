"""Persist trained model artifacts and build probability tables."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from clearml import Task

from iptc_entity_pipeline.config import EmbeddingConfig, EvaluationConfig
from iptc_entity_pipeline.data_loading import sanitize_name

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SavedModelPaths:
    """Filesystem paths for all artifacts produced by :func:`save_final_model_outputs`."""

    output_dir: str
    model_path: str
    test_embeddings_path: str
    config_yaml_path: str
    parameters_json_path: str
    probabilities_csv_path: str


def build_probability_dataframe(
    *,
    dataset: Any,
    pred_scores: Sequence[Any],
) -> Any:
    """Build per-article category probability table from model score outputs."""
    import pandas as pd

    categories = list(getattr(dataset.corpus, 'catList', []))
    rows: list[dict[str, Any]] = []
    for doc, doc_scores in zip(dataset.corpus, pred_scores):
        row = {
            'article_id': doc.id,
            'corpus_name': doc.metadata.get('corpusName', ''),
        }
        for cat_id in categories:
            raw_score = float(doc_scores[cat_id]) if cat_id in doc_scores else float('-inf')
            clipped_score = float(np.clip(raw_score, -50.0, 50.0))
            row[f'prob_{cat_id}'] = float(1.0 / (1.0 + np.exp(-clipped_score)))
        rows.append(row)
    return pd.DataFrame(rows)


def save_final_model_outputs(
    *,
    model: Any,
    test_data: Any,
    pred_scores: Sequence[Any],
    evaluation_config: EvaluationConfig,
    embedding_config: EmbeddingConfig,
    config_mapping: Mapping[str, Any],
    config_name: str,
    feature_dim: int,
) -> SavedModelPaths:
    """
    Save final model bundle and test probability CSV to disk and ClearML.

    :param model: Trained model.
    :param test_data: Test dataset.
    :param pred_scores: Prediction scores for each test document.
    :param evaluation_config: Typed evaluation config.
    :param embedding_config: Typed embedding config.
    :param config_mapping: Full serialized config dict (written to JSON as-is).
    :param config_name: Config variant name.
    :param feature_dim: Input feature dimensionality.
    :return: Paths to all saved artifacts.
    """
    import yaml

    from iptc_entity_pipeline.evaluate import get_iptc_topics

    def get_cat_name(*, cat_id: str) -> str:
        try:
            return get_iptc_topics().getCategory(cat_id).getLongName(abbreviate=True, shorten=True)
        except KeyError:
            return cat_id

    task = Task.current_task()
    logger = task.get_logger()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('results') / 'saved_models' / f'{sanitize_name(value=config_name)}_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / 'model.nn.bin'
    model.save(str(model_path))

    test_embeddings_path = output_dir / 'test_embeddings.tsv'
    test_data.saveEmbeds(str(test_embeddings_path))

    probabilities_df = build_probability_dataframe(
        dataset=test_data,
        pred_scores=pred_scores,
    )
    probabilities_csv_path = output_dir / 'test_article_probabilities.csv'
    probabilities_df.to_csv(probabilities_csv_path, index=False)

    selected_cat_ids = list(getattr(model, 'catList', []))
    config_data = {
        'testEmbeddingPath': str(test_embeddings_path),
        'embedSvcModelId': embedding_config.article_model_name,
        'embedDim': int(feature_dim),
        'nnModelPath': str(model_path),
        'threshold': evaluation_config.threshold_eval,
        'thresholdByTopic': {},
        'selectedTopics': [get_cat_name(cat_id=cat_id) for cat_id in selected_cat_ids],
    }

    config_yaml_path = output_dir / 'iptc.config.yaml'
    with open(config_yaml_path, 'w', encoding='utf-8') as file:
        yaml.safe_dump(config_data, file, default_flow_style=False, sort_keys=False)

    parameters_json_path = output_dir / 'pipeline_parameters.json'
    with open(parameters_json_path, 'w', encoding='utf-8') as file:
        json.dump(config_mapping, file, indent=4)

    logger.report_text(f'Saved final model bundle to {output_dir}')
    logger.report_text(f'Saved test probability CSV to {probabilities_csv_path}')

    task.upload_artifact('saved_model_file', artifact_object=str(model_path))
    task.upload_artifact('saved_model_test_embeddings', artifact_object=str(test_embeddings_path))
    task.upload_artifact('saved_model_config_yaml', artifact_object=str(config_yaml_path))
    task.upload_artifact('saved_model_parameters_json', artifact_object=str(parameters_json_path))
    task.upload_artifact('saved_model_test_probabilities_csv', artifact_object=str(probabilities_csv_path))

    return SavedModelPaths(
        output_dir=str(output_dir),
        model_path=str(model_path),
        test_embeddings_path=str(test_embeddings_path),
        config_yaml_path=str(config_yaml_path),
        parameters_json_path=str(parameters_json_path),
        probabilities_csv_path=str(probabilities_csv_path),
    )
