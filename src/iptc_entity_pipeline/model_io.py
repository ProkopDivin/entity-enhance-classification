"""Persist trained model artifacts and raw prediction score outputs."""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from clearml import Task

from iptc_entity_pipeline.config import EmbeddingCnf, EvaluationCnf
from iptc_entity_pipeline.data_loading import sanitize_name

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_FILENAME = 'predictions.pkl'
EVAL_CORPUS_FILENAME = 'eval_corpus.pkl'


@dataclass(frozen=True)
class SavedModelPaths:
    """Filesystem paths for all artifacts produced by :func:`save_final_model_outputs`."""

    output_dir: str
    model_path: str
    test_embeddings_path: str
    config_yaml_path: str
    parameters_json_path: str
    predictions_path: str
    eval_corpus_path: str


def save_outputs(
    *,
    model: Any,
    test_data: Any,
    pred_scores: Sequence[Any],
    eval_cnf: EvaluationCnf,
    emb_cnf: EmbeddingCnf,
    config_mapping: Mapping[str, Any],
    config_name: str,
    feature_dim: int,
    upload_artifacts: bool = False,
) -> SavedModelPaths:
    """
    Save final model bundle, raw prediction scores and eval corpus to disk and ClearML.

    :param model: Trained model.
    :param test_data: Test dataset.
    :param pred_scores: Raw prediction scores for each test document
        (``list[list[tuple[str, float]]]`` aligned with ``test_data.corpus``).
    :param eval_cnf: Typed evaluation config.
    :param emb_cnf: Typed embedding config.
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
    output_dir = PROJECT_ROOT / 'results' / 'saved_models' / f'{sanitize_name(value=config_name)}_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / 'model.nn.bin'
    model.save(str(model_path))

    test_embeddings_path = output_dir / 'test_embeddings.tsv'
    test_data.saveEmbeds(str(test_embeddings_path))

    predictions_path = output_dir / PREDICTIONS_FILENAME
    with open(predictions_path, 'wb') as f:
        pickle.dump(list(pred_scores), f)

    eval_corpus_path = output_dir / EVAL_CORPUS_FILENAME
    with open(eval_corpus_path, 'wb') as f:
        pickle.dump(test_data.corpus, f)

    selected_cat_ids = list(getattr(model, 'catList', []))
    config_data = {
        'testEmbeddingPath': str(test_embeddings_path),
        'embedSvcModelId': emb_cnf.article_model_name,
        'embedDim': int(feature_dim),
        'nnModelPath': str(model_path),
        'threshold': eval_cnf.threshold_eval,
        'thresholdByTopic': {},
        'selectedTopics': [get_cat_name(cat_id=cat_id) for cat_id in selected_cat_ids],
    }

    config_yaml_path = output_dir / 'iptc.config.yaml'
    with open(config_yaml_path, 'w', encoding='utf-8') as file:
        yaml.safe_dump(config_data, file, default_flow_style=False, sort_keys=False)

    parameters_json_path = output_dir / 'pipeline_parameters.json'
    with open(parameters_json_path, 'w', encoding='utf-8') as file:
        json.dump(config_mapping, file, indent=4)

    logger.report_text(f'Saved final model bundle to {output_dir.resolve()}')
    logger.report_text(f'Saved raw predictions to {predictions_path.resolve()}')
    logger.report_text(f'Saved eval corpus to {eval_corpus_path.resolve()}')

    if upload_artifacts:
        task.upload_artifact('saved_model_file', artifact_object=str(model_path))
        task.upload_artifact('saved_model_test_embeddings', artifact_object=str(test_embeddings_path))
        task.upload_artifact('saved_model_config_yaml', artifact_object=str(config_yaml_path))
        task.upload_artifact('saved_model_parameters_json', artifact_object=str(parameters_json_path))
        task.upload_artifact('saved_model_predictions', artifact_object=str(predictions_path))
        task.upload_artifact('saved_model_eval_corpus', artifact_object=str(eval_corpus_path))

    return SavedModelPaths(
        output_dir=str(output_dir),
        model_path=str(model_path),
        test_embeddings_path=str(test_embeddings_path),
        config_yaml_path=str(config_yaml_path),
        parameters_json_path=str(parameters_json_path),
        predictions_path=str(predictions_path),
        eval_corpus_path=str(eval_corpus_path),
    )


def export_eval_excel(
    *,
    excel_path: Path,
    cv_dev_df: pd.DataFrame,
    df_corpora_test: pd.DataFrame,
    df_classes_test: pd.DataFrame,
    comparison_result: Any = None,
) -> None:
    """Write evaluation tables (and optional comparison) to a single Excel workbook.

    :param excel_path: Target ``.xlsx`` path.
    :param cv_dev_df: Best-trial CV dev summary.
    :param df_corpora_test: Test corpora metrics.
    :param df_classes_test: Test per-class metrics.
    :param comparison_result: Optional ``ComparisonResult`` from baseline comparison.
    """
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(excel_path) as writer:
        cv_dev_df.to_excel(excel_writer=writer, sheet_name='dev_cv_summary')
        df_corpora_test.to_excel(excel_writer=writer, sheet_name='test_corpora')
        df_classes_test.to_excel(excel_writer=writer, sheet_name='test_classes')
        if comparison_result is not None:
            comparison_result.corpora_comparison.to_excel(
                excel_writer=writer, sheet_name='comparison_corpora', index=False,
            )
            comparison_result.classes_comparison.to_excel(
                excel_writer=writer, sheet_name='comparison_classes', index=False,
            )
            comparison_result.summary_comparison.to_excel(
                excel_writer=writer, sheet_name='comparison_summary', index=False,
            )
            comparison_result.top_improved_categories.to_excel(
                excel_writer=writer, sheet_name='comparison_top_up', index=False,
            )
            comparison_result.top_degraded_categories.to_excel(
                excel_writer=writer, sheet_name='comparison_top_down', index=False,
            )
            comparison_result.hamming_loss_comparison.to_excel(
                excel_writer=writer, sheet_name='comparison_hamming', index=False,
            )
            comparison_result.pr_auc_per_class.to_excel(
                excel_writer=writer, sheet_name='comparison_pr_auc', index=False,
            )
            comparison_result.pr_auc_summary.to_excel(
                excel_writer=writer, sheet_name='comparison_pr_auc_sum', index=False,
            )
    LOGGER.info(f'Saved final evaluation tables to {excel_path}')
