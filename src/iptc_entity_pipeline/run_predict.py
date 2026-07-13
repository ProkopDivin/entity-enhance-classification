'''
Run IPTC predictions on the debug test corpus for a saved model.
'''

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from geneea.catlib.model.nnet import NeuralCategModel
from geneea.core import logutil

from iptc_entity_pipeline.article_embeddings import ArticleEmbeddingProvider
from iptc_entity_pipeline.build_dataset import build_article_only_matrix, get_pooling
from iptc_entity_pipeline.config import BaseCnf, EmbeddingCnf, get_config, conf_from_dict
from iptc_entity_pipeline.data_loading import attach_entities, load_and_normalize, load_wdid_map
from iptc_entity_pipeline.dataset_builder import build_emb_data, build_ragged_emb_data
from iptc_entity_pipeline.entity_embeddings import EntityEmbeddingStore
from iptc_entity_pipeline.evaluation.comparison import load_custom_thresholds
from iptc_entity_pipeline.evaluation.evaluate import get_cat_name, pred_cats_from_matrix
from iptc_entity_pipeline.feature_builder import FeatureBuilder
from iptc_entity_pipeline.legacy_reuse import predict_score_matrix

LOG = logutil.getLogger(__package__, __file__)

DEBUG_CONFIG_NAME = 'debug'


def _resolve_run_dir(*, model_path: Path) -> Path:
    '''
    Resolve the saved-run directory that may contain sidecar artifacts.

    :param model_path: Model file or saved-run directory.
    :return: Directory containing thresholds/parameters artifacts.
    '''
    return model_path if model_path.is_dir() else model_path.parent


def _load_emb_config(*, model_path: Path, cfg: BaseCnf) -> EmbeddingCnf:
    '''
    Load embedding config for feature building.

    Uses ``pipeline_parameters.json`` next to the model when available so
    attention / pooling settings match training. Paths still come from the
    hardcoded debug config.

    :param model_path: Model file or saved-run directory.
    :param cfg: Debug pipeline configuration.
    :return: Embedding configuration for inference.
    '''
    params_path = _resolve_run_dir(model_path=model_path) / 'pipeline_parameters.json'
    if not params_path.is_file():
        return cfg.emb
    with params_path.open(encoding='utf-8') as src:
        payload = json.load(src)
    emb_payload = payload.get('emb')
    if not isinstance(emb_payload, Mapping):
        LOG.warning(f'No emb block in {params_path}; using debug embedding config')
        return cfg.emb
    LOG.info(f'Using embedding config from {params_path.resolve()}')
    return conf_from_dict(EmbeddingCnf, emb_payload)


def _load_debug_corpora(*, cfg: BaseCnf, emb: EmbeddingCnf):
    '''
    Load train/test corpora with entities attached using the debug config paths.

    :param cfg: Debug pipeline configuration.
    :param emb: Embedding configuration used for entity attachment/filtering.
    :return: Corpus group with linked entities on train and test splits.
    '''
    paths = cfg.paths
    corpora = load_and_normalize(
        train_csv=paths.train_csv,
        test_csv=paths.test_csv,
        removed_cat_ids=paths.removed_cat_ids,
    )
    wdid_mapping = load_wdid_map(wdid_mapping_tsv=paths.wdid_mapping_tsv)
    attach_entities(
        corpus=corpora.train,
        csv_path=paths.train_csv,
        wdid_mapping=wdid_mapping,
        min_relevance=emb.entity_relevance_threshold,
        remove_types=emb.remove_types,
    )
    attach_entities(
        corpus=corpora.test,
        csv_path=paths.test_csv,
        wdid_mapping=wdid_mapping,
        min_relevance=emb.entity_relevance_threshold,
        remove_types=emb.remove_types,
    )
    return corpora


def _build_test_dataset(*, corpora, cfg: BaseCnf, emb: EmbeddingCnf):
    '''
    Build an embedding dataset for the debug test split.

    :param corpora: Loaded corpus group.
    :param cfg: Debug pipeline configuration.
    :param emb: Embedding configuration for feature construction.
    :return: Test embedding dataset ready for model inference.
    '''
    paths = cfg.paths

    if not emb.use_article_embeddings and not emb.use_entity_embeddings:
        raise ValueError('Invalid embedding config: both article and entity embeddings are disabled')

    article_provider = None
    if emb.use_article_embeddings:
        article_provider = ArticleEmbeddingProvider(
            embeddings_dir=paths.article_embeddings_dir,
            model_name=emb.article_model_name,
            embed_svc_url=emb.embed_svc_url,
            embedding_dim=emb.article_embedding_dim,
        )
        article_provider.prepare_embeddings(corpus=corpora.test)

    if not emb.use_entity_embeddings:
        assert article_provider is not None
        x_test = build_article_only_matrix(
            split_corpus=corpora.test,
            split_name='test',
            article_provider=article_provider,
            logger=LOG,
        )
        return build_emb_data(corpus=corpora.test, x_matrix=x_test)

    selected_langs = tuple(emb.entity_langs) if emb.entity_langs else (emb.entity_lang,)
    entity_store = EntityEmbeddingStore(
        root_dir=paths.entity_embeddings_dir,
        langs=selected_langs,
        lang_mode=emb.entity_lang_mode,
    )
    entity_store.compute_train_mean_from_corpus(corpus=corpora.train)
    pooling = get_pooling(emb_cfg=emb, logger=LOG)
    builder = FeatureBuilder(
        article_embedding_provider=article_provider,
        entity_embedding_store=entity_store,
        pooling_strategy=pooling,
        use_article_embeddings=emb.use_article_embeddings,
        combine_method=emb.combine_method,
    )

    if emb.entity_pooling == 'no_pooling':
        test_ragged = builder.build_ragged_features(corpus=corpora.test)
        return build_ragged_emb_data(
            corpus=corpora.test,
            article_matrix=test_ragged.article_matrix,
            entity_matrices=test_ragged.entity_matrices,
        )

    x_test = builder.build_features(corpus=corpora.test)
    return build_emb_data(corpus=corpora.test, x_matrix=x_test)


def _load_model(*, model_path: Path) -> NeuralCategModel:
    '''
    Load a trained classifier from disk.

    :param model_path: Path to ``model.nn.bin`` or a saved-run directory.
    :return: Loaded neural categorization model.
    '''
    if model_path.is_dir():
        candidate = model_path / 'model.nn.bin'
        if not candidate.is_file():
            raise FileNotFoundError(f'No model.nn.bin found in directory: {model_path}')
        model_path = candidate
    if not model_path.is_file():
        raise FileNotFoundError(f'Model file does not exist: {model_path}')
    LOG.info(f'Loading model from {model_path.resolve()}')
    return NeuralCategModel.load(str(model_path))


def _resolve_thresholds(*, model_path: Path, cfg: BaseCnf, threshold: float | None) -> tuple[float, dict[str, float] | None]:
    '''
    Resolve scalar and per-class thresholds for prediction.

    :param model_path: Model file or saved-run directory.
    :param cfg: Debug pipeline configuration.
    :param threshold: Optional CLI override for the scalar threshold.
    :return: Scalar threshold and optional per-class threshold map.
    '''
    run_dir = _resolve_run_dir(model_path=model_path)
    custom_thresholds = load_custom_thresholds(run_dir=run_dir) or None
    scalar_threshold = cfg.eval.threshold_eval if threshold is None else float(threshold)
    return scalar_threshold, custom_thresholds


def _predict_labels(
    *,
    model: NeuralCategModel,
    test_data,
    cat_list: Sequence[str],
    threshold: float,
    custom_thresholds: Mapping[str, float] | None,
) -> list[list[str]]:
    '''
    Predict normalized IPTC category ids for each test document.

    :param model: Trained model.
    :param test_data: Test embedding dataset.
    :param cat_list: Category ids aligned with model output columns.
    :param threshold: Fallback decision threshold.
    :param custom_thresholds: Optional per-class threshold overrides.
    :return: Per-document predicted category id lists.
    '''
    score_matrix = predict_score_matrix(model=model, eval_data=test_data)
    return pred_cats_from_matrix(
        score_matrix=score_matrix,
        cat_list=cat_list,
        threshold=threshold,
        cat_to_thr=custom_thresholds,
    )


def _format_cat_labels(*, cat_ids: Sequence[str]) -> str:
    '''
    Format category ids as human-readable labels.

    :param cat_ids: Predicted IPTC category ids.
    :return: Semicolon-separated label string.
    '''
    return '; '.join(get_cat_name(cat_id=cat_id) for cat_id in cat_ids)


def predict_debug_corpus(
    *,
    model_path: Path,
    threshold: float | None = None,
    include_gold: bool = True,
) -> list[dict[str, str]]:
    '''
    Load a model and predict labels for articles in the debug test corpus.

    :param model_path: Path to ``model.nn.bin`` or a saved-run directory.
    :param threshold: Optional scalar threshold override.
    :param include_gold: Whether to include gold labels in returned rows.
    :return: List of per-article prediction records.
    '''
    cfg = get_config(config_name=DEBUG_CONFIG_NAME)
    emb = _load_emb_config(model_path=model_path, cfg=cfg)
    corpora = _load_debug_corpora(cfg=cfg, emb=emb)
    test_data = _build_test_dataset(corpora=corpora, cfg=cfg, emb=emb)

    model = _load_model(model_path=model_path)
    cat_list = list(model.catList) if model.catList is not None else list(test_data.corpus.catList)
    scalar_threshold, custom_thresholds = _resolve_thresholds(
        model_path=model_path,
        cfg=cfg,
        threshold=threshold,
    )
    LOG.info(
        f'Predicting debug test corpus: docs={len(test_data.corpus)}, '
        f'threshold={scalar_threshold}, custom_thresholds='
        f'{len(custom_thresholds) if custom_thresholds else 0}'
    )
    pred_cats = _predict_labels(
        model=model,
        test_data=test_data,
        cat_list=cat_list,
        threshold=scalar_threshold,
        custom_thresholds=custom_thresholds,
    )

    rows: list[dict[str, str]] = []
    for doc, cats in zip(test_data.corpus, pred_cats, strict=True):
        row = {
            'doc_id': str(doc.id),
            'predicted_cat_ids': ';'.join(cats),
            'predicted_labels': _format_cat_labels(cat_ids=cats),
        }
        if include_gold:
            gold_ids = list(doc.cats)
            row['gold_cat_ids'] = ';'.join(gold_ids)
            row['gold_labels'] = _format_cat_labels(cat_ids=gold_ids)
        rows.append(row)
    return rows


def main() -> None:
    '''Run article prediction for a saved model on the debug test corpus.'''
    argparser = argparse.ArgumentParser(
        description='Predict IPTC labels for debug test articles using a saved model.',
    )
    argparser.add_argument(
        '-m',
        '--model',
        required=True,
        help='Path to model.nn.bin or a saved-run directory containing it.',
    )
    argparser.add_argument(
        '-o',
        '--output',
        help='Optional output TSV path. Defaults to stdout.',
    )
    argparser.add_argument(
        '--threshold',
        type=float,
        help='Scalar decision threshold. Defaults to debug config threshold_eval.',
    )
    argparser.add_argument(
        '--no-gold',
        action='store_true',
        help='Omit gold labels from the output table.',
    )
    argparser.add_argument(
        '--json',
        action='store_true',
        help='Write JSON instead of TSV.',
    )
    logutil.addLogArguments(argparser)
    args = argparser.parse_args()
    logutil.configureFromArgs(args)

    model_path = Path(args.model)
    rows = predict_debug_corpus(
        model_path=model_path,
        threshold=args.threshold,
        include_gold=not args.no_gold,
    )

    if args.json:
        payload = json.dumps(rows, ensure_ascii=False, indent=2)
        if args.output:
            Path(args.output).write_text(payload + '\n', encoding='utf-8')
        else:
            sys.stdout.write(payload + '\n')
        return

    fieldnames = ['doc_id', 'predicted_cat_ids', 'predicted_labels']
    if not args.no_gold:
        fieldnames.extend(['gold_cat_ids', 'gold_labels'])

    with open(args.output, 'w', encoding='utf-8', newline='') if args.output else sys.stdout as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


if __name__ == '__main__':
    main()
