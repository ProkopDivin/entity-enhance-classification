'''Run loading, entity parsing, and per-article alignment helpers.'''

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from iptc_entity_pipeline.config import EvaluationCnf
from iptc_entity_pipeline.evaluation.evaluate import (
    REMOVED_CAT_IDS,
    evaluate_predictions,
    get_iptc_topics,
)
from iptc_entity_pipeline.model_io import EVAL_CORPUS_FILENAME, PREDICTIONS_FILENAME

LOG = logging.getLogger(__name__)

# Supported on-disk filenames carrying per-class thresholds for a saved run.
# ``custom_thresholds.json`` matches the legacy IPTC pipeline convention; the
# alternative ``thresholds.json`` is kept for backward compatibility with runs
# saved by earlier versions of this pipeline before the rename.
THRESHOLD_FILENAMES: tuple[str, ...] = ('custom_thresholds.json', 'thresholds.json')


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GoldArticle:
    '''Gold metadata for one article.'''

    article_id: str
    corpus_name: str
    gold_categories: tuple[str, ...]


@dataclass(frozen=True)
class GoldLabelMap:
    '''Gold labels loaded once and reused across all comparisons.'''

    df: pd.DataFrame
    article_map: Mapping[str, GoldArticle]

    @classmethod
    def from_corpus(cls, *, corpus: Any) -> 'GoldLabelMap':
        '''Build a gold label map from a ``geneea.catlib.data.Corpus``.'''
        df = gold_df_from_corpus(corpus=corpus)
        article_map = {
            row.article_id: GoldArticle(
                article_id=row.article_id,
                corpus_name=row.corpus_name,
                gold_categories=row.gold_categories,
            )
            for row in df.itertuples(index=False)
        }
        return cls(df=df, article_map=article_map)

    def cat_ids(self, *, prob_dfs: Sequence[pd.DataFrame]) -> list[str]:
        '''Collect category ids from gold labels plus all probability tables.'''
        cat_ids = {cat_id for gold in self.article_map.values() for cat_id in gold.gold_categories}
        for prob_df in prob_dfs:
            cat_ids.update(col.removeprefix('prob_') for col in prob_df.columns if col.startswith('prob_'))
        return sorted(cat_ids)

    def gold_matrix(self, *, article_ids: Sequence[str], cat_ids: Sequence[str]) -> np.ndarray:
        '''Build gold binary matrix for selected articles and categories.'''
        cat_to_idx = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}
        matrix = np.zeros((len(article_ids), len(cat_ids)), dtype=np.int8)
        for row_idx, article_id in enumerate(article_ids):
            gold = self.article_map.get(article_id)
            if gold is None:
                continue
            for cat_id in gold.gold_categories:
                cat_idx = cat_to_idx.get(cat_id)
                if cat_idx is not None:
                    matrix[row_idx, cat_idx] = 1
        return matrix


@dataclass(frozen=True)
class RunEval:
    '''One run after table rebuild and per-article alignment.'''

    aligned_df: pd.DataFrame
    corpora_df: pd.DataFrame
    classes_df: pd.DataFrame


@dataclass(frozen=True)
class ArticleEntity:
    '''One normalized entity mention attached to an article.'''

    gkb_id: str | None
    wdids: tuple[str, ...]
    entity_type: str | None
    std_form: str | None
    relevance: float | None
    mention_count: int | None


@dataclass(frozen=True)
class ArticleEvalRecord:
    '''Normalized article payload used before DataFrame conversion.'''

    article_id: str
    corpus_name: str
    gold_categories: tuple[str, ...]
    pred_scores: tuple[tuple[str, float], ...]
    article_text: str | None
    article_length: int | None
    entities: tuple[ArticleEntity, ...]


# ---------------------------------------------------------------------------
# Run loading
# ---------------------------------------------------------------------------

def load_run(*, run_dir: str | Path) -> tuple[list[list[tuple[str, float]]], Any]:
    '''Load raw predictions and eval corpus from a saved run directory.

    :param run_dir: Directory containing ``predictions.pkl`` and ``eval_corpus.pkl``.
    :return: ``(pred_scores, eval_corpus)`` tuple aligned positionally.
    '''
    run_path = Path(run_dir)
    predictions_path = run_path / PREDICTIONS_FILENAME
    eval_corpus_path = run_path / EVAL_CORPUS_FILENAME
    with open(predictions_path, 'rb') as f:
        pred_scores = pickle.load(f)
    with open(eval_corpus_path, 'rb') as f:
        eval_corpus = pickle.load(f)
    if len(pred_scores) != len(eval_corpus):
        raise ValueError(
            f'Misaligned run {run_path}: predictions={len(pred_scores)} corpus={len(eval_corpus)}'
        )
    return pred_scores, eval_corpus


def load_custom_thresholds(*, run_dir: str | Path) -> dict[str, float]:
    '''Load per-class thresholds from a saved-run directory.

    Tries each filename in :data:`THRESHOLD_FILENAMES` in order and returns
    the first successfully parsed JSON object as a ``{cat_id: threshold}``
    mapping. Returns an empty dict if no file is present or the file is not
    a JSON object.

    :param run_dir: Saved-run directory possibly containing a thresholds JSON.
    :return: Mapping ``category_id -> threshold`` (empty when not available).
    '''
    run_path = Path(run_dir)
    for filename in THRESHOLD_FILENAMES:
        candidate = run_path / filename
        if not candidate.is_file():
            continue
        with open(candidate, encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, Mapping):
            LOG.warning(f'Custom thresholds file is not a JSON object: {candidate}')
            continue
        thresholds = {str(cid): float(thr) for cid, thr in data.items()}
        LOG.info(f'Loaded {len(thresholds)} per-class thresholds from {candidate}')
        return thresholds
    return {}


def thresholds_vector(
    *,
    cat_ids: Sequence[str],
    cat_to_thr: Mapping[str, float] | None,
    default_threshold: float,
) -> np.ndarray:
    '''Build a per-class threshold vector aligned with ``cat_ids``.

    :param cat_ids: Category id sequence defining the column order.
    :param cat_to_thr: Optional ``cat_id -> threshold`` map.
    :param default_threshold: Fallback threshold for unmapped classes.
    :return: 1-D ``float64`` array of shape ``(len(cat_ids),)``.
    '''
    if not cat_to_thr:
        return np.full(len(cat_ids), float(default_threshold), dtype=float)
    return np.asarray(
        [float(cat_to_thr.get(cid, default_threshold)) for cid in cat_ids],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------

def build_run(
    *,
    pred_scores: Sequence[Any],
    eval_corpus: Any,
    evaluation_config: EvaluationCnf,
    cat_to_thr: Mapping[str, float] | None = None,
) -> RunEval:
    '''Evaluate one run and build its aligned per-article dataframe.

    :param cat_to_thr: Optional per-class thresholds applied during the
        F1/precision/recall table computation.
    '''
    corpora_df, classes_df = evaluate_predictions(
        pred_wgh_cats=pred_scores,
        eval_corpus=eval_corpus,
        evaluation_config=evaluation_config,
        cat_to_thr=cat_to_thr,
    )
    aligned_df = build_aligned_df(eval_corpus=eval_corpus, pred_scores=pred_scores)
    return RunEval(aligned_df=aligned_df, corpora_df=corpora_df, classes_df=classes_df)


def build_aligned_df(*, eval_corpus: Any, pred_scores: Sequence[Any]) -> pd.DataFrame:
    '''Build per-article dataframe with gold categories and dense ``prob_*`` columns.'''
    records = build_article_records(eval_corpus=eval_corpus, pred_scores=pred_scores)
    return records_to_df(records=records)


def build_article_records(*, eval_corpus: Any, pred_scores: Sequence[Any]) -> list[ArticleEvalRecord]:
    '''Build normalized article records from corpus docs and prediction tuples.'''
    if len(pred_scores) != len(eval_corpus):
        raise ValueError(
            f'Misaligned predictions: predictions={len(pred_scores)} corpus={len(eval_corpus)}'
        )
    records: list[ArticleEvalRecord] = []
    for doc, doc_pred in zip(eval_corpus, pred_scores):
        metadata = _doc_metadata(doc=doc)
        records.append(
            ArticleEvalRecord(
                article_id=str(doc.id),
                corpus_name=str(metadata.get('corpusName', '')),
                gold_categories=tuple(sorted(norm_cat_ids(cat_ids=doc.cats))),
                pred_scores=tuple((str(cat_id), float(score)) for cat_id, score in doc_pred),
                article_text=_extract_article_text(doc=doc),
                article_length=_extract_article_length(doc=doc, metadata=metadata),
                entities=_extract_entities(doc=doc, metadata=metadata),
            )
        )
    return records


def records_to_df(*, records: Sequence[ArticleEvalRecord]) -> pd.DataFrame:
    '''Convert article records to the aligned DataFrame used by metric code.'''
    rows = [
        {
            'article_id': record.article_id,
            'corpus_name': record.corpus_name,
            'gold_categories': record.gold_categories,
            'pred_scores': list(record.pred_scores),
            'article_text': record.article_text,
            'article_length': record.article_length,
            'entities': list(record.entities),
        }
        for record in records
    ]
    base_df = pd.DataFrame(rows)
    return add_prob_columns(df=base_df)


# ---------------------------------------------------------------------------
# Entity parsing helpers
# ---------------------------------------------------------------------------

def _doc_metadata(*, doc: Any) -> Mapping[str, Any]:
    '''Return document metadata mapping or an empty mapping.'''
    metadata = getattr(doc, 'metadata', None)
    return metadata if isinstance(metadata, Mapping) else {}


def _extract_article_text(*, doc: Any) -> str | None:
    '''Extract article text from document text payload.'''
    text = getattr(doc, 'text', None)
    if isinstance(text, str) and text:
        return text
    return None


def _extract_article_length(*, doc: Any, metadata: Mapping[str, Any]) -> int | None:
    '''Extract known typed article length if available.'''
    for key in ('article_length', 'articleLength', 'length'):
        value = _mapping_value(item=metadata, keys=(key,))
        converted = _as_int(value=value)
        if converted is not None:
            return converted

    text = getattr(doc, 'text', None)
    if isinstance(text, str) and text:
        return len(text)
    return None


def _extract_entities(*, doc: Any, metadata: Mapping[str, Any]) -> tuple[ArticleEntity, ...]:
    '''Extract normalized entities from doc attributes or metadata.'''
    raw_entities = getattr(doc, 'entities', None)
    if raw_entities is None:
        raw_entities = _mapping_value(item=metadata, keys=('entities', 'article_entities', 'entity_list'))
    if not isinstance(raw_entities, Sequence) or isinstance(raw_entities, (str, bytes)):
        return ()
    entities = [_parse_entity(item=item) for item in raw_entities]
    return tuple(entity for entity in entities if entity is not None)


def _parse_entity(*, item: Any) -> ArticleEntity | None:
    '''Normalize entity from known source shapes (raw CSV dict or LinkedEntity).'''
    raw_payload: Mapping[str, Any] | None = None

    if isinstance(item, Mapping):
        raw_payload = item
        gkb_raw = raw_payload.get('gkbId')
        wdids_raw = raw_payload.get('wdid') or raw_payload.get('wdids')
        entity_type_raw = raw_payload.get('type')
        std_form_raw = raw_payload.get('stdForm')
        relevance_raw = raw_payload.get('relevance')
        mentions_raw = raw_payload.get('mentions')
    else:
        gkb_raw = getattr(item, 'gkb_id', None)
        wdids_raw = getattr(item, 'wd_ids', None)
        relevance_raw = getattr(item, 'relevance', None)
        mentions_raw = getattr(item, 'mention_count', None)
        entity_type_raw = getattr(item, 'entity_type', None)
        raw_maybe = getattr(item, 'raw_entity', None)
        if isinstance(raw_maybe, Mapping):
            raw_payload = raw_maybe
        if entity_type_raw is None and raw_payload is not None:
            entity_type_raw = raw_payload.get('type')
        std_form_raw = raw_payload.get('stdForm') if raw_payload is not None else None

    if relevance_raw is None and raw_payload is not None:
        feats_raw = raw_payload.get('feats')
        if isinstance(feats_raw, Mapping):
            relevance_raw = feats_raw.get('relevance')
    if mentions_raw is None and raw_payload is not None:
        mentions_raw = raw_payload.get('mentions')
    if raw_payload is not None:
        raw_type = _mapping_value(item=raw_payload, keys=('type', 'entityType'))
        if _normalize_entity_type(value=entity_type_raw) in {None, 'other'}:
            entity_type_raw = raw_type

    gkb_id = _as_str(value=gkb_raw)
    wdids = _as_wdid_tuple(value=wdids_raw)
    entity_type = _normalize_entity_type(value=entity_type_raw)
    std_form = _as_str(value=std_form_raw)
    relevance = _as_float(value=relevance_raw)
    mention_count = _as_mentions_count(value=mentions_raw)

    if (
        gkb_id is None
        and not wdids
        and entity_type is None
        and std_form is None
        and relevance is None
        and mention_count is None
    ):
        return None
    return ArticleEntity(
        gkb_id=gkb_id,
        wdids=wdids,
        entity_type=entity_type,
        std_form=std_form,
        relevance=relevance,
        mention_count=mention_count,
    )


def _mapping_value(*, item: Mapping[str, Any], keys: Sequence[str]) -> Any:
    '''Return first present mapping key value.'''
    for key in keys:
        if key in item:
            return item[key]
    return None


def _normalize_entity_type(*, value: Any) -> str | None:
    '''Normalize entity type to canonical lower-case labels.'''
    if value is None:
        return None
    if isinstance(value, Enum):
        enum_value = getattr(value, 'value', None)
        if enum_value is not None:
            return _normalize_entity_type(value=enum_value)
        return _normalize_entity_type(value=value.name)
    value_str = _as_str(value=value)
    if value_str is None:
        return None
    normalized = value_str.strip()
    if normalized.startswith('EntityType.'):
        normalized = normalized.split('.', 1)[1]
    normalized = normalized.lower()
    return normalized or None


def _as_wdid_tuple(*, value: Any) -> tuple[str, ...]:
    '''Normalize one-or-many Wikidata ids to an immutable tuple.'''
    if value is None:
        return ()

    wdids: list[str] = []
    if isinstance(value, str):
        normalized = value.replace('|', ',')
        chunks = [chunk.strip() for chunk in normalized.split(',')]
        wdids.extend(chunk for chunk in chunks if chunk)
    elif isinstance(value, Sequence):
        for item in value:
            normalized = _as_str(value=item)
            if normalized:
                wdids.append(normalized)
    else:
        normalized = _as_str(value=value)
        if normalized:
            wdids.append(normalized)

    seen: set[str] = set()
    unique_wdids = []
    for item in wdids:
        if item in seen:
            continue
        seen.add(item)
        unique_wdids.append(item)
    return tuple(unique_wdids)


def _as_float(*, value: Any) -> float | None:
    '''Best-effort cast to float.'''
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(*, value: Any) -> int | None:
    '''Best-effort cast to int.'''
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_mentions_count(*, value: Any) -> int | None:
    '''Normalize mention count from integer-like value or mentions list.'''
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value)
    return _as_int(value=value)


def _as_str(*, value: Any) -> str | None:
    '''Best-effort cast to non-empty string.'''
    if value is None:
        return None
    value_str = str(value).strip()
    return value_str if value_str else None


# ---------------------------------------------------------------------------
# Gold and probability helpers
# ---------------------------------------------------------------------------

def add_prob_columns(*, df: pd.DataFrame) -> pd.DataFrame:
    '''Expand ``pred_scores`` into dense ``prob_<cat_id>`` columns, preserving other columns.'''
    cat_ids = sorted({cat_id for pred_scores in df['pred_scores'] for cat_id, _ in pred_scores})
    score_dicts = df['pred_scores'].map(dict)
    prob_data = {
        f'prob_{cat_id}': score_dicts.map(lambda d, _cid=cat_id: float(d.get(_cid, 0.0)))
        for cat_id in cat_ids
    }
    prob_df = pd.DataFrame(prob_data, index=df.index)
    return pd.concat([df, prob_df], axis=1)


def gold_df_from_corpus(*, corpus: Any) -> pd.DataFrame:
    '''Extract normalized gold dataframe from a ``geneea.catlib.data.Corpus``.'''
    rows = []
    for doc in corpus:
        article_id = str(doc.id)
        rows.append(
            {
                'article_id': article_id,
                'corpus_name': str(doc.metadata.get('corpusName', '')),
                'gold_categories': tuple(sorted(norm_cat_ids(cat_ids=doc.cats))),
            }
        )
    df = pd.DataFrame(rows)
    return df


def norm_cat_ids(*, cat_ids: Sequence[str]) -> list[str]:
    '''Apply the same IPTC normalization used by the legacy evaluator.'''
    iptc_topics = get_iptc_topics()
    valid_cats = []
    for cat_id in cat_ids:
        if not cat_id:
            continue
        try:
            valid_cats.append(iptc_topics.getCategory(str(cat_id)))
        except KeyError:
            LOG.warning('Skipping unknown IPTC category during normalization: cat_id=%s', cat_id)
    norm_cats = iptc_topics.normalizeCategories(valid_cats)
    return [cat.id for cat in norm_cats if cat.id and cat.id not in REMOVED_CAT_IDS]
