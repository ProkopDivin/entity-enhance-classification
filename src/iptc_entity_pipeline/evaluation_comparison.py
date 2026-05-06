"""Compare evaluation outputs between a current and base run.

Each run is a directory holding two pickled artifacts produced by
:func:`iptc_entity_pipeline.model_io.save_final_model_outputs`:

- ``predictions.pkl``: raw weighted predictions, ``list[list[tuple[str, float]]]``
  aligned positionally with the corpus.
- ``eval_corpus.pkl``: the ``geneea.catlib.data.Corpus`` used during evaluation,
  carrying gold labels and ``corpusName`` metadata.

Gold labels for cross-run metrics are derived from the *current* run's corpus.
"""

from __future__ import annotations

import argparse
import logging
import math
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from iptc_entity_pipeline.config import EvaluationCnf
from iptc_entity_pipeline.data_loading import sanitize_name
from iptc_entity_pipeline.evaluate import REMOVED_CAT_IDS, evaluate_predictions, get_cat_name, get_iptc_topics
from iptc_entity_pipeline.model_io import EVAL_CORPUS_FILENAME, PREDICTIONS_FILENAME

LOG = logging.getLogger(__name__)

AGG_CLASS_ROWS = frozenset({'All - micro avg', 'All - macro avg', 'All - datapoint avg'})
AGG_CORPUS_ROWS = frozenset({'All-macro', 'All-micro', 'All-datapoint'})
SUMMARY_ROWS = {
    'macro_over_classes_all': ('classes', 'All - macro avg'),
    'micro_over_labels': ('classes', 'All - micro avg'),
}
SUPPORT_BUCKETS: tuple[tuple[int, int, str], ...] = (
    (0, 10, '0-10'),
    (10, 100, '10-100'),
    (100, 1000, '100-1000'),
    (1000, 10000, '1000-10000'),
)
LANG_PREFIXES: tuple[str, ...] = ('en', 'es', 'nl', 'fr', 'de', 'cs')
EUROSPORT_TOKEN = 'eurosport'
ENTITY_TABLE_LIMIT = 1000
ENTITY_RANDOM_SEED = 42
MCNEMAR_ALPHA = 0.05
MCNEMAR_MIN_DISAGREEMENTS = 25
BOOTSTRAP_PR_AUC_ITERATIONS = 1000
BOOTSTRAP_PR_AUC_SEED = 43
BOOTSTRAP_PR_AUC_ALPHA = 0.05
BOOTSTRAP_PR_AUC_MIN_POSITIVES = 15

_CMP_METRICS: tuple[tuple[str, str], ...] = (
    ('precision', 'Precision'),
    ('recall', 'Recall'),
    ('f1', 'F1'),
)

_RESULT_SHEETS: tuple[tuple[str, str, bool], ...] = (
    ('corpora_comparison', 'corpora_comparison', False),
    ('classes_comparison', 'classes_comparison', False),
    ('summary_comparison', 'summary_comparison', False),
    ('top_improved_categories', 'top_improved', False),
    ('top_degraded_categories', 'top_degraded', False),
    ('top_improved_stats', 'top_improved_stats', False),
    ('top_degraded_stats', 'top_degraded_stats', False),
    ('hamming_loss_comparison', 'hamming_loss', False),
    ('pr_auc_per_class', 'pr_auc_per_class', False),
    ('pr_auc_summary', 'pr_auc_summary', False),
    ('entity_impact_improvers', 'entity_impact_improvers', False),
    ('entity_impact_decaders', 'entity_impact_decaders', False),
    ('entity_impact_random', 'entity_impact_random', False),
    ('article_f1_diff_avg_stats', 'article_f1_diff_avg_stats', False),
    ('current_corpora', 'current_corpora', True),
    ('current_classes', 'current_classes', True),
    ('base_corpora', 'base_corpora', True),
    ('base_classes', 'base_classes', True),
)
_TOP_CHANGE_CATEGORY_SHEETS: tuple[tuple[str, str, bool], ...] = (
    ('top_improved_categories', 'top_improved', False),
    ('top_degraded_categories', 'top_degraded', False),
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ComparisonResult:
    """Structured outputs from one evaluation comparison run."""

    corpora_comparison: pd.DataFrame
    classes_comparison: pd.DataFrame
    summary_comparison: pd.DataFrame
    top_improved_categories: pd.DataFrame
    top_degraded_categories: pd.DataFrame
    top_improved_stats: pd.DataFrame
    top_degraded_stats: pd.DataFrame
    hamming_loss_comparison: pd.DataFrame
    pr_auc_per_class: pd.DataFrame
    pr_auc_summary: pd.DataFrame
    entity_impact_improvers: pd.DataFrame
    entity_impact_decaders: pd.DataFrame
    entity_impact_random: pd.DataFrame
    article_f1_diff_avg_stats: pd.DataFrame
    current_corpora: pd.DataFrame
    current_classes: pd.DataFrame
    base_corpora: pd.DataFrame
    base_classes: pd.DataFrame
    excel_path: Path | None = None


@dataclass(frozen=True)
class GoldArticle:
    """Gold metadata for one article."""

    article_id: str
    corpus_name: str
    gold_categories: tuple[str, ...]


@dataclass(frozen=True)
class GoldLabelMap:
    """Gold labels loaded once and reused across all comparisons."""

    df: pd.DataFrame
    article_map: Mapping[str, GoldArticle]

    @classmethod
    def from_corpus(cls, *, corpus: Any) -> 'GoldLabelMap':
        """Build a gold label map from a ``geneea.catlib.data.Corpus``."""
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
        """Collect category ids from gold labels plus all probability tables."""
        cat_ids = {cat_id for gold in self.article_map.values() for cat_id in gold.gold_categories}
        for prob_df in prob_dfs:
            cat_ids.update(col.removeprefix('prob_') for col in prob_df.columns if col.startswith('prob_'))
        return sorted(cat_ids)

    def gold_matrix(self, *, article_ids: Sequence[str], cat_ids: Sequence[str]) -> np.ndarray:
        """Build gold binary matrix for selected articles and categories."""
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
    """One run after table rebuild and per-article alignment."""

    aligned_df: pd.DataFrame
    corpora_df: pd.DataFrame
    classes_df: pd.DataFrame


@dataclass(frozen=True)
class ArticleEntity:
    """One normalized entity mention attached to an article."""

    gkb_id: str | None
    wdids: tuple[str, ...]
    entity_type: str | None
    std_form: str | None
    relevance: float | None
    mention_count: int | None


@dataclass(frozen=True)
class ArticleEvalRecord:
    """Normalized article payload used before DataFrame conversion."""

    article_id: str
    corpus_name: str
    gold_categories: tuple[str, ...]
    pred_scores: tuple[tuple[str, float], ...]
    article_text: str | None
    article_length: int | None
    entities: tuple[ArticleEntity, ...]


# ---------------------------------------------------------------------------
# Run loading and comparison orchestration
# ---------------------------------------------------------------------------

def load_run(*, run_dir: str | Path) -> tuple[list[list[tuple[str, float]]], Any]:
    """Load raw predictions and eval corpus from a saved run directory.

    :param run_dir: Directory containing ``predictions.pkl`` and ``eval_corpus.pkl``.
    :return: ``(pred_scores, eval_corpus)`` tuple aligned positionally.
    """
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


def compare_runs(
    *,
    current_run_dir: str | Path,
    base_run_dir: str | Path,
    threshold_eval: float,
    averaging_type: str = 'datapoint',
    top_n: int = 20,
    only_diff: bool = False,
    top_changes_only: bool = False,
    output_path: str | Path | None = None,
) -> ComparisonResult:
    """Compare current and base runs loaded from their saved directories.

    Gold labels are derived from the current run's pickled corpus and reused
    for both runs' cross-run metrics (Hamming loss, PR-AUC).

    :param current_run_dir: Directory of the current run.
    :param base_run_dir: Directory of the base run.
    :param threshold_eval: Threshold applied during evaluation.
    :param averaging_type: One of ``'datapoint'``, ``'micro'``, ``'macro'``.
    :param top_n: Preview size for improved/degraded categories logging.
    :param only_diff: Drop base metric columns from comparison sheets.
    :param top_changes_only: Persist only top improved/degraded category tables.
    :param output_path: Optional Excel workbook path.
    :return: Structured :class:`ComparisonResult`.
    """
    current_pred, current_corpus = load_run(run_dir=current_run_dir)
    base_pred, base_corpus = load_run(run_dir=base_run_dir)

    evaluation_config = EvaluationCnf(
        threshold_eval=threshold_eval,
        averaging_type=averaging_type,
    )

    current_run = build_run(
        pred_scores=current_pred,
        eval_corpus=current_corpus,
        evaluation_config=evaluation_config,
    )
    base_run = build_run(
        pred_scores=base_pred,
        eval_corpus=base_corpus,
        evaluation_config=evaluation_config,
    )

    classes_cmp_full = build_cmp_df(
        current_df=current_run.classes_df,
        base_df=base_run.classes_df,
        key_col='IPTC Category',
        info_cols=['Data Count'],
    )
    top_improved_df, top_degraded_df = build_top_change_dfs(classes_df=classes_cmp_full)
    gold_map = GoldLabelMap.from_corpus(corpus=current_corpus)
    shared_ids = shared_article_ids(current_df=current_run.aligned_df, base_df=base_run.aligned_df)
    cat_ids = gold_map.cat_ids(prob_dfs=[current_run.aligned_df, base_run.aligned_df])
    current_aligned_df = subset_by_ids(df=current_run.aligned_df, article_ids=shared_ids)
    base_aligned_df = subset_by_ids(df=base_run.aligned_df, article_ids=shared_ids)
    mcnemar_df = build_mcnemar_significance_df(
        current_df=current_aligned_df,
        base_df=base_aligned_df,
        gold_map=gold_map,
        cat_ids=cat_ids,
        threshold_eval=threshold_eval,
    )
    top_improved_df, top_degraded_df = add_mcnemar_to_top_change_dfs(
        improved_df=top_improved_df,
        degraded_df=top_degraded_df,
        mcnemar_df=mcnemar_df,
    )
    bootstrap_df = build_bootstrap_pr_auc_df(
        current_df=current_aligned_df,
        base_df=base_aligned_df,
        gold_map=gold_map,
        cat_ids=cat_ids,
        n_iterations=BOOTSTRAP_PR_AUC_ITERATIONS,
        seed=BOOTSTRAP_PR_AUC_SEED,
        alpha=BOOTSTRAP_PR_AUC_ALPHA,
        min_positives=BOOTSTRAP_PR_AUC_MIN_POSITIVES,
    )
    top_improved_df, top_degraded_df = add_bootstrap_pr_auc_to_top_change_dfs(
        improved_df=top_improved_df,
        degraded_df=top_degraded_df,
        bootstrap_df=bootstrap_df,
    )
    if top_changes_only:
        empty_df = pd.DataFrame()
        excel_path = Path(output_path) if output_path is not None else None
        result = ComparisonResult(
            corpora_comparison=empty_df,
            classes_comparison=empty_df,
            summary_comparison=empty_df,
            top_improved_categories=top_improved_df,
            top_degraded_categories=top_degraded_df,
            top_improved_stats=empty_df,
            top_degraded_stats=empty_df,
            hamming_loss_comparison=empty_df,
            pr_auc_per_class=empty_df,
            pr_auc_summary=empty_df,
            entity_impact_improvers=empty_df,
            entity_impact_decaders=empty_df,
            entity_impact_random=empty_df,
            article_f1_diff_avg_stats=empty_df,
            current_corpora=current_run.corpora_df,
            current_classes=current_run.classes_df,
            base_corpora=base_run.corpora_df,
            base_classes=base_run.classes_df,
            excel_path=excel_path,
        )
        if excel_path is not None:
            write_csv(result=result, output_path=excel_path.parent, result_sheets=_TOP_CHANGE_CATEGORY_SHEETS)
            write_excel(result=result, output_path=excel_path, result_sheets=_TOP_CHANGE_CATEGORY_SHEETS)
            log_top_changes(result=result, top_n=top_n)
        return result

    corpora_cmp_full = build_cmp_df(
        current_df=current_run.corpora_df,
        base_df=base_run.corpora_df,
        key_col='Corpus Name',
        info_cols=['Data Count', 'Docs No Labels', 'Decent Labels'],
    )
    corpora_cmp_df = diff_only_df(df=corpora_cmp_full, key_col='Corpus Name') if only_diff else corpora_cmp_full
    classes_cmp_df = diff_only_df(df=classes_cmp_full, key_col='IPTC Category') if only_diff else classes_cmp_full
    summary_df = build_summary_df(
        current_run=current_run,
        base_run=base_run,
        classes_cmp=classes_cmp_full,
        corpora_cmp=corpora_cmp_full,
    )
    top_improved_stats_df, top_degraded_stats_df = build_top_change_stats_dfs(
        improved_df=top_improved_df,
        degraded_df=top_degraded_df,
    )
    hamming_df = build_hamming_df(
        current_df=current_aligned_df,
        base_df=base_aligned_df,
        gold_map=gold_map,
        cat_ids=cat_ids,
        threshold_eval=threshold_eval,
    )
    pr_auc_df, pr_auc_summary_df = build_pr_auc_dfs(
        current_df=current_aligned_df,
        base_df=base_aligned_df,
        gold_map=gold_map,
        cat_ids=cat_ids,
    )
    article_f1_df = build_article_f1_diff_df(
        current_df=current_aligned_df,
        base_df=base_aligned_df,
        gold_map=gold_map,
        cat_ids=cat_ids,
        threshold_eval=threshold_eval,
    )
    entity_improvers_df, entity_decaders_df, entity_random_df = build_entity_impact_tables(
        current_df=current_aligned_df,
        article_f1_df=article_f1_df,
        limit=ENTITY_TABLE_LIMIT,
        random_seed=ENTITY_RANDOM_SEED,
    )
    article_avg_stats_df = build_article_f1_diff_avg_stats(df=article_f1_df, limit=ENTITY_TABLE_LIMIT)

    excel_path = Path(output_path) if output_path is not None else None
    result = ComparisonResult(
        corpora_comparison=corpora_cmp_df,
        classes_comparison=classes_cmp_df,
        summary_comparison=summary_df,
        top_improved_categories=top_improved_df,
        top_degraded_categories=top_degraded_df,
        top_improved_stats=top_improved_stats_df,
        top_degraded_stats=top_degraded_stats_df,
        hamming_loss_comparison=hamming_df,
        pr_auc_per_class=pr_auc_df,
        pr_auc_summary=pr_auc_summary_df,
        entity_impact_improvers=entity_improvers_df,
        entity_impact_decaders=entity_decaders_df,
        entity_impact_random=entity_random_df,
        article_f1_diff_avg_stats=article_avg_stats_df,
        current_corpora=current_run.corpora_df,
        current_classes=current_run.classes_df,
        base_corpora=base_run.corpora_df,
        base_classes=base_run.classes_df,
        excel_path=excel_path,
    )
    if excel_path is not None:
        result_sheets = _TOP_CHANGE_CATEGORY_SHEETS if top_changes_only else _RESULT_SHEETS
        write_csv(result=result, output_path=excel_path.parent, result_sheets=result_sheets)
        write_excel(result=result, output_path=excel_path, result_sheets=result_sheets)
        log_top_changes(result=result, top_n=top_n)
    return result


# ---------------------------------------------------------------------------
# Run evaluation helpers
# ---------------------------------------------------------------------------

def build_run(
    *,
    pred_scores: Sequence[Any],
    eval_corpus: Any,
    evaluation_config: EvaluationCnf,
) -> RunEval:
    """Evaluate one run and build its aligned per-article dataframe."""
    corpora_df, classes_df = evaluate_predictions(
        pred_wgh_cats=pred_scores,
        eval_corpus=eval_corpus,
        evaluation_config=evaluation_config,
        cat_to_thr=None,
    )
    aligned_df = build_aligned_df(eval_corpus=eval_corpus, pred_scores=pred_scores)
    return RunEval(aligned_df=aligned_df, corpora_df=corpora_df, classes_df=classes_df)


def build_aligned_df(*, eval_corpus: Any, pred_scores: Sequence[Any]) -> pd.DataFrame:
    """Build per-article dataframe with gold categories and dense ``prob_*`` columns."""
    records = build_article_records(eval_corpus=eval_corpus, pred_scores=pred_scores)
    return records_to_df(records=records)


def build_article_records(*, eval_corpus: Any, pred_scores: Sequence[Any]) -> list[ArticleEvalRecord]:
    """Build normalized article records from corpus docs and prediction tuples."""
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
    """Convert article records to the aligned DataFrame used by metric code."""
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


def _doc_metadata(*, doc: Any) -> Mapping[str, Any]:
    """Return document metadata mapping or an empty mapping."""
    metadata = getattr(doc, 'metadata', None)
    return metadata if isinstance(metadata, Mapping) else {}


def _extract_article_text(*, doc: Any) -> str | None:
    """Extract article text from document text payload."""
    text = getattr(doc, 'text', None)
    if isinstance(text, str) and text:
        return text
    return None


def _extract_article_length(*, doc: Any, metadata: Mapping[str, Any]) -> int | None:
    """Extract known typed article length if available."""
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
    """Extract normalized entities from doc attributes or metadata."""
    raw_entities = getattr(doc, 'entities', None)
    if raw_entities is None:
        raw_entities = _mapping_value(item=metadata, keys=('entities', 'article_entities', 'entity_list'))
    if not isinstance(raw_entities, Sequence) or isinstance(raw_entities, (str, bytes)):
        return ()
    entities = [_parse_entity(item=item) for item in raw_entities]
    return tuple(entity for entity in entities if entity is not None)


def _parse_entity(*, item: Any) -> ArticleEntity | None:
    """Normalize entity from known source shapes (raw CSV dict or LinkedEntity)."""
    raw_payload: Mapping[str, Any] | None = None

    if isinstance(item, Mapping):
        # Raw entities from CSV use keys like: gkbId, stdForm, type, mentions, feats.relevance.
        raw_payload = item
        gkb_raw = raw_payload.get('gkbId')
        wdids_raw = raw_payload.get('wdid') or raw_payload.get('wdids')
        entity_type_raw = raw_payload.get('type')
        std_form_raw = raw_payload.get('stdForm')
        relevance_raw = raw_payload.get('relevance')
        mentions_raw = raw_payload.get('mentions')
    else:
        # Linked entities attached in data_loading.py expose normalized attrs + raw_entity payload.
        gkb_raw = getattr(item, 'gkb_id', None)
        wdids_raw = getattr(item, 'wd_ids', None)
        relevance_raw = getattr(item, 'relevance', None)
        mentions_raw = getattr(item, 'mention_count', None)
        raw_maybe = getattr(item, 'raw_entity', None)
        if isinstance(raw_maybe, Mapping):
            raw_payload = raw_maybe
        entity_type_raw = raw_payload.get('type') if raw_payload is not None else None
        std_form_raw = raw_payload.get('stdForm') if raw_payload is not None else None

    if relevance_raw is None and raw_payload is not None:
        feats_raw = raw_payload.get('feats')
        if isinstance(feats_raw, Mapping):
            relevance_raw = feats_raw.get('relevance')

    gkb_id = _as_str(value=gkb_raw)
    wdids = _as_wdid_tuple(value=wdids_raw)
    entity_type = _as_str(value=entity_type_raw)
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
    """Return first present mapping key value."""
    for key in keys:
        if key in item:
            return item[key]
    return None


def _as_wdid_tuple(*, value: Any) -> tuple[str, ...]:
    """Normalize one-or-many Wikidata ids to an immutable tuple."""
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
    """Best-effort cast to float."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(*, value: Any) -> int | None:
    """Best-effort cast to int."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_mentions_count(*, value: Any) -> int | None:
    """Normalize mention count from integer-like value or mentions list."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value)
    return _as_int(value=value)


def _as_str(*, value: Any) -> str | None:
    """Best-effort cast to non-empty string."""
    if value is None:
        return None
    value_str = str(value).strip()
    return value_str if value_str else None


def add_prob_columns(*, df: pd.DataFrame) -> pd.DataFrame:
    """Expand ``pred_scores`` into dense ``prob_<cat_id>`` columns, preserving other columns."""
    cat_ids = sorted({cat_id for pred_scores in df['pred_scores'] for cat_id, _ in pred_scores})
    score_dicts = df['pred_scores'].map(dict)
    prob_data = {
        f'prob_{cat_id}': score_dicts.map(lambda d, _cid=cat_id: float(d.get(_cid, 0.0)))
        for cat_id in cat_ids
    }
    prob_df = pd.DataFrame(prob_data, index=df.index)
    return pd.concat([df, prob_df], axis=1)


def gold_df_from_corpus(*, corpus: Any) -> pd.DataFrame:
    """Extract normalized gold dataframe from a ``geneea.catlib.data.Corpus``."""
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
    dup_ids = df['article_id'][df['article_id'].duplicated()].astype(str).unique().tolist()
    if dup_ids:
        raise ValueError(f'Duplicate article_id values in eval corpus: {dup_ids[:5]}')
    return df


def norm_cat_ids(*, cat_ids: Sequence[str]) -> list[str]:
    """Apply the same IPTC normalization used by the legacy evaluator."""
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


# ---------------------------------------------------------------------------
# Comparison table builders
# ---------------------------------------------------------------------------

def build_cmp_df(
    *,
    current_df: pd.DataFrame,
    base_df: pd.DataFrame,
    key_col: str,
    info_cols: Sequence[str],
) -> pd.DataFrame:
    """Join current/base metric tables and compute current-base deltas."""
    metric_cols = [col for _, col in _CMP_METRICS]
    df = current_df.reset_index().merge(
        base_df.reset_index(),
        on=key_col,
        how='outer',
        suffixes=('_current', '_base'),
        sort=False,
    )
    for metric in metric_cols:
        df[f'{metric}_diff'] = df[f'{metric}_current'] - df[f'{metric}_base']

    cols = [key_col]
    for info_col in info_cols:
        cols.extend([f'{info_col}_current', f'{info_col}_base'])
    for metric in metric_cols:
        cols.extend([f'{metric}_current', f'{metric}_base', f'{metric}_diff'])
    return df.reindex(columns=cols)


def diff_only_df(*, df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    """Drop base metric columns while preserving current values and diffs."""
    drop_cols = {f'{col}_base' for _, col in _CMP_METRICS}
    cols = [key_col]
    cols.extend(col for col in df.columns if col != key_col and col not in drop_cols)
    return df.loc[:, cols]


# ---------------------------------------------------------------------------
# Summary and metric row helpers
# ---------------------------------------------------------------------------

def _metric_row(*, summary_key: str, current: pd.Series, base: pd.Series) -> dict[str, Any]:
    """Build one summary row from current/base metric series."""
    row: dict[str, Any] = {'summary_key': summary_key}
    for key, col in _CMP_METRICS:
        row[f'{key}_current'] = current[col]
        row[f'{key}_base'] = base[col]
        row[f'{key}_diff'] = current[col] - base[col]
    return row


def _avg_metrics_row(*, summary_key: str, sub_df: pd.DataFrame) -> dict[str, Any]:
    """Macro-average precision/recall/f1 over the rows of ``sub_df``."""
    row: dict[str, Any] = {'summary_key': summary_key}
    if sub_df.empty:
        nan = float('nan')
        for key, _ in _CMP_METRICS:
            row[f'{key}_current'] = nan
            row[f'{key}_base'] = nan
            row[f'{key}_diff'] = nan
        return row
    for key, col in _CMP_METRICS:
        row[f'{key}_current'] = float(sub_df[f'{col}_current'].mean())
        row[f'{key}_base'] = float(sub_df[f'{col}_base'].mean())
        row[f'{key}_diff'] = float(sub_df[f'{col}_diff'].mean())
    return row


def build_summary_df(
    *,
    current_run: RunEval,
    base_run: RunEval,
    classes_cmp: pd.DataFrame,
    corpora_cmp: pd.DataFrame,
) -> pd.DataFrame:
    """Build compact summary rows from aggregate evaluation outputs."""
    rows: list[dict[str, Any]] = []
    tables = {
        'corpora_current': current_run.corpora_df,
        'corpora_base': base_run.corpora_df,
        'classes_current': current_run.classes_df,
        'classes_base': base_run.classes_df,
    }
    for summary_key, (group_name, row_name) in SUMMARY_ROWS.items():
        current_row = tables[f'{group_name}_current'].loc[row_name]
        base_row = tables[f'{group_name}_base'].loc[row_name]
        rows.append(_metric_row(summary_key=summary_key, current=current_row, base=base_row))

    classes_filtered = classes_cmp[~classes_cmp['IPTC Category'].isin(AGG_CLASS_ROWS)].copy()
    classes_filtered['support'] = classes_filtered['Data Count_current'].combine_first(
        classes_filtered['Data Count_base']
    )
    for low, high, label in SUPPORT_BUCKETS:
        mask = (classes_filtered['support'] >= low) & (classes_filtered['support'] < high)
        sub = classes_filtered[mask]
        rows.append(
            _avg_metrics_row(summary_key=f'macro_over_classes_support_{label}', sub_df=sub)
        )

    corpora_filtered = corpora_cmp[~corpora_cmp['Corpus Name'].isin(AGG_CORPUS_ROWS)].copy()
    for prefix in LANG_PREFIXES:
        mask = corpora_filtered['Corpus Name'].str.startswith(f'{prefix}_')
        sub = corpora_filtered[mask]
        rows.append(
            _avg_metrics_row(summary_key=f'macro_over_corpora_prefix_{prefix}', sub_df=sub)
        )

    eurosport_mask = corpora_filtered['Corpus Name'].str.contains(EUROSPORT_TOKEN, case=False, na=False)
    rows.append(
        _avg_metrics_row(
            summary_key=f'macro_over_corpora_{EUROSPORT_TOKEN}',
            sub_df=corpora_filtered[eurosport_mask],
        )
    )

    macro_corpora_current = current_run.corpora_df.loc['All-macro']
    macro_corpora_base = base_run.corpora_df.loc['All-macro']
    rows.append(
        _metric_row(
            summary_key='macro_over_corpora',
            current=macro_corpora_current,
            base=macro_corpora_base,
        )
    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Top-change analysis
# ---------------------------------------------------------------------------

def build_top_change_dfs(*, classes_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build ranked improved and degraded category tables."""
    df = classes_df[~classes_df['IPTC Category'].isin(AGG_CLASS_ROWS)].copy()
    df['article_frequency'] = df['Data Count_current'].combine_first(df['Data Count_base'])
    df['ranking_score'] = df['F1_diff'].abs()
    cols = [
        'IPTC Category',
        'article_frequency',
        'ranking_score',
        'Precision_current',
        'Precision_base',
        'Precision_diff',
        'Recall_current',
        'Recall_base',
        'Recall_diff',
        'F1_current',
        'F1_base',
        'F1_diff',
    ]
    improved_df = df[df['F1_diff'] > 0].sort_values(
        by=['F1_diff', 'ranking_score', 'article_frequency'],
        ascending=[False, False, False],
        na_position='last',
    )
    degraded_df = df[df['F1_diff'] < 0].sort_values(
        by=['F1_diff', 'ranking_score', 'article_frequency'],
        ascending=[True, False, False],
        na_position='last',
    )
    return improved_df.reindex(columns=cols), degraded_df.reindex(columns=cols)


def build_mcnemar_significance_df(
    *,
    current_df: pd.DataFrame,
    base_df: pd.DataFrame,
    gold_map: GoldLabelMap,
    cat_ids: Sequence[str],
    threshold_eval: float,
    alpha: float = MCNEMAR_ALPHA,
    min_disagreements: int = MCNEMAR_MIN_DISAGREEMENTS,
) -> pd.DataFrame:
    """Run per-class McNemar tests on paired current/base predictions.

    ``n10`` counts articles where the current model is correct and the base
    model is wrong. ``n01`` counts the opposite. Rows with too few
    disagreements do not pass significance and receive ``NaN`` p-values.
    """
    article_ids = list(current_df['article_id'])
    gold_matrix = gold_map.gold_matrix(article_ids=article_ids, cat_ids=cat_ids).astype(bool)
    current_pred = build_score_matrix(df=current_df, cat_ids=cat_ids) >= threshold_eval
    base_pred = build_score_matrix(df=base_df, cat_ids=cat_ids) >= threshold_eval
    current_correct = current_pred == gold_matrix
    base_correct = base_pred == gold_matrix
    not_gold_matrix = np.logical_not(gold_matrix)

    rows = []
    for idx, cat_id in enumerate(cat_ids):
        n10 = int(np.logical_and(current_correct[:, idx], np.logical_not(base_correct[:, idx])).sum())
        n01 = int(np.logical_and(np.logical_not(current_correct[:, idx]), base_correct[:, idx]).sum())
        disagreements = n10 + n01
        skipped = int(disagreements < min_disagreements)
        p_value = (
            float('nan')
            if skipped
            else mcnemar_p_value(n10=n10, n01=n01)
        )
        rows.append(
            {
                'IPTC Category': safe_cat_label(cat_id=cat_id),
                'cat_id': cat_id,
                'mcnemar_p_value': p_value,
                'mcnemar_n10_current_only_correct': n10,
                'mcnemar_n01_base_only_correct': n01,
                'mcnemar_disagreements': disagreements,
                'current_false_positives': int(np.logical_and(current_pred[:, idx], not_gold_matrix[:, idx]).sum()),
                'current_false_negatives': int(
                    np.logical_and(np.logical_not(current_pred[:, idx]), gold_matrix[:, idx]).sum()
                ),
                'base_false_positives': int(np.logical_and(base_pred[:, idx], not_gold_matrix[:, idx]).sum()),
                'base_false_negatives': int(
                    np.logical_and(np.logical_not(base_pred[:, idx]), gold_matrix[:, idx]).sum()
                ),
                'mcnemar_current_significant': int(not skipped and p_value < alpha and n10 > n01),
                'mcnemar_base_significant': int(not skipped and p_value < alpha and n01 > n10),
            }
        )
    return pd.DataFrame(rows)


def mcnemar_p_value(*, n10: int, n01: int) -> float:
    """Return asymptotic McNemar p-value with continuity correction."""
    disagreements = n10 + n01
    if disagreements == 0:
        return 1.0
    statistic = (abs(n10 - n01) - 1) ** 2 / disagreements
    return math.erfc(math.sqrt(statistic / 2))


def add_mcnemar_to_top_change_dfs(
    *,
    improved_df: pd.DataFrame,
    degraded_df: pd.DataFrame,
    mcnemar_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach McNemar significance columns to top improved/degraded tables."""
    return (
        _add_mcnemar_to_top_change_df(
            df=improved_df,
            mcnemar_df=mcnemar_df,
            pass_col='mcnemar_current_significant',
        ),
        _add_mcnemar_to_top_change_df(
            df=degraded_df,
            mcnemar_df=mcnemar_df,
            pass_col='mcnemar_base_significant',
        ),
    )


def _add_mcnemar_to_top_change_df(
    *,
    df: pd.DataFrame,
    mcnemar_df: pd.DataFrame,
    pass_col: str,
) -> pd.DataFrame:
    """Merge McNemar rows and expose a direction-aware pass flag."""
    cols = [
        'IPTC Category',
        'mcnemar_p_value',
        'mcnemar_n10_current_only_correct',
        'mcnemar_n01_base_only_correct',
        'mcnemar_disagreements',
        'current_false_positives',
        'current_false_negatives',
        'base_false_positives',
        'base_false_negatives',
        pass_col,
    ]
    result = df.merge(mcnemar_df.reindex(columns=cols), on='IPTC Category', how='left')
    result['mcnemar_pass'] = result[pass_col].fillna(0).astype(int)
    result = result.drop(columns=[pass_col])
    metric_cols = [
        'mcnemar_pass',
        'mcnemar_p_value',
        'mcnemar_n10_current_only_correct',
        'mcnemar_n01_base_only_correct',
        'mcnemar_disagreements',
        'current_false_positives',
        'current_false_negatives',
        'base_false_positives',
        'base_false_negatives',
    ]
    base_cols = [col for col in result.columns if col not in metric_cols]
    return result.loc[:, base_cols + metric_cols]


def bootstrap_pr_auc_test(
    *,
    y_true: np.ndarray,
    prob_base: np.ndarray,
    prob_current: np.ndarray,
    n_iterations: int = BOOTSTRAP_PR_AUC_ITERATIONS,
    min_positives: int = BOOTSTRAP_PR_AUC_MIN_POSITIVES,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Bootstrap one-label PR-AUC current-base differences."""
    y_true = np.asarray(y_true, dtype=np.int8)
    prob_base = np.asarray(prob_base, dtype=float)
    prob_current = np.asarray(prob_current, dtype=float)
    positive_count = int(y_true.sum())
    if positive_count < min_positives:
        return _bootstrap_pr_auc_empty_result(positive_count=positive_count)

    generator = rng if rng is not None else np.random.default_rng(BOOTSTRAP_PR_AUC_SEED)
    n_samples = len(y_true)
    differences = []
    for _ in range(n_iterations):
        indices = generator.integers(0, n_samples, n_samples)
        y_boot = y_true[indices]
        if int(y_boot.sum()) == 0:
            continue

        base_auc = average_precision(y_true=y_boot, y_score=prob_base[indices])
        current_auc = average_precision(y_true=y_boot, y_score=prob_current[indices])
        if np.isnan(base_auc) or np.isnan(current_auc):
            continue
        differences.append(current_auc - base_auc)

    if not differences:
        return _bootstrap_pr_auc_empty_result(positive_count=positive_count)

    diffs = np.asarray(differences, dtype=float)
    return {
        'bootstrap_pr_auc_p_value_current': float((np.sum(diffs <= 0) + 1) / (len(diffs) + 1)),
        'bootstrap_pr_auc_p_value_base': float((np.sum(diffs >= 0) + 1) / (len(diffs) + 1)),
        'bootstrap_pr_auc_mean_diff': float(np.mean(diffs)),
        'bootstrap_pr_auc_positive_count': positive_count,
        'bootstrap_pr_auc_iterations': int(len(diffs)),
    }


def _bootstrap_pr_auc_empty_result(*, positive_count: int) -> dict[str, Any]:
    """Return a skipped bootstrap result with stable output keys."""
    return {
        'bootstrap_pr_auc_p_value_current': float('nan'),
        'bootstrap_pr_auc_p_value_base': float('nan'),
        'bootstrap_pr_auc_mean_diff': float('nan'),
        'bootstrap_pr_auc_positive_count': positive_count,
        'bootstrap_pr_auc_iterations': 0,
    }


def build_bootstrap_pr_auc_df(
    *,
    current_df: pd.DataFrame,
    base_df: pd.DataFrame,
    gold_map: GoldLabelMap,
    cat_ids: Sequence[str],
    n_iterations: int = BOOTSTRAP_PR_AUC_ITERATIONS,
    seed: int = BOOTSTRAP_PR_AUC_SEED,
    alpha: float = BOOTSTRAP_PR_AUC_ALPHA,
    min_positives: int = BOOTSTRAP_PR_AUC_MIN_POSITIVES,
) -> pd.DataFrame:
    """Build per-class bootstrap PR-AUC significance rows with FDR correction."""
    article_ids = list(current_df['article_id'])
    gold_matrix = gold_map.gold_matrix(article_ids=article_ids, cat_ids=cat_ids)
    current_scores = build_score_matrix(df=current_df, cat_ids=cat_ids)
    base_scores = build_score_matrix(df=base_df, cat_ids=cat_ids)
    rng = np.random.default_rng(seed)

    rows = []
    for idx, cat_id in enumerate(cat_ids):
        row = {
            'IPTC Category': safe_cat_label(cat_id=cat_id),
            'cat_id': cat_id,
        }
        row.update(
            bootstrap_pr_auc_test(
                y_true=gold_matrix[:, idx],
                prob_base=base_scores[:, idx],
                prob_current=current_scores[:, idx],
                n_iterations=n_iterations,
                min_positives=min_positives,
                rng=rng,
            )
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    return add_bootstrap_fdr_columns(df=df, alpha=alpha)


def add_bootstrap_fdr_columns(*, df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Add Benjamini-Hochberg corrected p-values and direction-specific pass flags."""
    result = df.copy()
    for direction in ('current', 'base'):
        p_col = f'bootstrap_pr_auc_p_value_{direction}'
        fdr_col = f'bootstrap_pr_auc_p_value_fdr_{direction}'
        pass_col = f'bootstrap_pr_auc_{direction}_significant'
        valid_mask = result[p_col].notna()
        result[fdr_col] = np.nan
        if valid_mask.any():
            corrected = benjamini_hochberg(p_values=result.loc[valid_mask, p_col].to_numpy(dtype=float))
            result.loc[valid_mask, fdr_col] = corrected
        if direction == 'current':
            direction_mask = result['bootstrap_pr_auc_mean_diff'] > 0
        else:
            direction_mask = result['bootstrap_pr_auc_mean_diff'] < 0
        result[pass_col] = ((result[fdr_col] < alpha) & direction_mask).astype(int)
    return result


def benjamini_hochberg(*, p_values: Sequence[float]) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR correction preserving input order."""
    p_array = np.asarray(p_values, dtype=float)
    if p_array.size == 0:
        return np.asarray([], dtype=float)

    order = np.argsort(p_array)
    ranked = p_array[order]
    ranks = np.arange(1, len(ranked) + 1, dtype=float)
    adjusted_ranked = ranked * len(ranked) / ranks
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.minimum(adjusted_ranked, 1.0)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


def add_bootstrap_pr_auc_to_top_change_dfs(
    *,
    improved_df: pd.DataFrame,
    degraded_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach direction-aware bootstrap PR-AUC columns to top change tables."""
    return (
        _add_bootstrap_pr_auc_to_top_change_df(
            df=improved_df,
            bootstrap_df=bootstrap_df,
            direction='current',
        ),
        _add_bootstrap_pr_auc_to_top_change_df(
            df=degraded_df,
            bootstrap_df=bootstrap_df,
            direction='base',
        ),
    )


def _add_bootstrap_pr_auc_to_top_change_df(
    *,
    df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    direction: str,
) -> pd.DataFrame:
    """Merge bootstrap PR-AUC rows and expose generic direction-aware columns."""
    p_col = f'bootstrap_pr_auc_p_value_{direction}'
    fdr_col = f'bootstrap_pr_auc_p_value_fdr_{direction}'
    pass_col = f'bootstrap_pr_auc_{direction}_significant'
    cols = [
        'IPTC Category',
        p_col,
        fdr_col,
        'bootstrap_pr_auc_mean_diff',
        'bootstrap_pr_auc_positive_count',
        'bootstrap_pr_auc_iterations',
        pass_col,
    ]
    renamed = bootstrap_df.reindex(columns=cols).rename(
        columns={
            p_col: 'bootstrap_pr_auc_p_value',
            fdr_col: 'bootstrap_pr_auc_p_value_fdr',
            pass_col: 'bootstrap_pr_auc_pass',
        }
    )
    result = df.merge(renamed, on='IPTC Category', how='left')
    result['bootstrap_pr_auc_pass'] = result['bootstrap_pr_auc_pass'].fillna(0).astype(int)
    metric_cols = [
        'bootstrap_pr_auc_pass',
        'bootstrap_pr_auc_p_value',
        'bootstrap_pr_auc_p_value_fdr',
        'bootstrap_pr_auc_mean_diff',
        'bootstrap_pr_auc_positive_count',
        'bootstrap_pr_auc_iterations',
    ]
    base_cols = [col for col in result.columns if col not in metric_cols]
    return result.loc[:, base_cols + metric_cols]


CHANGE_THRESHOLDS: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7)
TOP_CHANGE_N: int = 100


def build_top_change_stats_dfs(
    *,
    improved_df: pd.DataFrame,
    degraded_df: pd.DataFrame,
    top_n: int = TOP_CHANGE_N,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build summary stats tables for the improved and degraded category rankings."""
    improved_stats = _change_stats_rows(df=improved_df, label='improved', top_n=top_n)
    degraded_stats = _change_stats_rows(df=degraded_df, label='degraded', top_n=top_n)
    return pd.DataFrame(improved_stats), pd.DataFrame(degraded_stats)


def _change_stats_rows(*, df: pd.DataFrame, label: str, top_n: int) -> list[dict[str, Any]]:
    """Build the metric/value rows for one change-direction table."""
    abs_diff = df['F1_diff'].abs() if not df.empty else pd.Series(dtype=float)
    rows: list[dict[str, Any]] = [
        {'metric': f'count_{label}', 'value': int(len(df))},
    ]
    for thr in CHANGE_THRESHOLDS:
        rows.append(
            {'metric': f'count_{label}_f1_diff_gt_{thr}', 'value': int((abs_diff > thr).sum())}
        )

    top = df.head(top_n)
    rows.append(
        {
            'metric': f'avg_f1_diff_top_{top_n}',
            'value': float(top['F1_diff'].mean()) if not top.empty else float('nan'),
        }
    )
    rows.append(
        {
            'metric': f'avg_article_frequency_top_{top_n}',
            'value': float(top['article_frequency'].mean()) if not top.empty else float('nan'),
        }
    )

    if not top.empty:
        top_levels = top['IPTC Category'].map(top_level_from_label).value_counts().sort_values(ascending=False)
        for name, count in top_levels.items():
            rows.append({'metric': f'top_level_top_{top_n}::{name}', 'value': int(count)})
    return rows


def top_level_from_label(label: str) -> str:
    """Extract the IPTC top-level category name from a quoted long-name label.

    Example labels::

        '"sport >> chess (20001154)"'         -> 'sport'
        '"arts+ - arts, culture, ... (...)"'  -> 'arts+'

    :param label: Quoted IPTC long-name label.
    :return: Top-level category name (best-effort string parse).
    """
    inner = str(label).strip().strip('"').strip()
    delim_positions = [pos for pos in (inner.find(' >'), inner.find(' -')) if pos != -1]
    if delim_positions:
        return inner[: min(delim_positions)].strip()
    paren_idx = inner.find('(')
    return inner[:paren_idx].strip() if paren_idx != -1 else inner


# ---------------------------------------------------------------------------
# Article-level alignment helpers
# ---------------------------------------------------------------------------

def shared_article_ids(*, current_df: pd.DataFrame, base_df: pd.DataFrame) -> list[str]:
    """Return article ids shared by current and base runs in current order."""
    base_ids = set(base_df['article_id'])
    article_ids = [article_id for article_id in current_df['article_id'] if article_id in base_ids]
    if not article_ids:
        raise ValueError('Current and base runs do not share any aligned article_id values.')
    return article_ids


def subset_by_ids(*, df: pd.DataFrame, article_ids: Sequence[str]) -> pd.DataFrame:
    """Keep rows for the requested article ids in that exact order."""
    if not article_ids:
        return df.iloc[0:0].copy()
    return df.set_index('article_id', drop=False).loc[list(article_ids)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Article-level F1 deltas and entity impact
# ---------------------------------------------------------------------------


def build_article_f1_diff_df(
    *,
    current_df: pd.DataFrame,
    base_df: pd.DataFrame,
    gold_map: GoldLabelMap,
    cat_ids: Sequence[str],
    threshold_eval: float,
) -> pd.DataFrame:
    """Compute per-article F1 for current/base and their delta."""
    article_ids = list(current_df['article_id'])
    gold_matrix = gold_map.gold_matrix(article_ids=article_ids, cat_ids=cat_ids)
    current_scores = build_score_matrix(df=current_df, cat_ids=cat_ids)
    base_scores = build_score_matrix(df=base_df, cat_ids=cat_ids)
    current_f1 = compute_article_f1(scores=current_scores, gold_matrix=gold_matrix, threshold_eval=threshold_eval)
    base_f1 = compute_article_f1(scores=base_scores, gold_matrix=gold_matrix, threshold_eval=threshold_eval)
    article_f1_df = pd.DataFrame(
        {
            'article_id': article_ids,
            'corpus_name': current_df['corpus_name'].tolist(),
            'f1_current': current_f1,
            'f1_base': base_f1,
        }
    )
    article_f1_df['f1_diff'] = article_f1_df['f1_current'] - article_f1_df['f1_base']
    if 'article_length' in current_df.columns:
        article_f1_df['article_length'] = pd.to_numeric(current_df['article_length'], errors='coerce')
    return article_f1_df


def compute_article_f1(*, scores: np.ndarray, gold_matrix: np.ndarray, threshold_eval: float) -> np.ndarray:
    """Compute per-article F1 scores from score and gold matrices."""
    pred_matrix = scores >= threshold_eval
    gold_bool = gold_matrix.astype(bool)
    tp = np.logical_and(pred_matrix, gold_bool).sum(axis=1, dtype=np.int32)
    fp = np.logical_and(pred_matrix, np.logical_not(gold_bool)).sum(axis=1, dtype=np.int32)
    fn = np.logical_and(np.logical_not(pred_matrix), gold_bool).sum(axis=1, dtype=np.int32)
    denom = 2 * tp + fp + fn
    f1 = np.zeros_like(denom, dtype=float)
    valid = denom > 0
    f1[valid] = (2 * tp[valid]) / denom[valid]
    return f1


def build_entity_impact_tables(
    *,
    current_df: pd.DataFrame,
    article_f1_df: pd.DataFrame,
    limit: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build improvers/decaders/random entity impact tables."""
    entity_df = build_entity_impact_df(current_df=current_df, article_f1_df=article_f1_df)
    if entity_df.empty:
        empty = entity_df.reindex(columns=entity_impact_columns())
        return (
            append_avg_footer_row(df=empty, id_col_values={'gkbid': 'AVG', 'stdform': 'AVG'}),
            append_avg_footer_row(df=empty, id_col_values={'gkbid': 'AVG', 'stdform': 'AVG'}),
            append_avg_footer_row(df=empty, id_col_values={'gkbid': 'AVG', 'stdform': 'AVG'}),
        )

    improvers = entity_df[entity_df['entity_score'] > 0].sort_values(by='entity_score', ascending=False).head(limit)
    decaders = entity_df[entity_df['entity_score'] < 0].sort_values(by='entity_score', ascending=True).head(limit)
    random_n = min(limit, len(entity_df))
    random_df = entity_df.sample(n=random_n, random_state=random_seed) if random_n > 0 else entity_df.iloc[0:0].copy()
    improvers = append_avg_footer_row(df=improvers, id_col_values={'gkbid': 'AVG', 'stdform': 'AVG'})
    decaders = append_avg_footer_row(df=decaders, id_col_values={'gkbid': 'AVG', 'stdform': 'AVG'})
    random_df = append_avg_footer_row(df=random_df, id_col_values={'gkbid': 'AVG', 'stdform': 'AVG'})
    return improvers, decaders, random_df


def build_entity_impact_df(*, current_df: pd.DataFrame, article_f1_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate entity impact from article F1 deltas."""
    exploded = explode_entities(df=current_df)
    if exploded.empty:
        return pd.DataFrame(columns=entity_impact_columns())

    exploded = exploded.merge(article_f1_df[['article_id', 'f1_diff']], on='article_id', how='inner')
    if exploded.empty:
        return pd.DataFrame(columns=entity_impact_columns())

    score_df = exploded[['gkbid', 'article_id', 'f1_diff']].drop_duplicates(subset=['gkbid', 'article_id'])
    score_agg = score_df.groupby('gkbid', as_index=False).agg(
        entity_score=('f1_diff', 'sum'),
        article_count=('article_id', 'nunique'),
    )
    relevance_agg = exploded.groupby('gkbid', as_index=False).agg(avg_relevance=('relevance', 'mean'))
    mentions_per_article = exploded.groupby(['gkbid', 'article_id'], as_index=False).agg(
        mention_per_article=('mention_count', 'mean')
    )
    mention_agg = mentions_per_article.groupby('gkbid', as_index=False).agg(
        avg_mentions_count=('mention_per_article', 'mean')
    )
    stdform = choose_stdform_by_gkbid(df=exploded)
    entity_df = (
        score_agg.merge(relevance_agg, on='gkbid', how='left')
        .merge(mention_agg, on='gkbid', how='left')
        .merge(stdform, on='gkbid', how='left')
    )
    entity_df = entity_df.reindex(columns=entity_impact_columns())
    return entity_df


def explode_entities(*, df: pd.DataFrame) -> pd.DataFrame:
    """Explode article entities into one row per entity occurrence."""
    if 'entities' not in df.columns or 'article_id' not in df.columns:
        return pd.DataFrame(columns=['article_id', 'gkbid', 'stdform', 'relevance', 'mention_count'])
    entity_rows = df[['article_id', 'entities']].copy()
    entity_rows = entity_rows.explode('entities')
    entity_rows = entity_rows.dropna(subset=['entities'])
    if entity_rows.empty:
        return pd.DataFrame(columns=['article_id', 'gkbid', 'stdform', 'relevance', 'mention_count'])
    entity_rows['gkbid'] = entity_rows['entities'].map(lambda item: item.gkb_id if item is not None else None)
    entity_rows['stdform'] = entity_rows['entities'].map(lambda item: item.std_form if item is not None else None)
    entity_rows['relevance'] = entity_rows['entities'].map(lambda item: item.relevance if item is not None else None)
    entity_rows['mention_count'] = entity_rows['entities'].map(
        lambda item: item.mention_count if item is not None else None
    )
    entity_rows = entity_rows.drop(columns=['entities'])
    entity_rows['gkbid'] = entity_rows['gkbid'].astype(object)
    entity_rows = entity_rows[entity_rows['gkbid'].notna() & (entity_rows['gkbid'].astype(str).str.len() > 0)]
    entity_rows['relevance'] = pd.to_numeric(entity_rows['relevance'], errors='coerce')
    entity_rows['mention_count'] = pd.to_numeric(entity_rows['mention_count'], errors='coerce')
    return entity_rows


def choose_stdform_by_gkbid(*, df: pd.DataFrame) -> pd.DataFrame:
    """Select representative stdform per gkbid by most frequent non-empty value."""
    names = df[['gkbid', 'stdform']].dropna(subset=['gkbid']).copy()
    names['stdform'] = names['stdform'].fillna('').astype(str).str.strip()
    names = names[names['stdform'] != '']
    if names.empty:
        return pd.DataFrame(columns=['gkbid', 'stdform'])
    counts = names.groupby(['gkbid', 'stdform'], as_index=False).size()
    counts = counts.sort_values(by=['gkbid', 'size', 'stdform'], ascending=[True, False, True])
    return counts.drop_duplicates(subset=['gkbid'], keep='first')[['gkbid', 'stdform']]


def build_article_f1_diff_avg_stats(*, df: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Build three-row average stats table from article F1 deltas."""
    top_improving = df.nlargest(limit, 'f1_diff')
    top_decreasing = df.nsmallest(limit, 'f1_diff')
    rows = [
        article_avg_row(segment='top_1000_improving_articles', df=top_improving),
        article_avg_row(segment='top_1000_decreasing_articles', df=top_decreasing),
        article_avg_row(segment='all_articles', df=df),
    ]
    return pd.DataFrame(rows)


def article_avg_row(*, segment: str, df: pd.DataFrame) -> dict[str, Any]:
    """Build one row of numeric mean stats for an article subset."""
    row: dict[str, Any] = {'segment': segment, 'article_count': int(len(df))}
    numeric_cols = [
        col for col in ('f1_current', 'f1_base', 'f1_diff', 'article_length') if col in df.columns
    ]
    for col in numeric_cols:
        row[col] = float(pd.to_numeric(df[col], errors='coerce').mean()) if not df.empty else float('nan')
    return row


def append_avg_footer_row(*, df: pd.DataFrame, id_col_values: Mapping[str, str]) -> pd.DataFrame:
    """Append one footer row with numeric means and fixed identifier labels."""
    footer: dict[str, Any] = {**id_col_values}
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    for col in numeric_cols:
        footer[col] = float(df[col].mean()) if not df.empty else float('nan')
    return pd.concat([df, pd.DataFrame([footer], columns=df.columns)], ignore_index=True)


def entity_impact_columns() -> list[str]:
    """Return output column order for entity impact tables."""
    return ['gkbid', 'stdform', 'avg_relevance', 'avg_mentions_count', 'entity_score', 'article_count']


# ---------------------------------------------------------------------------
# Hamming loss
# ---------------------------------------------------------------------------

def build_hamming_df(
    *,
    current_df: pd.DataFrame,
    base_df: pd.DataFrame,
    gold_map: GoldLabelMap,
    cat_ids: Sequence[str],
    threshold_eval: float,
) -> pd.DataFrame:
    """Build current/base Hamming loss comparison."""
    article_ids = list(current_df['article_id'])
    gold_matrix = gold_map.gold_matrix(article_ids=article_ids, cat_ids=cat_ids)
    current_loss = compute_hamming(df=current_df, cat_ids=cat_ids, threshold_eval=threshold_eval, gold_matrix=gold_matrix)
    base_loss = compute_hamming(df=base_df, cat_ids=cat_ids, threshold_eval=threshold_eval, gold_matrix=gold_matrix)
    return pd.DataFrame(
        [{'metric': 'hamming_loss', 'current': current_loss, 'base': base_loss, 'diff': current_loss - base_loss}]
    )


def compute_hamming(
    *,
    df: pd.DataFrame,
    cat_ids: Sequence[str],
    threshold_eval: float,
    gold_matrix: np.ndarray,
) -> float:
    """Compute Hamming loss for one aligned probability table."""
    score_matrix = build_score_matrix(df=df, cat_ids=cat_ids)
    pred_matrix = (score_matrix >= threshold_eval).astype(np.int8)
    return float(np.mean(pred_matrix != gold_matrix))


# ---------------------------------------------------------------------------
# PR-AUC
# ---------------------------------------------------------------------------

def build_pr_auc_dfs(
    *,
    current_df: pd.DataFrame,
    base_df: pd.DataFrame,
    gold_map: GoldLabelMap,
    cat_ids: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build per-class and summary PR-AUC tables."""
    article_ids = list(current_df['article_id'])
    gold_matrix = gold_map.gold_matrix(article_ids=article_ids, cat_ids=cat_ids)
    current_scores = build_score_matrix(df=current_df, cat_ids=cat_ids)
    base_scores = build_score_matrix(df=base_df, cat_ids=cat_ids)

    rows = []
    for idx, cat_id in enumerate(cat_ids):
        y_true = gold_matrix[:, idx].astype(np.int8)
        current_pr_auc = average_precision(y_true=y_true, y_score=current_scores[:, idx])
        base_pr_auc = average_precision(y_true=y_true, y_score=base_scores[:, idx])
        rows.append(
            {
                'IPTC Category': safe_cat_label(cat_id=cat_id),
                'cat_id': cat_id,
                'pr_auc_current': current_pr_auc,
                'pr_auc_base': base_pr_auc,
                'pr_auc_diff': current_pr_auc - base_pr_auc
                if not np.isnan(current_pr_auc) and not np.isnan(base_pr_auc)
                else np.nan,
                'positive_support': int(y_true.sum()),
                'article_frequency': int(y_true.sum()),
            }
        )

    pr_auc_df = pd.DataFrame(rows)
    pr_auc_summary_df = build_pr_auc_summary_df(
        pr_auc_df=pr_auc_df,
        current_df=current_df,
        gold_matrix=gold_matrix,
        current_scores=current_scores,
        base_scores=base_scores,
    )
    return pr_auc_df, pr_auc_summary_df


def safe_cat_label(*, cat_id: str) -> str:
    """Return the quoted long name used by other tables, falling back to the id."""
    try:
        return '"' + get_cat_name(cat_id) + '"'
    except KeyError:
        LOG.warning('No IPTC name for cat_id=%s; using raw id in PR-AUC table', cat_id)
        return cat_id


def build_pr_auc_summary_df(
    *,
    pr_auc_df: pd.DataFrame,
    current_df: pd.DataFrame,
    gold_matrix: np.ndarray,
    current_scores: np.ndarray,
    base_scores: np.ndarray,
) -> pd.DataFrame:
    """Build PR-AUC summary rows: micro, macro, support buckets, per-corpus groups."""
    rows: list[dict[str, Any]] = []

    rows.append(
        _pr_auc_row(
            aggregation='macro_over_classes_all',
            current=_safe_mean(values=pr_auc_df['pr_auc_current']),
            base=_safe_mean(values=pr_auc_df['pr_auc_base']),
        )
    )
    rows.append(
        _pr_auc_row(
            aggregation='micro',
            current=micro_pr_auc(gold_matrix=gold_matrix, scores=current_scores),
            base=micro_pr_auc(gold_matrix=gold_matrix, scores=base_scores),
        )
    )

    for low, high, label in SUPPORT_BUCKETS:
        mask = (pr_auc_df['positive_support'] >= low) & (pr_auc_df['positive_support'] < high)
        sub = pr_auc_df[mask]
        rows.append(
            _pr_auc_row(
                aggregation=f'macro_over_classes_support_{label}',
                current=_safe_mean(values=sub['pr_auc_current']),
                base=_safe_mean(values=sub['pr_auc_base']),
            )
        )

    current_per_corpus = per_corpus_pr_auc(df=current_df, scores=current_scores, gold_matrix=gold_matrix)
    base_per_corpus = per_corpus_pr_auc(df=current_df, scores=base_scores, gold_matrix=gold_matrix)

    for prefix in LANG_PREFIXES:
        rows.append(
            _pr_auc_row(
                aggregation=f'macro_over_corpora_prefix_{prefix}',
                current=_avg_filtered(values_dict=current_per_corpus, predicate=_prefix_predicate(prefix)),
                base=_avg_filtered(values_dict=base_per_corpus, predicate=_prefix_predicate(prefix)),
            )
        )

    rows.append(
        _pr_auc_row(
            aggregation=f'macro_over_corpora_{EUROSPORT_TOKEN}',
            current=_avg_filtered(values_dict=current_per_corpus, predicate=_contains_predicate(EUROSPORT_TOKEN)),
            base=_avg_filtered(values_dict=base_per_corpus, predicate=_contains_predicate(EUROSPORT_TOKEN)),
        )
    )

    rows.append(
        _pr_auc_row(
            aggregation='macro_over_corpora',
            current=_safe_mean(values=pd.Series(list(current_per_corpus.values()), dtype=float)),
            base=_safe_mean(values=pd.Series(list(base_per_corpus.values()), dtype=float)),
        )
    )
    return pd.DataFrame(rows)


def _pr_auc_row(*, aggregation: str, current: float, base: float) -> dict[str, Any]:
    """Build one PR-AUC summary row with diff."""
    diff = current - base if not np.isnan(current) and not np.isnan(base) else np.nan
    return {'aggregation': aggregation, 'current': current, 'base': base, 'diff': diff}


def _safe_mean(*, values: pd.Series) -> float:
    """Mean over non-NaN values; NaN if empty."""
    cleaned = values.dropna()
    return float(cleaned.mean()) if not cleaned.empty else float('nan')


def _avg_filtered(*, values_dict: Mapping[str, float], predicate: Callable[[str], bool]) -> float:
    """Mean of dict values whose key matches ``predicate`` and is not NaN."""
    matched = [v for k, v in values_dict.items() if predicate(k) and not np.isnan(v)]
    return float(np.mean(matched)) if matched else float('nan')


def _prefix_predicate(prefix: str) -> Callable[[str], bool]:
    """Predicate matching corpus names starting with ``<prefix>_``."""
    return lambda name: name.startswith(f'{prefix}_')


def _contains_predicate(token: str) -> Callable[[str], bool]:
    """Predicate matching corpus names containing ``token`` (case-insensitive)."""
    lower = token.lower()
    return lambda name: lower in name.lower()


# ---------------------------------------------------------------------------
# Score matrix and average precision
# ---------------------------------------------------------------------------

def micro_pr_auc(*, gold_matrix: np.ndarray, scores: np.ndarray) -> float:
    """Compute micro PR-AUC by flattening across all classes."""
    if gold_matrix.size == 0:
        return float('nan')
    return average_precision(y_true=gold_matrix.flatten().astype(np.int8), y_score=scores.flatten())


def per_corpus_pr_auc(
    *,
    df: pd.DataFrame,
    scores: np.ndarray,
    gold_matrix: np.ndarray,
) -> dict[str, float]:
    """Compute per-corpus macro PR-AUC over classes with positive support in that corpus."""
    corpus_to_value: dict[str, float] = {}
    corpus_names = df['corpus_name'].to_numpy()
    for corpus_name in sorted({name for name in corpus_names if name}):
        mask = corpus_names == corpus_name
        per_class_values = []
        for idx in range(scores.shape[1]):
            y_true = gold_matrix[mask, idx].astype(np.int8)
            if y_true.sum() == 0:
                continue
            per_class_values.append(average_precision(y_true=y_true, y_score=scores[mask, idx]))
        corpus_to_value[corpus_name] = (
            float(np.mean(per_class_values)) if per_class_values else float('nan')
        )
    return corpus_to_value


def build_score_matrix(*, df: pd.DataFrame, cat_ids: Sequence[str]) -> np.ndarray:
    """Build dense score matrix for selected category ids."""
    cols = [f'prob_{cat_id}' for cat_id in cat_ids]
    return df.reindex(columns=cols, fill_value=0.0).to_numpy(dtype=float)


def average_precision(*, y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute discrete PR-AUC as average precision."""
    positives = int(y_true.sum())
    if positives == 0:
        return np.nan

    order = np.argsort(-y_score, kind='mergesort')
    sorted_true = y_true[order]
    tp = np.cumsum(sorted_true)
    fp = np.cumsum(1 - sorted_true)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives
    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


# ---------------------------------------------------------------------------
# Serialization (CSV / Excel)
# ---------------------------------------------------------------------------

def write_csv(
    *,
    result: ComparisonResult,
    output_path: Path,
    result_sheets: Sequence[tuple[str, str, bool]] = _RESULT_SHEETS,
) -> None:
    """Persist comparison outputs into CSV files."""
    output_path.mkdir(parents=True, exist_ok=True)
    for attr, filename, _ in result_sheets:
        getattr(result, attr).to_csv(output_path / f'{filename}.csv', index=False)


def write_excel(
    *,
    result: ComparisonResult,
    output_path: Path,
    result_sheets: Sequence[tuple[str, str, bool]] = _RESULT_SHEETS,
) -> None:
    """Persist comparison outputs into one Excel workbook."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path) as writer:
        for attr, sheet_name, include_index in result_sheets:
            getattr(result, attr).to_excel(writer, sheet_name=sheet_name, index=include_index)


def log_top_changes(*, result: ComparisonResult, top_n: int) -> None:
    """Log concise previews of top improved and degraded categories."""
    if not result.top_improved_categories.empty:
        LOG.info('Top improved categories:\n%s', result.top_improved_categories.head(top_n).to_string(index=False))
    if not result.top_degraded_categories.empty:
        LOG.info('Top degraded categories:\n%s', result.top_degraded_categories.head(top_n).to_string(index=False))
    if result.excel_path is not None:
        LOG.info('Saved comparison report to %s', result.excel_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_path(*, output_root: str | Path, config_name: str) -> Path:
    """Create timestamped comparison workbook path."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_name = sanitize_name(value=config_name)
    output_dir = Path(output_root) / f'{safe_name}_{timestamp}'
    return output_dir / f'evaluation_comparison_{safe_name}.xlsx'


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser for standalone run comparison."""
    parser = argparse.ArgumentParser(description='Compare current and base evaluation runs.')
    parser.add_argument(
        '--current-run', '-c', required=True,
        help='Current run directory containing predictions.pkl and eval_corpus.pkl.',
    )
    parser.add_argument(
        '--base-run', '-b', required=True,
        help='Base run directory containing predictions.pkl and eval_corpus.pkl.',
    )
    parser.add_argument('--config-name', default='comparison', help='Output name fragment.')
    parser.add_argument('--threshold-eval', type=float, default=0.5, help='Evaluation threshold.')
    parser.add_argument('--averaging-type', default='micro', choices=['datapoint', 'micro', 'macro'])
    parser.add_argument('--top-n', type=int, default=20, help='Preview size for improved/degraded categories.')
    parser.add_argument('--only-diff', action='store_true', help='Drop base metric columns from comparison sheets.')
    parser.add_argument(
        '--top-changes-only',
        action='store_true',
        help='Write only top_improved and top_degraded category tables.',
    )
    parser.add_argument(
        '--output-root',
        default='results/comparisons',
        help='Directory where the Excel report should be written.',
    )
    return parser


def main() -> None:
    """Run the comparison CLI."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    args = build_arg_parser().parse_args()
    output_path = build_path(output_root=args.output_root, config_name=args.config_name)
    compare_runs(
        current_run_dir=args.current_run,
        base_run_dir=args.base_run,
        threshold_eval=args.threshold_eval,
        averaging_type=args.averaging_type,
        top_n=args.top_n,
        only_diff=args.only_diff,
        top_changes_only=args.top_changes_only,
        output_path=output_path,
    )


if __name__ == '__main__':
    main()
