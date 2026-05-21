"""Shared evaluation logic for corpora and per-class metric tables."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, NamedTuple, Sequence

import pandas as pd

from iptc_entity_pipeline.category_sets import load_relevant_cat_ids
from iptc_entity_pipeline.config import EvaluationCnf

LOGGER = logging.getLogger(__name__)

REMOVED_CAT_IDS = frozenset({'20000419'})
CLASS_RELEVANT_MACRO_ROW = 'All-relevant-macro'


class PreparedStats(NamedTuple):
    """Aggregated stats produced by :func:`evaluate_corpora` per sub-corpus."""

    count: int
    stats: Any
    zero_docs: int
    decent: list[str]


@dataclass
class CoreMetricAccumulator:
    """Column-oriented accumulator for Precision / Recall / F1 / Data Count."""

    data_count: list[Any] = field(default_factory=list)
    precision: list[Any] = field(default_factory=list)
    recall: list[Any] = field(default_factory=list)
    f1: list[Any] = field(default_factory=list)

    def append_core_metrics(self, *, data_count: int, stats: Any, digits: int = 3) -> None:
        """Append one row of P/R/F1 derived from a stats object (AvgData or ClassData)."""
        self.data_count.append(data_count)
        self.precision.append(round(stats.precision, digits))
        self.recall.append(round(stats.recall, digits))
        self.f1.append(round(stats.fmeasure(beta=1), digits))


@dataclass
class MetricAccumulator(CoreMetricAccumulator):
    """Per-class table: adds one false-positive count per row."""

    false_positive_count: list[Any] = field(default_factory=list)

    def append_metrics(
        self,
        *,
        data_count: int,
        stats: Any,
        digits: int = 3,
        false_positive_count: int | float | None = None,
    ) -> None:
        """Append one row of core metrics from an evaluation stats object."""
        self.append_core_metrics(data_count=data_count, stats=stats, digits=digits)
        if false_positive_count is not None:
            fp_val: int | float = false_positive_count
        elif hasattr(stats, 'predCnt') and hasattr(stats, 'correctCnt'):
            fp_val = int(stats.predCnt - stats.correctCnt)
        else:
            fp_val = float('nan')
        self.false_positive_count.append(fp_val)

    def to_dataframe_dict(self) -> dict[str, list[Any]]:
        """Return dict keyed by DataFrame column names."""
        d = {
            'Data Count': self.data_count,
            'Precision': self.precision,
            'Recall': self.recall,
            'F1': self.f1,
        }
        d['False Positive Count'] = self.false_positive_count
        return d


@dataclass
class CorporaMetricAccumulator(CoreMetricAccumulator):
    """Corpora table: false positive rate (micro over gold-negative slots) and label stats."""

    docs_no_labels: list[Any] = field(default_factory=list)
    decent_labels: list[Any] = field(default_factory=list)
    false_positive_rate: list[Any] = field(default_factory=list)

    def append_metrics(self, *, data_count: int, stats: Any, digits: int = 3) -> None:
        """Append one row of core metrics (no false-positive count column)."""
        self.append_core_metrics(data_count=data_count, stats=stats, digits=digits)

    def append_label_metrics(
        self,
        *,
        zero_label_docs: int,
        pred_cats: Sequence[Sequence[str]],
        decent_labels: Sequence[str],
        false_positive_rate: float,
    ) -> None:
        """Append one row of label-level metrics."""
        self.docs_no_labels.append(round(zero_label_docs / len(pred_cats), 3))
        self.decent_labels.append(len(decent_labels))
        self.false_positive_rate.append(false_positive_rate)

    def append_column_means(self) -> None:
        """Append per-column means as a summary row (All-macro)."""
        for col in (
            self.data_count,
            self.precision,
            self.recall,
            self.f1,
            self.false_positive_rate,
            self.docs_no_labels,
            self.decent_labels,
        ):
            col.append(pd.Series(col, dtype=float).mean())

    def to_dataframe_dict(self) -> dict[str, list[Any]]:
        """Return dict keyed by DataFrame column names."""
        d = {
            'Data Count': self.data_count,
            'Precision': self.precision,
            'Recall': self.recall,
            'F1': self.f1,
            'False Positive Rate': self.false_positive_rate,
            'Docs No Labels': self.docs_no_labels,
            'Decent Labels': self.decent_labels,
        }
        return d


def get_iptc_topics() -> Any:
    """Load IPTC topic metadata once per process."""
    from geneea.mediacats import iptc

    return iptc.IptcTopics.load()


def get_cat_name(cat_id: str) -> str:
    """Return human-readable IPTC category name."""
    return get_iptc_topics().getCategory(cat_id).getLongName(abbreviate=True, shorten=True)


def normalize_pred_cats(*, pred_cats: Sequence[Sequence[str]]) -> list[list[str]]:
    """Apply IPTC normalization and remove excluded categories from predictions."""
    iptc_topics = get_iptc_topics()
    norm = [iptc_topics.normalizeCategories([iptc_topics.getCategory(c) for c in cats]) for cats in pred_cats]
    return [sorted(c.id for c in cats if c.id not in REMOVED_CAT_IDS) for cats in norm]


def filter_and_normalize(
    *,
    pred_wgh_cats: Sequence[Any],
    thr: float,
    cat_to_thr: Mapping[str, float] | None = None,
) -> list[list[str]]:
    """Filter weighted predictions by threshold and normalize categories."""
    from geneea.catlib.model.model import filterLabels

    pred_cats = [filterLabels(dc, thr=thr, thrByLabel=cat_to_thr, keepWgh=False) for dc in pred_wgh_cats]
    return normalize_pred_cats(pred_cats=pred_cats)


def _is_decent_label(stats: Any) -> bool:
    return stats.precision >= 0.6 and stats.fmeasure(beta=0.4) >= 0.5 and stats.trueCnt >= 10


def _micro_false_positives_and_negatives(
    *,
    docs: Sequence[Any],
    pred_cats: Sequence[Sequence[str]],
    cat_list: Sequence[str],
) -> tuple[int, int]:
    """Count multi-label false positives and gold-negative (document, label) pairs.

    A gold-negative slot is a (doc, category) pair where the category is absent
    from gold labels for that document. False positives are predicted positives
    on those slots.

    :param docs: Evaluation documents (same length as ``pred_cats``).
    :param pred_cats: Normalized predicted category sets per document.
    :param cat_list: IPTC category ids defining the label set (same as eval corpus ``catList``).
    """
    doc_list = list(docs)
    if len(doc_list) != len(pred_cats):
        raise ValueError(f'docs and pred_cats length mismatch: {len(doc_list)} vs {len(pred_cats)}')
    fp = 0
    neg = 0
    for cat in cat_list:
        for doc, preds in zip(doc_list, pred_cats):
            if cat not in doc.cats:
                neg += 1
                if cat in preds:
                    fp += 1
    return fp, neg


def _false_positive_rate_micro(*, fp: int, negatives: int, digits: int = 4) -> float:
    """False positive rate as FP / (gold-negative label slots)."""
    if negatives == 0:
        return float('nan')
    return round(fp / negatives, digits)


def evaluate_corpora(
    *,
    pred_cats: Sequence[Sequence[str]],
    eval_corpus: Any,
    per_corpus: bool = True,
    averaging_type: str = 'datapoint',
) -> pd.DataFrame:
    """
    Build per-corpus + aggregate evaluation metrics table.

    :param pred_cats: Normalized predicted category lists per document.
    :param eval_corpus: Corpus with gold labels and ``corpusName`` metadata.
    :param per_corpus: Whether to include per-corpus rows.
    :param averaging_type: One of ``'datapoint'``, ``'micro'``, ``'macro'``.
    :return: DataFrame indexed by corpus name with Precision/Recall/F1 columns plus
        false positive rate (micro over gold-negative label slots), Docs No Labels, Decent Labels.
    """
    from geneea.evaluation import utils as evalutil
    from geneea.evaluation.utils import AvgData

    def prepare_stats(*, sub_data: Any, sub_pred_cats: list[list[str]]) -> PreparedStats:
        avg_stats, micro_stats, indiv_stats = evalutil.multiStats(
            goldVals=[d.cats for d in sub_data],
            predVals=sub_pred_cats,
        )
        if averaging_type == 'datapoint':
            stats = avg_stats
        elif averaging_type == 'micro':
            stats = micro_stats
        elif averaging_type == 'macro':
            stats = AvgData.empty()
            for cat in indiv_stats:
                stats.update(prec=indiv_stats[cat].precision, recall=indiv_stats[cat].recall)
        else:
            raise ValueError(f'Unsupported averaging_type={averaging_type}')
        zero_docs = sum(1 for cats in sub_pred_cats if not cats)
        decent = [name for name, st in indiv_stats.items() if _is_decent_label(st)]
        return PreparedStats(count=avg_stats.cnt, stats=stats, zero_docs=zero_docs, decent=decent)

    accumulator = CorporaMetricAccumulator()
    corpora_names: list[str] = []
    if per_corpus:
        corpora_names = sorted({d.metadata['corpusName'] for d in eval_corpus})
        for corpus_name in corpora_names:
            mask = [d.metadata['corpusName'] == corpus_name for d in eval_corpus]
            sub_data = eval_corpus.filterByBools(mask)
            sub_pred = [cats for in_scope, cats in zip(mask, pred_cats) if in_scope]
            prepared = prepare_stats(sub_data=sub_data, sub_pred_cats=sub_pred)
            fp_sub, neg_sub = _micro_false_positives_and_negatives(
                docs=sub_data, pred_cats=sub_pred, cat_list=eval_corpus.catList,
            )
            accumulator.append_metrics(data_count=prepared.count, stats=prepared.stats)
            accumulator.append_label_metrics(
                zero_label_docs=prepared.zero_docs,
                pred_cats=sub_pred,
                decent_labels=prepared.decent,
                false_positive_rate=_false_positive_rate_micro(fp=fp_sub, negatives=neg_sub),
            )

    accumulator.append_column_means()
    corpora_names.append('All-macro')

    gold_vals = [d.cats for d in eval_corpus]
    avg_stats, micro_stats, indiv_stats = evalutil.multiStats(goldVals=gold_vals, predVals=pred_cats)
    decent_labels = [name for name, st in indiv_stats.items() if _is_decent_label(st)]
    zero_label_docs = sum(1 for cats in pred_cats if not cats)

    fp_all, neg_all = _micro_false_positives_and_negatives(
        docs=eval_corpus, pred_cats=pred_cats, cat_list=eval_corpus.catList,
    )
    fpr_all = _false_positive_rate_micro(fp=fp_all, negatives=neg_all)

    accumulator.append_metrics(data_count=avg_stats.cnt, stats=micro_stats)
    accumulator.append_label_metrics(
        zero_label_docs=zero_label_docs,
        pred_cats=pred_cats,
        decent_labels=decent_labels,
        false_positive_rate=fpr_all,
    )
    corpora_names.append('All-micro')

    accumulator.append_metrics(data_count=avg_stats.cnt, stats=avg_stats)
    accumulator.append_label_metrics(
        zero_label_docs=zero_label_docs,
        pred_cats=pred_cats,
        decent_labels=decent_labels,
        false_positive_rate=fpr_all,
    )
    corpora_names.append('All-datapoint')

    df = pd.DataFrame(data=accumulator.to_dataframe_dict())
    df.index = corpora_names
    df.index.name = 'Corpus Name'
    return df


def evaluate_classes(
    *,
    pred_cats: Sequence[Sequence[str]],
    eval_corpus: Any,
    per_class: bool = True,
) -> pd.DataFrame:
    """
    Build per-class + aggregate evaluation metrics table.

    :param pred_cats: Normalized predicted category lists per document.
    :param eval_corpus: Corpus with gold labels and ``catList``.
    :param per_class: Whether to include per-class rows.
    :return: DataFrame indexed by category name with Precision/Recall/F1 and false positive counts.
        Aggregate rows: ``All - micro avg`` (micro over labels), ``All-relevant-macro``
        (unweighted macro over ``relevant_cats.yaml``), ``All - datapoint avg``.
    """
    from geneea.evaluation import utils as evalutil
    from geneea.evaluation.utils import AvgData

    accumulator = MetricAccumulator()
    category_names: list[str] = []
    if per_class:
        categories = eval_corpus.catList
        category_names = ['"' + get_cat_name(cat) + '"' for cat in categories]
        fp_by_cat: dict[str, int] = {}
        for category in categories:
            pred_vals = [1 if category in cats else 0 for cats in pred_cats]
            gold_vals = [1 if category in doc.cats else 0 for doc in eval_corpus]
            class_to_data, _, _ = evalutil.classStats(trueVals=gold_vals, predVals=pred_vals)
            pos = class_to_data[1]
            fp_by_cat[str(category)] = int(pos.predCnt - pos.correctCnt)
            accumulator.append_metrics(data_count=sum(gold_vals), stats=pos)
    else:
        fp_by_cat = {}

    gold_vals = [doc.cats for doc in eval_corpus]
    avg_stats, micro_stats, class_stats = evalutil.multiStats(goldVals=gold_vals, predVals=pred_cats)

    micro_fp = int(micro_stats.predCnt - micro_stats.correctCnt)
    relevant_ids = load_relevant_cat_ids()
    available_ids = {str(cat) for cat in class_stats}
    relevant_cat_ids = [cat_id for cat_id in relevant_ids if cat_id in available_ids]
    if not relevant_cat_ids:
        LOGGER.warning(
            'No relevant classes found in eval catList; falling back to all classes for relevant-macro metrics'
        )
        relevant_cat_ids = [str(cat) for cat in class_stats]
    relevant_mean_fp = (
        sum(fp_by_cat[cat_id] for cat_id in relevant_cat_ids if cat_id in fp_by_cat) / len(relevant_cat_ids)
        if relevant_cat_ids
        else float('nan')
    )

    accumulator.append_metrics(data_count=avg_stats.cnt, stats=micro_stats)
    category_names.append('All - micro avg')

    relevant_macro_avg = AvgData.empty()
    for cat in relevant_cat_ids:
        relevant_macro_avg.update(prec=class_stats[cat].precision, recall=class_stats[cat].recall)
    accumulator.append_metrics(
        data_count=avg_stats.cnt,
        stats=relevant_macro_avg,
        false_positive_count=relevant_mean_fp,
    )
    category_names.append(CLASS_RELEVANT_MACRO_ROW)

    accumulator.append_metrics(
        data_count=avg_stats.cnt,
        stats=avg_stats,
        false_positive_count=micro_fp,
    )
    category_names.append('All - datapoint avg')

    df = pd.DataFrame(data=accumulator.to_dataframe_dict())
    df.index = category_names
    df.index.name = 'IPTC Category'
    return df


def evaluate_predictions(
    *,
    pred_wgh_cats: Sequence[Any],
    eval_corpus: Any,
    evaluation_config: EvaluationCnf,
    cat_to_thr: Mapping[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter, normalize, and evaluate predictions into corpora + class tables.

    Convenience wrapper that combines :func:`filter_and_normalize`,
    :func:`evaluate_corpora`, and :func:`evaluate_classes`.

    :param pred_wgh_cats: Raw weighted prediction scores per document.
    :param eval_corpus: Corpus with gold labels.
    :param evaluation_config: Grouped evaluation config.
    :param cat_to_thr: Optional per-label thresholds.
    :return: Tuple of (corpora DataFrame, classes DataFrame).
    """
    pred_cats = filter_and_normalize(
        pred_wgh_cats=pred_wgh_cats,
        thr=evaluation_config.threshold_eval,
        cat_to_thr=cat_to_thr,
    )
    corpora_df = evaluate_corpora(
        pred_cats=pred_cats,
        eval_corpus=eval_corpus,
        per_corpus=evaluation_config.per_corpus,
        averaging_type=evaluation_config.averaging_type,
    )
    classes_df = evaluate_classes(
        pred_cats=pred_cats,
        eval_corpus=eval_corpus,
        per_class=evaluation_config.per_class,
    )
    return corpora_df, classes_df


_NUMERIC_KIND = 'biufc'


def aggregate_fold_dfs(
    *,
    fold_dfs: Sequence[pd.DataFrame],
    keep_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """
    Aggregate per-fold evaluation DataFrames into a single ``mean + std`` table.

    Every numeric column gets a mean column (under the original name, so the
    output is shape-compatible with ``eval_final``'s test tables) and a
    ``<col>_std`` companion column (population std across folds). Non-numeric
    columns listed in ``keep_columns`` are passed through from the first
    fold. The output is row-aligned with the union of fold indices.

    :param fold_dfs: Per-fold DataFrames; expected to share index labels and
        column names. Empty input returns an empty DataFrame.
    :param keep_columns: Non-numeric columns to copy from the first fold.
    """
    if not fold_dfs:
        return pd.DataFrame()
    first = fold_dfs[0]
    union_index = first.index
    for df in fold_dfs[1:]:
        union_index = union_index.union(df.index, sort=False)
    aligned = [df.reindex(union_index) for df in fold_dfs]

    numeric_cols = [
        col for col in first.columns
        if first[col].dtype.kind in _NUMERIC_KIND
    ]
    out_data: dict[str, list[Any]] = {}
    for col in keep_columns:
        if col in first.columns:
            out_data[col] = aligned[0][col].tolist()
    for col in numeric_cols:
        stacked = pd.concat([df[col] for df in aligned], axis=1)
        out_data[col] = stacked.mean(axis=1, skipna=True).tolist()
        out_data[f'{col}_std'] = stacked.std(axis=1, ddof=0, skipna=True).tolist()

    out = pd.DataFrame(out_data, index=union_index)
    out.index.name = first.index.name
    return out
