"""Shared evaluation logic for corpora and per-class metric tables."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, NamedTuple, Sequence

import pandas as pd

LOGGER = logging.getLogger(__name__)

REMOVED_CAT_IDS = frozenset({'20000419'})


class PreparedStats(NamedTuple):
    """Aggregated stats produced by :func:`evaluate_corpora` per sub-corpus."""

    count: int
    stats: Any
    zero_docs: int
    decent: list[str]


@dataclass
class MetricAccumulator:
    """Column-oriented accumulator for core evaluation metrics."""

    data_count: list[Any] = field(default_factory=list)
    precision: list[Any] = field(default_factory=list)
    recall: list[Any] = field(default_factory=list)
    f1: list[Any] = field(default_factory=list)

    def append_metrics(self, *, data_count: int, stats: Any, digits: int = 3) -> None:
        """Append one row of core metrics from an evaluation stats object."""
        self.data_count.append(data_count)
        self.precision.append(round(stats.precision, digits))
        self.recall.append(round(stats.recall, digits))
        self.f1.append(round(stats.fmeasure(beta=1), digits))

    def to_dataframe_dict(self) -> dict[str, list[Any]]:
        """Return dict keyed by DataFrame column names."""
        return {
            'Data Count': self.data_count,
            'Precision': self.precision,
            'Recall': self.recall,
            'F1': self.f1,
        }


@dataclass
class CorporaMetricAccumulator(MetricAccumulator):
    """Extended accumulator that also tracks label-level statistics."""

    docs_no_labels: list[Any] = field(default_factory=list)
    decent_labels: list[Any] = field(default_factory=list)

    def append_label_metrics(
        self,
        *,
        zero_label_docs: int,
        pred_cats: Sequence[Sequence[str]],
        decent_labels: Sequence[str],
    ) -> None:
        """Append one row of label-level metrics."""
        self.docs_no_labels.append(round(zero_label_docs / len(pred_cats), 3))
        self.decent_labels.append(len(decent_labels))

    def append_column_means(self) -> None:
        """Append per-column means as a summary row (All-macro)."""
        for col in (self.data_count, self.precision, self.recall, self.f1, self.docs_no_labels, self.decent_labels):
            col.append(pd.Series(col, dtype=float).mean())

    def to_dataframe_dict(self) -> dict[str, list[Any]]:
        """Return dict keyed by DataFrame column names."""
        d = super().to_dataframe_dict()
        d['Docs No Labels'] = self.docs_no_labels
        d['Decent Labels'] = self.decent_labels
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
    :return: DataFrame indexed by corpus name with Precision/Recall/F1 columns.
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
            accumulator.append_metrics(data_count=prepared.count, stats=prepared.stats)
            accumulator.append_label_metrics(
                zero_label_docs=prepared.zero_docs, pred_cats=sub_pred, decent_labels=prepared.decent,
            )

    accumulator.append_column_means()
    corpora_names.append('All-macro')

    gold_vals = [d.cats for d in eval_corpus]
    avg_stats, micro_stats, indiv_stats = evalutil.multiStats(goldVals=gold_vals, predVals=pred_cats)
    decent_labels = [name for name, st in indiv_stats.items() if _is_decent_label(st)]
    zero_label_docs = sum(1 for cats in pred_cats if not cats)

    accumulator.append_metrics(data_count=avg_stats.cnt, stats=micro_stats)
    accumulator.append_label_metrics(
        zero_label_docs=zero_label_docs, pred_cats=pred_cats, decent_labels=decent_labels,
    )
    corpora_names.append('All-micro')

    macro_stats = AvgData.empty()
    for cat in indiv_stats:
        macro_stats.update(prec=indiv_stats[cat].precision, recall=indiv_stats[cat].recall)
    accumulator.append_metrics(data_count=avg_stats.cnt, stats=avg_stats)
    accumulator.append_label_metrics(
        zero_label_docs=zero_label_docs, pred_cats=pred_cats, decent_labels=decent_labels,
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
    :return: DataFrame indexed by category name with Precision/Recall/F1 columns.
    """
    from geneea.evaluation import utils as evalutil
    from geneea.evaluation.utils import AvgData

    accumulator = MetricAccumulator()
    category_names: list[str] = []
    if per_class:
        categories = eval_corpus.catList
        category_names = ['"' + get_cat_name(cat) + '"' for cat in categories]
        for category in categories:
            pred_vals = [1 if category in cats else 0 for cats in pred_cats]
            gold_vals = [1 if category in doc.cats else 0 for doc in eval_corpus]
            class_to_data, _, _ = evalutil.classStats(trueVals=gold_vals, predVals=pred_vals)
            accumulator.append_metrics(data_count=sum(gold_vals), stats=class_to_data[1])

    gold_vals = [doc.cats for doc in eval_corpus]
    avg_stats, micro_stats, class_stats = evalutil.multiStats(goldVals=gold_vals, predVals=pred_cats)

    accumulator.append_metrics(data_count=avg_stats.cnt, stats=micro_stats)
    category_names.append('All - micro avg')

    macro_avg = AvgData.empty()
    for cat in class_stats:
        macro_avg.update(prec=class_stats[cat].precision, recall=class_stats[cat].recall)
    accumulator.append_metrics(data_count=avg_stats.cnt, stats=macro_avg)
    category_names.append('All - macro avg')

    accumulator.append_metrics(data_count=avg_stats.cnt, stats=avg_stats)
    category_names.append('All - datapoint avg')

    df = pd.DataFrame(data=accumulator.to_dataframe_dict())
    df.index = category_names
    df.index.name = 'IPTC Category'
    return df


def evaluate_predictions(
    *,
    pred_wgh_cats: Sequence[Any],
    eval_corpus: Any,
    thr: float,
    cat_to_thr: Mapping[str, float] | None = None,
    per_corpus: bool = True,
    per_class: bool = True,
    averaging_type: str = 'datapoint',
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter, normalize, and evaluate predictions into corpora + class tables.

    Convenience wrapper that combines :func:`filter_and_normalize`,
    :func:`evaluate_corpora`, and :func:`evaluate_classes`.

    :param pred_wgh_cats: Raw weighted prediction scores per document.
    :param eval_corpus: Corpus with gold labels.
    :param thr: Evaluation threshold.
    :param cat_to_thr: Optional per-label thresholds.
    :param per_corpus: Whether to include per-corpus rows.
    :param per_class: Whether to include per-class rows.
    :param averaging_type: One of ``'datapoint'``, ``'micro'``, ``'macro'``.
    :return: Tuple of (corpora DataFrame, classes DataFrame).
    """
    pred_cats = filter_and_normalize(pred_wgh_cats=pred_wgh_cats, thr=thr, cat_to_thr=cat_to_thr)
    corpora_df = evaluate_corpora(
        pred_cats=pred_cats,
        eval_corpus=eval_corpus,
        per_corpus=per_corpus,
        averaging_type=averaging_type,
    )
    classes_df = evaluate_classes(
        pred_cats=pred_cats,
        eval_corpus=eval_corpus,
        per_class=per_class,
    )
    return corpora_df, classes_df
