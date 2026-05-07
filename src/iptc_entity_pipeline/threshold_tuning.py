"""Per-class decision-threshold tuning by F-beta maximization on dev folds.

The pipeline trains its multi-label classifier with raw sigmoid scores in
[0, 1]. A single global threshold is suboptimal because category prior and
score calibration vary heavily across classes. This module implements a
per-class threshold sweep on dev predictions:

1. For each candidate threshold ``t`` in the configured grid, filter the
   predictions and compute per-class stats via :func:`evalutil.multiStats`.
2. For each class, pick the threshold that maximizes F-beta (default beta=1).
3. Aggregate per-fold per-class thresholds across CV folds (mean by default).
4. Use the aggregated map as ``customThresholds`` for the final model
   evaluation.

Public entry points:

- :func:`eval_at_thresholds` — per-threshold per-class stats.
- :func:`select_thresholds_by_f1` — per-class argmax over thresholds.
- :func:`tune_thresholds` — fold-level convenience wrapper.
- :func:`aggregate_fold_thresholds` — mean/mode aggregation + report table.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any, Mapping, Sequence

import pandas as pd

from iptc_entity_pipeline.config import ThresholdTuningCnf
from iptc_entity_pipeline.evaluate import filter_and_normalize, get_cat_name

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ThresholdTuningResult:
    """Aggregated outputs of one CV-wide threshold tuning run.

    :param cat_to_threshold: Final per-class threshold map, one entry per
        category in the corpus ``catList``. Categories never observed in any
        fold fall back to ``default_threshold``.
    :param fold_thresholds: Tuple of per-fold ``class -> threshold`` mappings.
    :param report_df: Per-class report with mean / std / mode / min / max
        thresholds plus the selected value, indexed by category id.
    """

    cat_to_threshold: dict[str, float]
    fold_thresholds: tuple[dict[str, float], ...]
    report_df: pd.DataFrame


def eval_at_thresholds(
    *,
    pred_wgh_cats: Sequence[Any],
    eval_corpus: Any,
    thresholds: Sequence[float],
) -> dict[float, dict[str, Any]]:
    """Compute per-class evaluation stats for each threshold.

    For every ``t`` in ``thresholds`` the raw weighted predictions are filtered
    with the same global ``filterLabels`` semantics used by the main evaluator
    and per-class precision/recall/F1 are produced via ``evalutil.multiStats``.

    :param pred_wgh_cats: Raw weighted predictions per document
        (output of ``model.classifyDataset(..., returnScores=True)``).
    :param eval_corpus: Corpus with gold labels.
    :param thresholds: Threshold grid to evaluate.
    :return: Mapping ``threshold -> {category_id -> ClassData}``.
    """
    from geneea.evaluation import utils as evalutil

    gold_vals = [doc.cats for doc in eval_corpus]
    thr_to_class_stats: dict[float, dict[str, Any]] = {}
    for thr in thresholds:
        pred_cats = filter_and_normalize(pred_wgh_cats=pred_wgh_cats, thr=float(thr))
        _, _, class_stats = evalutil.multiStats(goldVals=gold_vals, predVals=pred_cats)
        thr_to_class_stats[float(thr)] = dict(class_stats)
    return thr_to_class_stats


def select_thresholds_by_f1(
    *,
    thr_stats: Mapping[float, Mapping[str, Any]],
    f_beta: float = 1.0,
) -> dict[str, tuple[float, Any]]:
    """Select the F-beta-maximizing threshold per class.

    Mirrors the reference ``selectThresholdsByF1`` snippet but supports an
    arbitrary ``f_beta`` and returns ``(threshold, stats)`` per class.

    :param thr_stats: Mapping ``threshold -> {class -> ClassData}``.
    :param f_beta: F-measure beta (1.0 == F1, larger favors recall).
    :return: Mapping ``class -> (best_threshold, best_stats)``.
    """
    cat_to_thr_stats: dict[str, dict[float, Any]] = defaultdict(dict)
    for thr, cat_to_stats in thr_stats.items():
        for cat, stats in cat_to_stats.items():
            cat_to_thr_stats[cat][float(thr)] = stats

    def score(stats: Any) -> float:
        return float(stats.fmeasure(beta=f_beta))

    return {
        cat: max(d.items(), key=lambda kv: score(kv[1]))
        for cat, d in cat_to_thr_stats.items()
    }


def tune_thresholds(
    *,
    pred_wgh_cats: Sequence[Any],
    eval_corpus: Any,
    tuning_cfg: ThresholdTuningCnf,
) -> dict[str, float]:
    """Run a full per-class threshold tuning sweep on one dev split.

    :param pred_wgh_cats: Raw weighted predictions from the trained model.
    :param eval_corpus: Dev/val corpus with gold labels.
    :param tuning_cfg: Threshold tuning configuration (grid + beta).
    :return: Mapping ``class -> selected_threshold`` for classes present in
        either gold or prediction at any tested threshold.
    """
    if not tuning_cfg.thresholds:
        raise ValueError('tuning_cfg.thresholds must not be empty')
    thr_stats = eval_at_thresholds(
        pred_wgh_cats=pred_wgh_cats,
        eval_corpus=eval_corpus,
        thresholds=tuning_cfg.thresholds,
    )
    cat_to_best = select_thresholds_by_f1(thr_stats=thr_stats, f_beta=tuning_cfg.f_beta)
    return {cat: float(thr) for cat, (thr, _) in cat_to_best.items()}


def _safe_mode(values: Sequence[float]) -> float:
    """Return the most common value, breaking ties by first-occurrence order."""
    counts = Counter(values)
    best_count = max(counts.values())
    for val in values:
        if counts[val] == best_count:
            return float(val)
    return float(values[0])


def _try_cat_name(cat_id: str) -> str:
    """Resolve a human-readable category name, falling back to the raw id."""
    try:
        return get_cat_name(cat_id)
    except Exception:
        LOGGER.debug(f'Could not resolve category name for cat_id={cat_id}')
        return cat_id


def aggregate_fold_thresholds(
    *,
    fold_thresholds: Sequence[Mapping[str, float]],
    cat_list: Sequence[str],
    default_threshold: float,
    aggregation: str = 'mean',
) -> ThresholdTuningResult:
    """Aggregate per-fold per-class thresholds into a single map plus report.

    Categories not seen in any fold fall back to ``default_threshold``. The
    report table has one row per category and includes mean, std (population),
    mode, min, max, the number of folds the class was tuned on, and the final
    selected value. Rows are sorted by category id for deterministic output.

    :param fold_thresholds: Per-fold ``class -> threshold`` mappings.
    :param cat_list: Full category list (every entry gets a row).
    :param default_threshold: Fallback for classes never seen in any fold.
    :param aggregation: ``'mean'`` (default) or ``'mode'``.
    :return: Aggregated thresholds and per-class report DataFrame.
    """
    if aggregation not in {'mean', 'mode'}:
        raise ValueError(f'Unsupported threshold aggregation: {aggregation}')

    per_class: dict[str, list[float]] = defaultdict(list)
    for fold in fold_thresholds:
        for cat, thr in fold.items():
            per_class[cat].append(float(thr))

    rows: list[dict[str, Any]] = []
    cat_to_threshold: dict[str, float] = {}
    for cat in cat_list:
        values = per_class.get(cat, [])
        if not values:
            selected = float(default_threshold)
            cat_to_threshold[cat] = selected
            rows.append({
                'category_id': cat,
                'category_name': _try_cat_name(cat),
                'n_folds': 0,
                'threshold_mean': float('nan'),
                'threshold_std': float('nan'),
                'threshold_mode': float('nan'),
                'threshold_min': float('nan'),
                'threshold_max': float('nan'),
                'threshold_selected': selected,
            })
            continue
        thr_mean = float(mean(values))
        thr_std = float(pstdev(values)) if len(values) > 1 else 0.0
        thr_mode = _safe_mode(values)
        thr_min = float(min(values))
        thr_max = float(max(values))
        selected = thr_mean if aggregation == 'mean' else thr_mode
        cat_to_threshold[cat] = float(selected)
        rows.append({
            'category_id': cat,
            'category_name': _try_cat_name(cat),
            'n_folds': len(values),
            'threshold_mean': thr_mean,
            'threshold_std': thr_std,
            'threshold_mode': thr_mode,
            'threshold_min': thr_min,
            'threshold_max': thr_max,
            'threshold_selected': float(selected),
        })

    report_df = pd.DataFrame(rows).sort_values(by='category_id').reset_index(drop=True)
    fold_thresholds_tuple = tuple(dict(fold) for fold in fold_thresholds)
    return ThresholdTuningResult(
        cat_to_threshold=cat_to_threshold,
        fold_thresholds=fold_thresholds_tuple,
        report_df=report_df,
    )
