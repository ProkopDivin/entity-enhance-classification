"""Per-class assembly built on top of each member's ``run_cv`` results.

The dual-model assembly mode replaces the single ``run_cv`` call with one
``run_cv`` call per member. Each member's per-fold evaluation is driven by
that member's externally-loaded per-class thresholds (passed through
``run_cv``'s ``eval_thresholds`` argument), so the per-class CV F1 is
measured at the same thresholds the member would use in production.

The assembly itself is a pure aggregator:

1. For each class, pick the member with the highest CV mean F1 (ties go
   to the primary member).
2. Stitch the per-class thresholds from each selected member's loaded
   threshold file.

This module no longer runs a CV loop of its own.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from dataclasses import asdict

from iptc_entity_pipeline.assembly.model import ClassToModelMap
from iptc_entity_pipeline.config import EvaluationCnf
from iptc_entity_pipeline.cross_validation import CvOutputs
from iptc_entity_pipeline.evaluation.evaluate import get_cat_name

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AssemblyCvResult:
    """Outputs of the assembly aggregation step.

    Field names overlap with :class:`cross_validation.CV` attributes so
    ``eval_final`` consumes either typed result.

    :param cv_dev_df: One row per member (no fused row), indexed by member
        label. Columns mirror ``run_cv``'s ``cv_dev_df`` shape.
    :param objective_metrics: Per-member CV summary plus per-class
        selection counts.
    :param tuned_thresholds: Stitched per-class thresholds (each class's
        threshold comes from the loaded thresholds of the member that won
        that class).
    :param threshold_report_df: Per-class selection + threshold report.
    :param class_to_model: Per-class model selection map.
    :param per_class_f1_df: Long-form per-class F1 (mean + std) per member.
    :param per_corpora_df: Per-corpora CV table stacked across members
        (each member's rows tagged with ``member_label``).
    """

    cv_dev_df: pd.DataFrame
    objective_metrics: dict[str, Any]
    tuned_thresholds: dict[str, float]
    threshold_report_df: pd.DataFrame
    class_to_model: ClassToModelMap
    per_class_f1_df: pd.DataFrame
    per_corpora_df: pd.DataFrame


def _try_cat_name(cat_id: str) -> str:
    """Resolve a human-readable category name, falling back to the raw id."""
    try:
        return get_cat_name(cat_id)
    except Exception:
        LOGGER.debug(f'Could not resolve category name for cat_id={cat_id}')
        return cat_id


def _class_index_for(cat_id: str) -> str:
    """Return the index label used by :func:`evaluate.evaluate_classes`."""
    return f'"{_try_cat_name(cat_id)}"'


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Return ``float(value)`` unless missing/NaN, then ``default``."""
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return float(value)


def build_per_class_f1_df(
    *,
    member_cv_per_class_dfs: Sequence[pd.DataFrame | None],
    cat_list: Sequence[str],
    member_labels: Sequence[str],
) -> pd.DataFrame:
    """
    Long-form per-class F1 (mean + std) per member.

    Sourced from each member's CV per-class table; one row per
    ``(cat_id, member)`` pair. Classes missing from a member's table
    default to F1=0.
    """
    rows: list[dict[str, Any]] = []
    for member_idx, member_label in enumerate(member_labels):
        df = member_cv_per_class_dfs[member_idx]
        for cat_id in cat_list:
            cid = str(cat_id)
            mean_f1 = std_f1 = mean_p = mean_r = 0.0
            if df is not None:
                row_idx = _class_index_for(cid)
                if row_idx in df.index:
                    row = df.loc[row_idx]
                    mean_f1 = _safe_float(row.get('F1'))
                    std_f1 = _safe_float(row.get('F1_std'))
                    mean_p = _safe_float(row.get('Precision'))
                    mean_r = _safe_float(row.get('Recall'))
            rows.append({
                'cat_id': cid,
                'cat_name': _try_cat_name(cid),
                'member_idx': member_idx,
                'member_label': member_label,
                'mean_f1': mean_f1,
                'std_f1': std_f1,
                'mean_precision': mean_p,
                'mean_recall': mean_r,
            })
    return pd.DataFrame(rows)


def select_class_to_model(
    *,
    per_class_f1_df: pd.DataFrame,
    cat_list: Sequence[str],
    member_labels: Sequence[str],
    primary_idx: int = 0,
) -> ClassToModelMap:
    """
    Pick, per class, the member with the highest CV mean F1.
    Ties resolve to ``primary_idx``.
    """
    grouped: dict[str, dict[int, float]] = defaultdict(dict)
    for _, row in per_class_f1_df.iterrows():
        grouped[str(row['cat_id'])][int(row['member_idx'])] = float(row['mean_f1'])

    assignments: dict[str, int] = {}
    for cat_id in cat_list:
        cid = str(cat_id)
        f1_per_member = grouped.get(cid, {})
        if not f1_per_member:
            assignments[cid] = primary_idx
            continue
        best_f1 = max(f1_per_member.values())
        if f1_per_member.get(primary_idx, float('-inf')) == best_f1:
            assignments[cid] = primary_idx
        else:
            assignments[cid] = max(f1_per_member.items(), key=lambda kv: kv[1])[0]
    return ClassToModelMap(assignments=assignments, member_labels=tuple(member_labels))


def select_class_to_model_sign_test(
    *,
    member_fold_class_dfs: Sequence[Sequence[pd.DataFrame] | None],
    cat_list: Sequence[str],
    member_labels: Sequence[str],
    primary_idx: int = 0,
) -> ClassToModelMap:
    """Sign-test per-class selection between two members.

    The primary member is kept for every class by default. For each
    non-primary member, count the folds where its F1 strictly exceeds
    the primary's F1 for that class; if the count is at least
    ``n_folds - 1`` (e.g. 4 out of 5), the class is reassigned to the
    non-primary. Ties or NaN F1 in a fold count as a primary win.

    :param member_fold_class_dfs: Per-member tuple of per-fold per-class
        DataFrames. Index labels follow :func:`_class_index_for` (same
        encoding as ``cv_per_class_df``). ``None`` for a member is
        treated as "no fold data" — that member never wins.
    """
    n_members = len(member_labels)
    if n_members < 2:
        return ClassToModelMap(
            assignments={str(cid): primary_idx for cid in cat_list},
            member_labels=tuple(member_labels),
        )

    primary_folds = member_fold_class_dfs[primary_idx]
    if primary_folds is None or len(primary_folds) == 0:
        return ClassToModelMap(
            assignments={str(cid): primary_idx for cid in cat_list},
            member_labels=tuple(member_labels),
        )
    n_folds = len(primary_folds)
    win_threshold = max(1, n_folds - 1)

    assignments: dict[str, int] = {}
    for cat_id in cat_list:
        cid = str(cat_id)
        row_idx = _class_index_for(cid)
        primary_f1s = [_fold_f1(df=df, row_idx=row_idx) for df in primary_folds]

        chosen = primary_idx
        for m_idx in range(n_members):
            if m_idx == primary_idx:
                continue
            other_folds = member_fold_class_dfs[m_idx]
            if other_folds is None or len(other_folds) != n_folds:
                continue
            wins = sum(
                1 for fold_idx in range(n_folds)
                if _fold_f1(df=other_folds[fold_idx], row_idx=row_idx) > primary_f1s[fold_idx]
            )
            if wins >= win_threshold:
                chosen = m_idx
                break
        assignments[cid] = chosen
    return ClassToModelMap(assignments=assignments, member_labels=tuple(member_labels))


def _fold_f1(*, df: pd.DataFrame | None, row_idx: str) -> float:
    """Return F1 for one class in one fold, or ``-inf`` if missing."""
    if df is None or row_idx not in df.index:
        return float('-inf')
    value = df.loc[row_idx].get('F1')
    if value is None:
        return float('-inf')
    try:
        if pd.isna(value):
            return float('-inf')
    except (TypeError, ValueError):
        pass
    return float(value)


def stitch_thresholds(
    *,
    class_to_model: ClassToModelMap,
    member_thresholds: Sequence[Mapping[str, float]],
    cat_list: Sequence[str],
    default_threshold: float,
) -> dict[str, float]:
    """Per-class threshold from the selected member's threshold map.

    Missing classes use ``default_threshold``.
    """
    out: dict[str, float] = {}
    for cat_id in cat_list:
        cid = str(cat_id)
        member_idx = class_to_model.assignments.get(cid)
        if member_idx is None:
            out[cid] = float(default_threshold)
            continue
        per_member = member_thresholds[member_idx]
        out[cid] = float(per_member.get(cid, default_threshold))
    return out


def build_threshold_report(
    *,
    per_class_f1_df: pd.DataFrame,
    class_to_model: ClassToModelMap,
    stitched_thresholds: Mapping[str, float],
    cat_list: Sequence[str],
    member_labels: Sequence[str],
) -> pd.DataFrame:
    """
    Build per-class selection report:
    cat_id, cat_name, mean_f1_<label>, std_f1_<label>, ...,
    selected_member_idx, selected_member_label, selected_threshold.
    """
    f1_by_cat: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    for _, row in per_class_f1_df.iterrows():
        f1_by_cat[str(row['cat_id'])][int(row['member_idx'])] = {
            'mean_f1': float(row['mean_f1']),
            'std_f1': float(row['std_f1']),
        }

    rows: list[dict[str, Any]] = []
    for cat_id in cat_list:
        cid = str(cat_id)
        per_member = f1_by_cat.get(cid, {})
        record: dict[str, Any] = {'cat_id': cid, 'cat_name': _try_cat_name(cid)}
        for member_idx, label in enumerate(member_labels):
            stats = per_member.get(member_idx, {'mean_f1': 0.0, 'std_f1': 0.0})
            record[f'mean_f1_{label}'] = float(stats['mean_f1'])
            record[f'std_f1_{label}'] = float(stats['std_f1'])
        selected_idx = class_to_model.assignments.get(cid, 0)
        record['selected_member_idx'] = int(selected_idx)
        record['selected_member_label'] = member_labels[selected_idx]
        record['selected_threshold'] = float(stitched_thresholds.get(cid, float('nan')))
        rows.append(record)
    return pd.DataFrame(rows).sort_values(by='cat_id').reset_index(drop=True)


def build_member_cv_dev_df(
    *,
    member_cv_results: Sequence[CvOutputs],
    member_labels: Sequence[str],
) -> pd.DataFrame:
    """
    One row per member with each member's CV mean/std P/R/F1/Loss + best
    params, indexed by member label.
    """
    rows: list[dict[str, Any]] = []
    used_labels: list[str] = []
    for member_idx, label in enumerate(member_labels):
        cv = member_cv_results[member_idx]
        if cv.cv_dev_df is None or cv.cv_dev_df.empty:
            continue
        row = cv.cv_dev_df.iloc[0].to_dict()
        model_dict = asdict(cv.best_model_config)
        train_dict = asdict(cv.best_training_config)
        row['params'] = json.dumps(
            {
                **model_dict,
                **{
                    k: v for k, v in train_dict.items()
                    if k in {'batch_size', 'learning_rate'}
                },
                'member_label': label,
            },
            sort_keys=True,
        )
        rows.append(row)
        used_labels.append(label)
    df = pd.DataFrame(rows)
    df.index = pd.Index(used_labels, name='member')
    return df


def build_per_corpora_df(
    *,
    member_cv_results: Sequence[CvOutputs],
    member_labels: Sequence[str],
) -> pd.DataFrame:
    """
    Per-corpora CV table stacked across members.

    Each member's per-corpora rows are concatenated with a ``member_label``
    column inserted as the first column.
    """
    blocks: list[pd.DataFrame] = []
    for member_idx, label in enumerate(member_labels):
        cv_corp = member_cv_results[member_idx].per_corpora_df
        if cv_corp is None:
            continue
        block = cv_corp.copy().reset_index()
        block.insert(0, 'member_label', label)
        blocks.append(block)
    if not blocks:
        return pd.DataFrame()
    return pd.concat(blocks, ignore_index=True)


def _resolve_member_thresholds(
    *,
    member_cv_results: Sequence[CvOutputs],
    extra_thresholds: Sequence[Mapping[str, float] | None] | None,
    cat_list: Sequence[str],
    default_threshold: float,
) -> list[dict[str, float]]:
    """For each member, pick the per-class threshold map to use for stitching.

    Order of preference: ``extra_thresholds[i]`` (the loaded threshold
    file passed through the assembly config) → CV-tuned thresholds the
    member's CV produced → flat ``default_threshold``.

    The loaded thresholds win because the user contract is: thresholds
    come from the previous run.
    """
    out: list[dict[str, float]] = []
    for m_idx, cv in enumerate(member_cv_results):
        loaded = (
            extra_thresholds[m_idx] if extra_thresholds is not None else None
        )
        if loaded:
            out.append({str(k): float(v) for k, v in loaded.items()})
            continue
        if cv.tuned_thresholds:
            out.append({str(k): float(v) for k, v in cv.tuned_thresholds.items()})
            continue
        out.append({str(cid): float(default_threshold) for cid in cat_list})
    return out


def build_assembly_from_cv(
    *,
    member_cv_results: Sequence[CvOutputs],
    member_labels: Sequence[str],
    cat_list: Sequence[str],
    eval_cfg: EvaluationCnf,
    objective_row: str,
    primary_idx: int = 0,
    member_loaded_thresholds: Sequence[Mapping[str, float] | None] | None = None,
    sign_test: bool = False,
) -> AssemblyCvResult:
    """
    Build the assembly result from each member's fitted :class:`CV`.

    :param member_cv_results: One :class:`CvOutputs` per member, in member
        index order. Each must have ``per_class_df`` populated.
    :param member_labels: Display labels indexed by member index. Index 0
        is the primary member; ties on mean F1 resolve to it.
    :param cat_list: Shared train ``catList`` (validated upstream).
    :param eval_cfg: Evaluation config; ``threshold_eval`` is the fallback
        for classes missing from a member's threshold map.
    :param objective_row: Evaluation table row for objective metrics.
    :param primary_idx: Index of the primary member (default 0).
    :param member_loaded_thresholds: Per-member externally-loaded
        threshold maps (one per member). These are the source of truth
        for the stitched per-class thresholds.
    """
    if len(member_cv_results) < 2:
        raise ValueError('Assembly requires at least two members')
    if len(member_labels) != len(member_cv_results):
        raise ValueError(
            f'member_labels ({len(member_labels)}) and member_cv_results '
            f'({len(member_cv_results)}) must align'
        )

    member_thresholds = _resolve_member_thresholds(
        member_cv_results=member_cv_results,
        extra_thresholds=member_loaded_thresholds,
        cat_list=cat_list,
        default_threshold=eval_cfg.threshold_eval,
    )

    per_class_f1_df = build_per_class_f1_df(
        member_cv_per_class_dfs=[cv.per_class_df for cv in member_cv_results],
        cat_list=cat_list,
        member_labels=member_labels,
    )
    if sign_test:
        class_to_model = select_class_to_model_sign_test(
            member_fold_class_dfs=[cv.per_class_fold_dfs for cv in member_cv_results],
            cat_list=cat_list,
            member_labels=member_labels,
            primary_idx=primary_idx,
        )
    else:
        class_to_model = select_class_to_model(
            per_class_f1_df=per_class_f1_df,
            cat_list=cat_list,
            member_labels=member_labels,
            primary_idx=primary_idx,
        )
    stitched = stitch_thresholds(
        class_to_model=class_to_model,
        member_thresholds=member_thresholds,
        cat_list=cat_list,
        default_threshold=eval_cfg.threshold_eval,
    )
    threshold_report_df = build_threshold_report(
        per_class_f1_df=per_class_f1_df,
        class_to_model=class_to_model,
        stitched_thresholds=stitched,
        cat_list=cat_list,
        member_labels=member_labels,
    )
    cv_dev_df = build_member_cv_dev_df(
        member_cv_results=member_cv_results,
        member_labels=member_labels,
    )
    per_corpora_df = build_per_corpora_df(
        member_cv_results=member_cv_results,
        member_labels=member_labels,
    )

    objective_metrics: dict[str, Any] = {
        'objective_row': objective_row,
        'n_classes': len(cat_list),
        'n_classes_selected_per_member': {
            label: int(sum(1 for v in class_to_model.assignments.values() if v == idx))
            for idx, label in enumerate(member_labels)
        },
    }
    for member_idx, label in enumerate(member_labels):
        cv = member_cv_results[member_idx]
        stats = cv.best_trial_stats or {}
        objective_metrics[f'F1_macro_relevant_mean_{label}'] = float(
            stats.get('F1_macro_relevant', float('nan'))
        )
        objective_metrics[f'F1_macro_relevant_std_{label}'] = float(
            stats.get('F1_macro_relevant_std', float('nan'))
        )

    LOGGER.info(
        f'Assembly selection complete: classes_per_member='
        f'{objective_metrics["n_classes_selected_per_member"]}'
    )
    return AssemblyCvResult(
        cv_dev_df=cv_dev_df,
        objective_metrics=objective_metrics,
        tuned_thresholds=stitched,
        threshold_report_df=threshold_report_df,
        class_to_model=class_to_model,
        per_class_f1_df=per_class_f1_df,
        per_corpora_df=per_corpora_df,
    )


def report_assembly_tables(
    *,
    clearml_logger: Any,
    assembly_result: AssemblyCvResult,
    member_labels: Sequence[str],
    print_logs: bool = False,
) -> None:
    """Emit the standard assembly tables and per-member counts to ClearML."""
    clearml_logger.report_table(
        title='Assembly',
        series='Per-class F1 aggregate',
        iteration=0,
        table_plot=assembly_result.per_class_f1_df,
    )
    clearml_logger.report_table(
        title='Assembly',
        series='Class-to-model',
        iteration=0,
        table_plot=assembly_result.threshold_report_df,
    )
    clearml_logger.report_table(
        title='Assembly',
        series='Member CV summary',
        iteration=0,
        table_plot=assembly_result.cv_dev_df,
    )
    if not assembly_result.per_corpora_df.empty:
        clearml_logger.report_table(
            title='Assembly',
            series='Per-corpora (per member)',
            iteration=0,
            table_plot=assembly_result.per_corpora_df,
        )
    counts = assembly_result.objective_metrics.get('n_classes_selected_per_member', {})
    for label in member_labels:
        clearml_logger.report_scalar(
            title='Assembly / classes per member',
            series=label,
            value=int(counts.get(label, 0)),
            iteration=0,
        )
    clearml_logger.report_text(
        f'Assembly: classes per member={counts}',
        print_console=print_logs,
    )
