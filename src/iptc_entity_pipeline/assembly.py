"""Cross-validated dual-model assembly: per-class model selection.

Replaces ``run_cv`` when ``assembly.enabled=True``. For each CV split each
member is trained and evaluated; per-class F1 is computed at the member's
own loaded per-class thresholds. Across folds the per-class F1 mean drives
member selection (tie -> primary). The per-class thresholds for the resulting
ensemble are stitched from each member's threshold file according to the
selected member.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, replace
from statistics import mean, pstdev
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
import pandas as pd

from iptc_entity_pipeline.assembly_model import ClassToModelMap
from iptc_entity_pipeline.config import CvCnf, EvaluationCnf, ModelCnf, TrainingCnf
from iptc_entity_pipeline.cross_validation import prepare_cv
from iptc_entity_pipeline.dataset_builder import slice_dataset
from iptc_entity_pipeline.evaluate import filter_and_normalize, get_cat_name
from iptc_entity_pipeline.legacy_reuse import evaluateModel
from iptc_entity_pipeline.training import train_model

if TYPE_CHECKING:
    from geneea.catlib.vec.dataset import EmbeddingDataset

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AssemblyCvResult:
    """Outputs of the assembly CV step.

    Field names overlap with :class:`cross_validation.CvResult` so
    ``eval_final`` consumes either typed result.

    :param cv_dev_df: One row per member (no fused row). Indexed by member
        label; columns mirror the run_cv ``cv_dev_df`` shape (Precision,
        Recall, F1, Loss + ``_std`` variants + ``params``, ``epochs``).
    :param objective_metrics: Per-member mean/std summary dict.
    :param tuned_thresholds: Stitched per-class thresholds.
    :param threshold_report_df: Per-class selection report.
    :param class_to_model: Per-class model selection map.
    :param per_class_f1_df: Long-form F1 aggregation per class per member.
    :param per_fold_f1_df: Long-form per-fold per-class F1 per member.
    """

    cv_dev_df: pd.DataFrame
    objective_metrics: dict[str, Any]
    tuned_thresholds: dict[str, float]
    threshold_report_df: pd.DataFrame
    class_to_model: ClassToModelMap
    per_class_f1_df: pd.DataFrame
    per_fold_f1_df: pd.DataFrame


def _try_cat_name(cat_id: str) -> str:
    """Resolve a human-readable category name, falling back to the raw id."""
    try:
        return get_cat_name(cat_id)
    except Exception:
        LOGGER.debug(f'Could not resolve category name for cat_id={cat_id}')
        return cat_id


def _per_class_stats_from_predictions(
    *,
    pred_wgh_cats: Sequence[Any],
    eval_corpus: Any,
    threshold_eval: float,
    cat_to_thr: Mapping[str, float],
) -> dict[str, Any]:
    """Compute per-class stats by filtering predictions at per-class thresholds.

    Returns ``{cat_id -> ClassData}`` with ``.precision``, ``.recall``,
    ``.fmeasure(beta)``, ``.predCnt``, ``.correctCnt``, ``.trueCnt``
    attributes (geneea ``ClassData`` shape).
    """
    from geneea.evaluation import utils as evalutil

    pred_cats = filter_and_normalize(
        pred_wgh_cats=pred_wgh_cats,
        thr=float(threshold_eval),
        cat_to_thr=cat_to_thr,
    )
    gold_vals = [doc.cats for doc in eval_corpus]
    _, _, class_stats = evalutil.multiStats(goldVals=gold_vals, predVals=pred_cats)
    return dict(class_stats)


def evaluate_assembly_fold(
    *,
    fits: Sequence[Any],
    vals: Sequence[Any],
    feature_dims: Sequence[int],
    member_model_cfgs: Sequence[ModelCnf],
    member_train_cfgs: Sequence[TrainingCnf],
    member_thresholds: Sequence[Mapping[str, float]],
    eval_cfg: EvaluationCnf,
    fold_idx: int,
    print_logs: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Train and evaluate every member on one CV fold.

    :param fits: Per-member training subsets.
    :param vals: Per-member validation subsets.
    :param feature_dims: Per-member feature dimensions.
    :param member_model_cfgs: Per-member model configs.
    :param member_train_cfgs: Per-member training configs.
    :param member_thresholds: Per-member loaded per-class thresholds.
    :param eval_cfg: Evaluation config (drives ``threshold_predict`` /
        ``threshold_eval`` / averaging).
    :param fold_idx: Fold index (1-based) for logging.
    :param print_logs: Forwarded to the trainer.
    :return: ``(micro_rows_per_member, class_stats_per_member)``.
        - ``micro_rows_per_member[i]`` is one row dict per member with
          Precision/Recall/F1/Loss/epochs.
        - ``class_stats_per_member[i]`` is ``{cat_id -> ClassData}``.
    """
    micro_rows: list[dict[str, Any]] = []
    class_stats_per_member: list[dict[str, Any]] = []

    for member_idx, (fit, val, feat_dim, model_cfg, train_cfg, thr_map) in enumerate(
        zip(fits, vals, feature_dims, member_model_cfgs, member_train_cfgs, member_thresholds)
    ):
        train_result = train_model(
            train_data=fit,
            dev_data=val,
            feature_dim=int(feat_dim),
            model_config=model_cfg,
            training_config=train_cfg,
            print_logs=print_logs,
            connect_config=False,
        )
        df_corpora_fold, _, pred_scores = evaluateModel(
            model=train_result.model,
            evalData=val,
            evaluation_config=eval_cfg,
            customThresholds=dict(thr_map),
            connect_config=False,
            returnPredictions=True,
        )

        if 'All-micro' in df_corpora_fold.index:
            row = df_corpora_fold.loc['All-micro'].to_dict()
        else:
            row = df_corpora_fold.iloc[0].to_dict()
        micro_rows.append({
            'fold_id': fold_idx,
            'member_idx': member_idx,
            'epochs': int(train_result.epochs_run),
            'Loss': float(train_result.final_dev_loss),
            'Precision': float(row.get('Precision', float('nan'))),
            'Recall': float(row.get('Recall', float('nan'))),
            'F1': float(row.get('F1', float('nan'))),
        })

        class_stats = _per_class_stats_from_predictions(
            pred_wgh_cats=pred_scores,
            eval_corpus=val.corpus,
            threshold_eval=eval_cfg.threshold_eval,
            cat_to_thr=dict(thr_map),
        )
        class_stats_per_member.append(class_stats)

    return micro_rows, class_stats_per_member


def aggregate_per_class_f1(
    *,
    fold_class_data: Sequence[Sequence[Mapping[str, Any]]],
    cat_list: Sequence[str],
    member_labels: Sequence[str],
) -> pd.DataFrame:
    """
    Build long-form ``cat_id, member_idx, member_label, mean_f1, std_f1, n_folds``.

    Classes missing from a fold's ``ClassData`` map (no gold and no
    predictions in that fold) contribute F1=0 to that fold's average.

    :param fold_class_data: ``[member_idx][fold_idx] -> {cat_id: ClassData}``.
    :param cat_list: Reference category list.
    :param member_labels: Member labels indexed by member index.
    """
    rows: list[dict[str, Any]] = []
    for member_idx, member_label in enumerate(member_labels):
        per_member_folds = fold_class_data[member_idx]
        for cat_id in cat_list:
            f1_values: list[float] = []
            p_values: list[float] = []
            r_values: list[float] = []
            for fold_stats in per_member_folds:
                stats = fold_stats.get(cat_id)
                if stats is None:
                    f1_values.append(0.0)
                    p_values.append(0.0)
                    r_values.append(0.0)
                else:
                    f1_values.append(float(stats.fmeasure(beta=1.0)))
                    p_values.append(float(stats.precision))
                    r_values.append(float(stats.recall))
            mean_f1 = float(mean(f1_values)) if f1_values else 0.0
            std_f1 = float(pstdev(f1_values)) if len(f1_values) > 1 else 0.0
            rows.append({
                'cat_id': cat_id,
                'cat_name': _try_cat_name(cat_id),
                'member_idx': member_idx,
                'member_label': member_label,
                'mean_f1': mean_f1,
                'std_f1': std_f1,
                'mean_precision': float(mean(p_values)) if p_values else 0.0,
                'mean_recall': float(mean(r_values)) if r_values else 0.0,
                'n_folds': len(f1_values),
            })
    return pd.DataFrame(rows)


def aggregate_per_fold_f1(
    *,
    fold_class_data: Sequence[Sequence[Mapping[str, Any]]],
    cat_list: Sequence[str],
    member_labels: Sequence[str],
) -> pd.DataFrame:
    """
    Build long-form ``fold_id, member_label, cat_id, F1, P, R, support``.
    """
    rows: list[dict[str, Any]] = []
    for member_idx, member_label in enumerate(member_labels):
        for fold_offset, fold_stats in enumerate(fold_class_data[member_idx]):
            fold_id = fold_offset + 1
            for cat_id in cat_list:
                stats = fold_stats.get(cat_id)
                if stats is None:
                    rows.append({
                        'fold_id': fold_id,
                        'member_label': member_label,
                        'cat_id': cat_id,
                        'cat_name': _try_cat_name(cat_id),
                        'F1': 0.0,
                        'Precision': 0.0,
                        'Recall': 0.0,
                        'support': 0,
                    })
                else:
                    rows.append({
                        'fold_id': fold_id,
                        'member_label': member_label,
                        'cat_id': cat_id,
                        'cat_name': _try_cat_name(cat_id),
                        'F1': float(stats.fmeasure(beta=1.0)),
                        'Precision': float(stats.precision),
                        'Recall': float(stats.recall),
                        'support': int(getattr(stats, 'trueCnt', 0)),
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
    Pick the member with highest mean F1 per class. Ties go to ``primary_idx``.
    """
    grouped: dict[str, dict[int, float]] = defaultdict(dict)
    for _, row in per_class_f1_df.iterrows():
        grouped[str(row['cat_id'])][int(row['member_idx'])] = float(row['mean_f1'])

    assignments: dict[str, int] = {}
    for cat_id in cat_list:
        f1_per_member = grouped.get(str(cat_id), {})
        if not f1_per_member:
            assignments[str(cat_id)] = primary_idx
            continue
        best_f1 = max(f1_per_member.values())
        if f1_per_member.get(primary_idx, float('-inf')) == best_f1:
            assignments[str(cat_id)] = primary_idx
        else:
            assignments[str(cat_id)] = max(f1_per_member.items(), key=lambda kv: kv[1])[0]
    return ClassToModelMap(assignments=assignments, member_labels=tuple(member_labels))


def stitch_thresholds(
    *,
    class_to_model: ClassToModelMap,
    member_thresholds: Sequence[Mapping[str, float]],
    cat_list: Sequence[str],
    default_threshold: float,
) -> dict[str, float]:
    """
    Build per-class thresholds picking each class's threshold from the
    selected member's threshold file. Missing classes use ``default_threshold``.
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
    cat_id, cat_name, F1_mean_<label>, F1_std_<label>, n_folds_<label>, ...,
    selected_member, selected_threshold.
    """
    f1_by_cat: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for _, row in per_class_f1_df.iterrows():
        f1_by_cat[str(row['cat_id'])][int(row['member_idx'])] = {
            'mean_f1': float(row['mean_f1']),
            'std_f1': float(row['std_f1']),
            'n_folds': int(row['n_folds']),
        }

    rows: list[dict[str, Any]] = []
    for cat_id in cat_list:
        cid = str(cat_id)
        per_member = f1_by_cat.get(cid, {})
        record: dict[str, Any] = {
            'cat_id': cid,
            'cat_name': _try_cat_name(cid),
        }
        for member_idx, label in enumerate(member_labels):
            stats = per_member.get(member_idx, {'mean_f1': 0.0, 'std_f1': 0.0, 'n_folds': 0})
            record[f'mean_f1_{label}'] = float(stats['mean_f1'])
            record[f'std_f1_{label}'] = float(stats['std_f1'])
            record[f'n_folds_{label}'] = int(stats['n_folds'])
        selected_idx = class_to_model.assignments.get(cid, 0)
        record['selected_member_idx'] = int(selected_idx)
        record['selected_member_label'] = member_labels[selected_idx]
        record['selected_threshold'] = float(stitched_thresholds.get(cid, float('nan')))
        rows.append(record)
    return pd.DataFrame(rows).sort_values(by='cat_id').reset_index(drop=True)


def build_member_cv_dev_df(
    *,
    micro_rows_per_member: Sequence[Sequence[Mapping[str, Any]]],
    member_labels: Sequence[str],
    member_model_cfgs: Sequence[ModelCnf],
    member_train_cfgs: Sequence[TrainingCnf],
) -> pd.DataFrame:
    """
    Build cv_dev_df with one row per member (no fused row), indexed by
    member label. Columns mirror ``cross_validation.build_cv_df``'s
    ``cv_dev_df`` shape: Precision/Recall/F1/Loss + ``_std`` + epochs/params.
    """
    import json as _json

    rows: list[dict[str, Any]] = []
    for member_idx, label in enumerate(member_labels):
        fold_rows = list(micro_rows_per_member[member_idx])
        if not fold_rows:
            continue
        precisions = [float(r['Precision']) for r in fold_rows]
        recalls = [float(r['Recall']) for r in fold_rows]
        f1s = [float(r['F1']) for r in fold_rows]
        losses = [float(r['Loss']) for r in fold_rows]
        epochs = [float(r['epochs']) for r in fold_rows]
        params = _json.dumps(
            {
                'hidden_dim': member_model_cfgs[member_idx].hidden_dim,
                'dropouts1': member_model_cfgs[member_idx].dropouts1,
                'dropouts2': member_model_cfgs[member_idx].dropouts2,
                'batch_size': member_train_cfgs[member_idx].batch_size,
                'learning_rate': member_train_cfgs[member_idx].learning_rate,
                'member_label': label,
            },
            sort_keys=True,
        )
        rows.append({
            'params': params,
            'epochs': float(np.mean(epochs)),
            'Precision': float(np.mean(precisions)),
            'Recall': float(np.mean(recalls)),
            'F1': float(np.mean(f1s)),
            'Loss': float(np.mean(losses)),
            'Precision_std': float(np.std(precisions, ddof=0)),
            'Recall_std': float(np.std(recalls, ddof=0)),
            'F1_std': float(np.std(f1s, ddof=0)),
            'Loss_std': float(np.std(losses, ddof=0)),
        })
    df = pd.DataFrame(rows)
    df.index = pd.Index(list(member_labels)[:len(rows)], name='member')
    return df


def run_assembly(
    *,
    member_train_data: Sequence[Any],
    member_feature_dims: Sequence[int],
    member_model_cfgs: Sequence[ModelCnf],
    member_train_cfgs: Sequence[TrainingCnf],
    member_thresholds: Sequence[Mapping[str, float]],
    member_labels: Sequence[str],
    eval_cfg: EvaluationCnf,
    cv_cfg: CvCnf,
    objective_corpora: str,
    debug: bool,
    print_logs: bool,
    clearml_logger: Any,
) -> AssemblyCvResult:
    """
    Run K-fold CV training each member per fold and pick per-class winners.

    :param member_train_data: Per-member training datasets (must be aligned
        document-by-document).
    :param member_feature_dims: Per-member input feature dimensions.
    :param member_model_cfgs: Per-member model configs.
    :param member_train_cfgs: Per-member training configs.
    :param member_thresholds: Per-member loaded per-class thresholds.
    :param member_labels: Member labels indexed by member index.
    :param eval_cfg: Shared evaluation config.
    :param cv_cfg: CV split config.
    :param objective_corpora: Objective corpus key (passed through into
        result for downstream consumers).
    :param debug: When True, run a single fold and stop.
    :param print_logs: Forwarded to the trainer.
    :param clearml_logger: ClearML logger for table reporting.
    """
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

    if len(member_train_data) < 2:
        raise ValueError('Assembly requires at least two members')
    cat_list = list(member_train_data[0].corpus.catList)
    docs_primary = len(member_train_data[0].corpus)
    for idx, m_data in enumerate(member_train_data[1:], start=1):
        if len(m_data.corpus) != docs_primary:
            raise ValueError(
                f'Assembly member {idx} has {len(m_data.corpus)} docs vs primary {docs_primary}; '
                f'the cat-list validation step should have caught this earlier'
            )
        if list(m_data.corpus.catList) != cat_list:
            raise ValueError(f'Assembly member {idx} has a different catList than primary')

    x_full, y_full = prepare_cv(train_data=member_train_data[0])
    splitter = MultilabelStratifiedKFold(
        n_splits=cv_cfg.folds,
        shuffle=True,
        random_state=cv_cfg.random_seed,
    )

    micro_rows_per_member: list[list[dict[str, Any]]] = [[] for _ in member_train_data]
    class_data_per_member: list[list[dict[str, Any]]] = [[] for _ in member_train_data]

    folds_planned = 1 if debug else cv_cfg.folds
    LOGGER.info(
        f'Assembly CV plan: members={len(member_train_data)} folds={folds_planned} '
        f'docs={docs_primary} classes={len(cat_list)}'
    )

    for fold_idx, (fit_indices, val_indices) in enumerate(
        splitter.split(x_full, y_full), start=1,
    ):
        fits = [
            slice_dataset(dataset=m_data, indices=fit_indices.tolist())
            for m_data in member_train_data
        ]
        vals = [
            slice_dataset(dataset=m_data, indices=val_indices.tolist())
            for m_data in member_train_data
        ]
        micro_rows, class_stats = evaluate_assembly_fold(
            fits=fits,
            vals=vals,
            feature_dims=member_feature_dims,
            member_model_cfgs=member_model_cfgs,
            member_train_cfgs=member_train_cfgs,
            member_thresholds=member_thresholds,
            eval_cfg=eval_cfg,
            fold_idx=fold_idx,
            print_logs=print_logs,
        )
        for m_idx, (mr, cs) in enumerate(zip(micro_rows, class_stats)):
            micro_rows_per_member[m_idx].append(mr)
            class_data_per_member[m_idx].append(cs)
            clearml_logger.report_text(
                f'Assembly fold {fold_idx}/{folds_planned} '
                f'member={member_labels[m_idx]} '
                f'micro F1={mr["F1"]:.4f} P={mr["Precision"]:.4f} R={mr["Recall"]:.4f} '
                f'epochs={mr["epochs"]}',
                print_console=print_logs,
            )
        if debug:
            break

    per_class_f1_df = aggregate_per_class_f1(
        fold_class_data=class_data_per_member,
        cat_list=cat_list,
        member_labels=member_labels,
    )
    per_fold_f1_df = aggregate_per_fold_f1(
        fold_class_data=class_data_per_member,
        cat_list=cat_list,
        member_labels=member_labels,
    )
    class_to_model = select_class_to_model(
        per_class_f1_df=per_class_f1_df,
        cat_list=cat_list,
        member_labels=member_labels,
        primary_idx=0,
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
        micro_rows_per_member=micro_rows_per_member,
        member_labels=member_labels,
        member_model_cfgs=member_model_cfgs,
        member_train_cfgs=member_train_cfgs,
    )

    objective_metrics: dict[str, Any] = {
        'objective_corpora': objective_corpora,
        'n_classes': len(cat_list),
        'n_classes_selected_per_member': {
            label: int(sum(1 for v in class_to_model.assignments.values() if v == idx))
            for idx, label in enumerate(member_labels)
        },
    }
    for member_idx, label in enumerate(member_labels):
        rows = micro_rows_per_member[member_idx]
        if not rows:
            continue
        f1s = [r['F1'] for r in rows]
        objective_metrics[f'F1_mean_{label}'] = float(np.mean(f1s))
        objective_metrics[f'F1_std_{label}'] = float(np.std(f1s, ddof=0))

    _report_assembly_tables(
        clearml_logger=clearml_logger,
        per_class_f1_df=per_class_f1_df,
        per_fold_f1_df=per_fold_f1_df,
        threshold_report_df=threshold_report_df,
        cv_dev_df=cv_dev_df,
        class_to_model=class_to_model,
        member_labels=member_labels,
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
        per_fold_f1_df=per_fold_f1_df,
    )


def _report_assembly_tables(
    *,
    clearml_logger: Any,
    per_class_f1_df: pd.DataFrame,
    per_fold_f1_df: pd.DataFrame,
    threshold_report_df: pd.DataFrame,
    cv_dev_df: pd.DataFrame,
    class_to_model: ClassToModelMap,
    member_labels: Sequence[str],
) -> None:
    """Emit the standard assembly tables to ClearML."""
    clearml_logger.report_table(
        title='Assembly',
        series='Per-fold per-class F1',
        iteration=0,
        table_plot=per_fold_f1_df,
    )
    clearml_logger.report_table(
        title='Assembly',
        series='Per-class F1 aggregate',
        iteration=0,
        table_plot=per_class_f1_df,
    )
    clearml_logger.report_table(
        title='Assembly',
        series='Class-to-model',
        iteration=0,
        table_plot=threshold_report_df,
    )
    clearml_logger.report_table(
        title='Assembly',
        series='Member CV summary',
        iteration=0,
        table_plot=cv_dev_df,
    )
    for member_idx, label in enumerate(member_labels):
        n_selected = int(sum(1 for v in class_to_model.assignments.values() if v == member_idx))
        clearml_logger.report_scalar(
            title='Assembly / classes per member',
            series=label,
            value=n_selected,
            iteration=0,
        )


def member_cfg_for_train_best(*, train_cfg: TrainingCnf) -> TrainingCnf:
    """Return a copy of ``train_cfg`` with early stopping disabled.

    ``train_best`` enforces ``early_stopping_patience == 0`` so the
    test-set is not used as a validation signal. Assembly skips HPO and
    has no CV-derived epoch reduction, so we simply turn early stopping
    off at the call site.
    """
    return replace(train_cfg, early_stopping_patience=0)
