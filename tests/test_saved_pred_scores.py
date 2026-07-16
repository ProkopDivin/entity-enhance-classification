from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from geneea.catlib.data import Corpus, Doc

import iptc_entity_pipeline.evaluation.comparison as ec
from iptc_entity_pipeline.evaluation.comparison import compare_runs, load_run, load_custom_thresholds
from iptc_entity_pipeline.model_io import EVAL_CORPUS_FILENAME, PREDICTIONS_FILENAME

CAT_A = '01000000'
CAT_B = '02000000'


def build_corpus() -> Corpus:
    docs = [
        Doc(id='a1', title='', lead='', text='', cats=[CAT_A], metadata={'corpusName': 'corpus_a'}),
        Doc(id='a2', title='', lead='', text='', cats=[CAT_B], metadata={'corpusName': 'corpus_a'}),
        Doc(id='a3', title='', lead='', text='', cats=[CAT_A, CAT_B], metadata={'corpusName': 'corpus_b'}),
    ]
    return Corpus(docs)


def build_current_predictions() -> list[list[tuple[str, float]]]:
    return [
        [(CAT_A, 0.9), (CAT_B, 0.1)],
        [(CAT_A, 0.4), (CAT_B, 0.8)],
        [(CAT_A, 0.8), (CAT_B, 0.7)],
    ]


def build_base_predictions() -> list[list[tuple[str, float]]]:
    return [
        [(CAT_A, 0.8), (CAT_B, 0.2)],
        [(CAT_A, 0.6), (CAT_B, 0.4)],
        [(CAT_A, 0.7), (CAT_B, 0.3)],
    ]


def write_run(*, run_dir: Path, predictions: list, corpus: Corpus) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / PREDICTIONS_FILENAME, 'wb') as f:
        pickle.dump(predictions, f)
    with open(run_dir / EVAL_CORPUS_FILENAME, 'wb') as f:
        pickle.dump(corpus, f)


def _patch_small_category_subsets(*, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ec, 'load_tail_cat_ids', lambda: {CAT_A, CAT_B})
    monkeypatch.setattr(ec, 'load_relevant_cat_ids', lambda: {CAT_A, CAT_B})


def test_load_run_reads_pickled_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / 'run'
    write_run(run_dir=run_dir, predictions=build_current_predictions(), corpus=build_corpus())

    pred_scores, eval_corpus = load_run(run_dir=run_dir)

    assert len(pred_scores) == 3
    assert pred_scores[0] == [(CAT_A, 0.9), (CAT_B, 0.1)]
    assert [doc.id for doc in eval_corpus] == ['a1', 'a2', 'a3']


def test_compare_runs_consumes_run_directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_small_category_subsets(monkeypatch=monkeypatch)
    current_dir = tmp_path / 'current'
    base_dir = tmp_path / 'base'
    write_run(run_dir=current_dir, predictions=build_current_predictions(), corpus=build_corpus())
    write_run(run_dir=base_dir, predictions=build_base_predictions(), corpus=build_corpus())

    output_path = tmp_path / 'comparison.xlsx'
    result = compare_runs(
        current_run_dir=current_dir,
        base_run_dir=base_dir,
        threshold_eval=0.5,
        output_path=output_path,
    )

    assert result.excel_path == output_path
    assert result.excel_path.exists()
    assert not result.summary_comparison.empty
    micro_row = result.summary_comparison.loc[result.summary_comparison['summary_key'] == 'micro'].iloc[0]
    assert micro_row['f1_current'] > micro_row['f1_base']


def test_compare_runs_can_persist_only_top_change_tables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_small_category_subsets(monkeypatch=monkeypatch)
    current_dir = tmp_path / 'current'
    base_dir = tmp_path / 'base'
    write_run(run_dir=current_dir, predictions=build_current_predictions(), corpus=build_corpus())
    write_run(run_dir=base_dir, predictions=build_base_predictions(), corpus=build_corpus())
    output_path = tmp_path / 'comparison.xlsx'
    result = compare_runs(
        current_run_dir=current_dir,
        base_run_dir=base_dir,
        threshold_eval=0.5,
        top_changes_only=True,
        output_path=output_path,
    )

    assert result.excel_path == output_path
    assert (tmp_path / 'top_improved.csv').exists()
    assert (tmp_path / 'top_degraded.csv').exists()
    assert not (tmp_path / 'corpora_comparison.csv').exists()
    assert pd.ExcelFile(output_path).sheet_names == ['top_improved', 'top_degraded']
    assert 'mcnemar_pass' in result.top_improved_categories.columns
    assert 'mcnemar_p_value' in result.top_improved_categories.columns
    assert 'mcnemar_p_value_fdr' in result.top_improved_categories.columns
    assert 'mcnemar_skipped' not in result.top_improved_categories.columns
    assert 'ranking_score' not in result.top_improved_categories.columns
    # This fixture has only 3 articles, so per-class current/base disagreements
    # fall below MCNEMAR_MIN_DISAGREEMENTS and the test is skipped -> NaN p-value.
    assert pd.isna(result.top_improved_categories['mcnemar_p_value'].iloc[0])
    assert pd.isna(result.top_improved_categories['mcnemar_p_value_fdr'].iloc[0])


def test_build_mcnemar_detects_current_improvement() -> None:
    article_ids = [f'a{i}' for i in range(30)]
    current_df = pd.DataFrame(
        {
            'article_id': article_ids,
            'prob_01000000': [0.9] * len(article_ids),
        }
    )
    base_df = pd.DataFrame(
        {
            'article_id': article_ids,
            'prob_01000000': [0.1] * len(article_ids),
        }
    )
    gold_map = ec.GoldLabelMap(
        df=pd.DataFrame(),
        article_map={
            article_id: ec.GoldArticle(article_id=article_id, corpus_name='c', gold_categories=('01000000',))
            for article_id in article_ids
        },
    )

    cat_ids = ['01000000']
    thr_vec = ec.thresholds_vector(cat_ids=cat_ids, cat_to_thr=None, default_threshold=0.5)
    result = ec.build_mcnemar_significance_df(
        current_df=current_df,
        base_df=base_df,
        gold_map=gold_map,
        cat_ids=cat_ids,
        current_thr_vec=thr_vec,
        base_thr_vec=thr_vec,
    )

    row = result.iloc[0]
    assert row['mcnemar_n10_current_only_correct'] == 30
    assert row['mcnemar_n01_base_only_correct'] == 0
    assert 'mcnemar_skipped' not in result.columns
    assert row['mcnemar_p_value'] < 0.05
    assert row['mcnemar_p_value_fdr'] == pytest.approx(row['mcnemar_p_value'])
    assert row['mcnemar_current_significant'] == 1
    assert row['mcnemar_base_significant'] == 0


def test_benjamini_hochberg_preserves_order_and_monotonicity() -> None:
    corrected = ec.benjamini_hochberg(p_values=[0.01, 0.04, 0.03])

    assert corrected.tolist() == pytest.approx([0.03, 0.04, 0.04])


def test_load_custom_thresholds_prefers_new_filename(tmp_path: Path) -> None:
    new_payload = {CAT_A: 0.61, CAT_B: 0.42}
    legacy_payload = {CAT_A: 0.99}
    (tmp_path / 'custom_thresholds.json').write_text(json.dumps(new_payload), encoding='utf-8')
    (tmp_path / 'thresholds.json').write_text(json.dumps(legacy_payload), encoding='utf-8')

    loaded = load_custom_thresholds(run_dir=tmp_path)

    assert loaded == {CAT_A: 0.61, CAT_B: 0.42}


def test_load_custom_thresholds_falls_back_to_legacy_filename(tmp_path: Path) -> None:
    legacy_payload = {CAT_A: 0.55, CAT_B: 0.31}
    (tmp_path / 'thresholds.json').write_text(json.dumps(legacy_payload), encoding='utf-8')

    loaded = load_custom_thresholds(run_dir=tmp_path)

    assert loaded == {CAT_A: 0.55, CAT_B: 0.31}


def test_load_custom_thresholds_missing_returns_empty(tmp_path: Path) -> None:
    assert load_custom_thresholds(run_dir=tmp_path) == {}


def test_thresholds_vector_uses_default_for_unmapped_classes() -> None:
    vec = ec.thresholds_vector(
        cat_ids=['A', 'B', 'C'],
        cat_to_thr={'A': 0.7, 'C': 0.2},
        default_threshold=0.5,
    )

    assert vec.tolist() == [0.7, 0.5, 0.2]


def test_compare_runs_loads_legacy_thresholds_filename(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_small_category_subsets(monkeypatch=monkeypatch)
    current_dir = tmp_path / 'current'
    base_dir = tmp_path / 'base'
    write_run(run_dir=current_dir, predictions=build_current_predictions(), corpus=build_corpus())
    write_run(run_dir=base_dir, predictions=build_base_predictions(), corpus=build_corpus())
    (current_dir / 'thresholds.json').write_text(json.dumps({CAT_A: 0.99, CAT_B: 0.99}), encoding='utf-8')

    output_path = tmp_path / 'comparison.xlsx'
    result = compare_runs(
        current_run_dir=current_dir,
        base_run_dir=base_dir,
        threshold_eval=0.5,
        output_path=output_path,
    )

    thresholds = result.class_thresholds.set_index('class_id')
    assert thresholds.loc[CAT_A, 'threshold_current'] == pytest.approx(0.99)
    assert thresholds.loc[CAT_A, 'threshold_base'] == pytest.approx(0.5)


def test_compare_runs_can_ignore_saved_thresholds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_small_category_subsets(monkeypatch=monkeypatch)
    current_dir = tmp_path / 'current'
    base_dir = tmp_path / 'base'
    write_run(run_dir=current_dir, predictions=build_current_predictions(), corpus=build_corpus())
    write_run(run_dir=base_dir, predictions=build_base_predictions(), corpus=build_corpus())
    (current_dir / 'custom_thresholds.json').write_text(json.dumps({CAT_A: 0.99, CAT_B: 0.99}), encoding='utf-8')

    output_path = tmp_path / 'comparison.xlsx'
    forced_uniform = compare_runs(
        current_run_dir=current_dir,
        base_run_dir=base_dir,
        threshold_eval=0.5,
        output_path=output_path,
        use_saved_thresholds=False,
    )
    autoload = compare_runs(
        current_run_dir=current_dir,
        base_run_dir=base_dir,
        threshold_eval=0.5,
        output_path=tmp_path / 'comparison_loaded.xlsx',
    )

    forced_thr = forced_uniform.class_thresholds.set_index('class_id')
    autoload_thr = autoload.class_thresholds.set_index('class_id')
    assert forced_thr.loc[CAT_A, 'threshold_current'] != autoload_thr.loc[CAT_A, 'threshold_current']
