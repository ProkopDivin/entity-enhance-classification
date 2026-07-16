"""Tests for article-record extraction and entity impact aggregation in the comparison module."""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

import iptc_entity_pipeline.evaluation.comparison as ec


def test_build_article_records_parses_entities(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ec, 'norm_cat_ids', lambda *, cat_ids: ['01000000'])

    eval_corpus = [
        SimpleNamespace(
            id='doc-1',
            cats=['ignored'],
            text='hello world article',
            metadata={
                'corpusName': 'en_test',
                'article_length': '123',
                'entities': [
                    {
                        'gkbId': 'gkb:1',
                        'wdid': 'Q1|Q2',
                        'stdForm': 'European Union',
                        'type': 'organization',
                        'relevance': '0.75',
                        'mentions': '3',
                    }
                ],
            },
        )
    ]
    pred_scores = [[('01000000', 0.9)]]

    records = ec.build_article_records(eval_corpus=eval_corpus, pred_scores=pred_scores)

    assert len(records) == 1
    record = records[0]
    assert record.article_id == 'doc-1'
    assert record.corpus_name == 'en_test'
    assert record.gold_categories == ('01000000',)
    assert record.pred_scores == (('01000000', 0.9),)
    assert record.article_text == 'hello world article'
    assert record.article_length == 123
    assert len(record.entities) == 1
    entity = record.entities[0]
    assert entity.gkb_id == 'gkb:1'
    assert entity.wdids == ('Q1', 'Q2')
    assert entity.std_form == 'European Union'
    assert entity.entity_type == 'organization'
    assert entity.relevance == 0.75
    assert entity.mention_count == 3


def test_build_article_records_handles_missing_entities(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ec, 'norm_cat_ids', lambda *, cat_ids: [])

    eval_corpus = [SimpleNamespace(id='doc-2', cats=[], metadata={'corpusName': 'cs_test'})]
    pred_scores = [[('02000000', 0.4)]]

    records = ec.build_article_records(eval_corpus=eval_corpus, pred_scores=pred_scores)

    assert len(records) == 1
    record = records[0]
    assert record.article_text is None
    assert record.article_length is None
    assert record.entities == ()


def test_build_article_records_extracts_fields_from_linked_entity_raw_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ec, 'norm_cat_ids', lambda *, cat_ids: [])

    linked_entity = SimpleNamespace(
        gkb_id='gkb:2',
        wd_ids=('Q10', 'Q11'),
        raw_entity={
            'stdForm': 'OpenAI',
            'type': 'organization',
            'feats': {'relevance': '88'},
            'mentions': [{'id': 'm1'}, {'id': 'm2'}],
        },
    )
    eval_corpus = [
        SimpleNamespace(
            id='doc-3',
            cats=[],
            metadata={'corpusName': 'en_test'},
            entities=[linked_entity],
        )
    ]
    pred_scores = [[('02000000', 0.4)]]

    records = ec.build_article_records(eval_corpus=eval_corpus, pred_scores=pred_scores)

    assert len(records) == 1
    entity = records[0].entities[0]
    assert entity.gkb_id == 'gkb:2'
    assert entity.wdids == ('Q10', 'Q11')
    assert entity.std_form == 'OpenAI'
    assert entity.entity_type == 'organization'
    assert entity.relevance == 88.0
    assert entity.mention_count == 2


def test_records_to_df_keeps_expected_alignment_columns() -> None:
    records = [
        ec.ArticleEvalRecord(
            article_id='a1',
            corpus_name='en_a',
            gold_categories=('01000000',),
            pred_scores=(('01000000', 0.8), ('02000000', 0.1)),
            article_text='text-a1',
            article_length=80,
            entities=(),
        ),
        ec.ArticleEvalRecord(
            article_id='a2',
            corpus_name='en_a',
            gold_categories=('02000000',),
            pred_scores=(('02000000', 0.9),),
            article_text=None,
            article_length=None,
            entities=(),
        ),
    ]

    df = ec.records_to_df(records=records)

    assert {'article_id', 'corpus_name', 'gold_categories', 'pred_scores'}.issubset(set(df.columns))
    assert {'prob_01000000', 'prob_02000000'}.issubset(set(df.columns))
    assert df.loc[0, 'article_text'] == 'text-a1'
    assert df.loc[1, 'article_text'] is None
    assert df.loc[0, 'prob_01000000'] == pytest.approx(0.8)
    assert df.loc[1, 'prob_01000000'] == pytest.approx(0.0)


def test_entity_impact_all_table() -> None:
    """Entity impact table must aggregate per-entity F1 deltas, sort by ``entity_score``,
    append an AVG footer row, expose the columns defined by ``entity_impact_columns()``,
    and be deterministic across calls with identical inputs.
    """
    gold_map = ec.GoldLabelMap(
        df=pd.DataFrame(),
        article_map={
            'a1': ec.GoldArticle(article_id='a1', corpus_name='c', gold_categories=('01000000',)),
            'a2': ec.GoldArticle(article_id='a2', corpus_name='c', gold_categories=('01000000',)),
        },
    )
    current_df = pd.DataFrame(
        {
            'article_id': ['a1', 'a2'],
            'corpus_name': ['c', 'c'],
            'prob_01000000': [0.9, 0.1],
            'entities': [
                [
                    ec.ArticleEntity(
                        gkb_id='gkb:1',
                        wdids=(),
                        entity_type='person',
                        std_form='Name One',
                        relevance=0.5,
                        mention_count=2,
                    ),
                    ec.ArticleEntity(
                        gkb_id='gkb:1',
                        wdids=(),
                        entity_type='person',
                        std_form='Name One',
                        relevance=0.7,
                        mention_count=4,
                    ),
                    ec.ArticleEntity(
                        gkb_id='gkb:2',
                        wdids=(),
                        entity_type='org',
                        std_form='Name Two',
                        relevance=0.2,
                        mention_count=1,
                    ),
                ],
                [
                    ec.ArticleEntity(
                        gkb_id='gkb:1',
                        wdids=(),
                        entity_type='person',
                        std_form='Name One',
                        relevance=0.3,
                        mention_count=1,
                    ),
                    ec.ArticleEntity(
                        gkb_id='gkb:3',
                        wdids=(),
                        entity_type='org',
                        std_form='Name Three',
                        relevance=0.9,
                        mention_count=5,
                    ),
                ],
            ],
        }
    )
    base_df = pd.DataFrame(
        {
            'article_id': ['a1', 'a2'],
            'corpus_name': ['c', 'c'],
            'prob_01000000': [0.1, 0.9],
            'entities': [[], []],
        }
    )

    cat_ids = ['01000000']
    thr_vec = ec.thresholds_vector(cat_ids=cat_ids, cat_to_thr=None, default_threshold=0.5)
    article_f1_df = ec.build_article_f1_diff_df(
        current_df=current_df,
        base_df=base_df,
        gold_map=gold_map,
        cat_ids=cat_ids,
        current_thr_vec=thr_vec,
        base_thr_vec=thr_vec,
    )
    # a1: base predicts 0, current predicts 1 (gold=1)  -> f1_diff = +1
    # a2: base predicts 1, current predicts 0 (gold=1)  -> f1_diff = -1
    assert list(article_f1_df['f1_diff']) == pytest.approx([1.0, -1.0])

    all_first = ec.build_entity_impact_all_df(current_df=current_df, article_f1_df=article_f1_df)

    assert list(all_first.columns) == ec.entity_impact_columns()

    non_footer = all_first[all_first['gkbid'] != 'AVG']
    assert set(non_footer['gkbid']) == {'gkb:1', 'gkb:2', 'gkb:3'}

    scores = list(non_footer['entity_score'])
    assert scores == sorted(scores, reverse=True), 'entities must be sorted by entity_score desc'

    top_row = non_footer.iloc[0]
    assert top_row['gkbid'] == 'gkb:2'
    assert top_row['entity_type'] == 'org'
    assert top_row['entity_score'] == pytest.approx(1.0)
    assert top_row['normalized'] == pytest.approx(1.0)

    for _, row in non_footer.iterrows():
        assert row['normalized'] == pytest.approx(row['entity_score'] / row['article_count'])

    footer = all_first.iloc[-1]
    assert footer['gkbid'] == 'AVG'
    assert footer['article_count'] == pytest.approx(non_footer['article_count'].mean())

    all_second = ec.build_entity_impact_all_df(current_df=current_df, article_f1_df=article_f1_df)
    assert all_first.equals(all_second), 'aggregation must be deterministic across identical calls'
