"""Tests for data-loading helpers backed by real-data samples."""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest

from iptc_entity_pipeline.data_loading import (
    DocWithEntities,
    EntityType,
    LinkedEntity,
    attach_entities,
    count_unmapped_entities,
    filter_linked_entities_by_type,
    get_article_text,
    get_doc_wdids,
    remove_types_except,
    get_doc_wdid_mention_counts,
    get_doc_weighted_wdids,
    load_wdid_map,
    parse_remove_types,
)

RESOURCE_DIR = Path(__file__).parent / 'resources' / 'data_loading'
WDID_MAPPING_SAMPLE = RESOURCE_DIR / 'wdid_mapping_sample.tsv'
ENTITIES_SAMPLE = RESOURCE_DIR / 'entities_sample.csv'
EXPECTED_WDID_MAPPING = {
    'G110989308': ['Q110989308'],
    'G2328646': ['Q2328646'],
    'G98219720': ['Q98219720'],
    'G15964180': ['Q15964180'],
    'G752762': ['Q3306663', 'Q6820684', 'Q752762'],
    'G47499': ['Q10901070', 'Q47499'],
    'G1490400': ['Q1490400', 'Q2067115'],
    'G736742': ['Q736742', 'Q874319'],
}


class FakeCorpus:
    """Minimal corpus shim for attach_entities_to_corpus tests."""

    def __init__(self, *, docs: list[SimpleNamespace]) -> None:
        self.docs = docs

    def __len__(self) -> int:
        return len(self.docs)


def _fake_from_doc(*, doc: SimpleNamespace, entities: list[LinkedEntity]) -> SimpleNamespace:
    """Return lightweight doc object used in tests."""
    return SimpleNamespace(id=doc.id, entities=entities)


def _load_sample_article_ids() -> list[str]:
    """Read article IDs from entities sample CSV."""
    article_ids: list[str] = []
    with ENTITIES_SAMPLE.open(mode='r', encoding='utf-8', newline='') as in_file:
        reader = csv.DictReader(in_file)
        for row in reader:
            article_id = (row.get('id') or '').strip()
            if article_id:
                article_ids.append(article_id)
    return article_ids


def _build_enriched_corpus(
    *,
    monkeypatch: pytest.MonkeyPatch,
    mapping: dict[str, list[str]],
    min_relevance: float = 0.0,
) -> FakeCorpus:
    """Create fake corpus docs and enrich them from real-data entity CSV sample."""
    monkeypatch.setattr(DocWithEntities, 'from_doc', _fake_from_doc)
    article_ids = _load_sample_article_ids()
    docs = [SimpleNamespace(id=article_id) for article_id in article_ids]
    docs.append(SimpleNamespace(id='missing_article_in_fixture'))
    corpus = FakeCorpus(docs=docs)
    attach_entities(
        corpus=corpus,
        csv_path=str(ENTITIES_SAMPLE),
        wdid_mapping=mapping,
        min_relevance=min_relevance,
    )
    return corpus


def _expected_relevances_per_article(*, min_relevance: float) -> dict[str, list[float]]:
    """Compute expected per-article relevance lists by parsing the sample CSV directly."""
    expected: dict[str, list[float]] = {}
    with ENTITIES_SAMPLE.open(mode='r', encoding='utf-8', newline='') as in_file:
        reader = csv.DictReader(in_file)
        for row in reader:
            article_id = (row.get('id') or '').strip()
            entities_json = (row.get('entities') or '').strip()
            if not article_id or not entities_json:
                continue
            kept: list[float] = []
            for ent in json.loads(entities_json):
                if not isinstance(ent, Mapping) or not ent.get('gkbId'):
                    continue
                rel_raw = ent.get('relevance')
                if rel_raw is None:
                    feats = ent.get('feats', {})
                    rel_raw = feats.get('relevance', 0.0) if isinstance(feats, Mapping) else 0.0
                try:
                    rel = float(rel_raw)
                except (TypeError, ValueError):
                    rel = 0.0
                if rel >= min_relevance:
                    kept.append(rel)
            expected[article_id] = kept
    return expected


def test_load_wdid_mapping_uses_real_sample_file() -> None:
    loaded = load_wdid_map(wdid_mapping_tsv=str(WDID_MAPPING_SAMPLE))

    assert loaded == EXPECTED_WDID_MAPPING
    assert any(len(wdids) == 1 for wdids in loaded.values())
    assert any(len(wdids) > 1 for wdids in loaded.values())


def test_attach_entities_to_corpus_uses_real_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = load_wdid_map(wdid_mapping_tsv=str(WDID_MAPPING_SAMPLE))
    corpus = _build_enriched_corpus(monkeypatch=monkeypatch, mapping=mapping)

    for doc in corpus.docs[:-1]:
        assert doc.entities
        for ent in doc.entities:
            assert isinstance(ent, LinkedEntity)
            assert ent.wd_ids == tuple(mapping.get(ent.gkb_id, ()))
            assert isinstance(ent.mention_count, int)
            assert ent.mention_count >= 1
            assert isinstance(ent.relevance, float)

    assert corpus.docs[-1].entities == []


def test_get_doc_weighted_wdids_from_enriched_real_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = load_wdid_map(wdid_mapping_tsv=str(WDID_MAPPING_SAMPLE))
    corpus = _build_enriched_corpus(monkeypatch=monkeypatch, mapping=mapping)
    doc = next(doc for doc in corpus.docs if any(len(ent.wd_ids) > 1 for ent in doc.entities))

    weighted = get_doc_weighted_wdids(doc)
    assert weighted
    assert len(weighted) == sum(len(ent.wd_ids) for ent in doc.entities if ent.wd_ids)
    assert sum(weight for _wdid, weight in weighted) == pytest.approx(
        sum(ent.relevance for ent in doc.entities if ent.wd_ids),
    )

    multi_ent = next(ent for ent in doc.entities if len(ent.wd_ids) > 1)
    split_weight = multi_ent.relevance / len(multi_ent.wd_ids)
    for wd_id in multi_ent.wd_ids:
        assert (wd_id, split_weight) in weighted


def test_get_doc_wdids_from_enriched_real_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = load_wdid_map(wdid_mapping_tsv=str(WDID_MAPPING_SAMPLE))
    corpus = _build_enriched_corpus(monkeypatch=monkeypatch, mapping=mapping)
    doc = next(doc for doc in corpus.docs if doc.entities)

    wdids = get_doc_wdids(doc)
    assert wdids
    assert wdids == [wd_id for ent in doc.entities for wd_id in ent.wd_ids]


def test_get_doc_wdid_mention_counts_from_enriched_real_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = load_wdid_map(wdid_mapping_tsv=str(WDID_MAPPING_SAMPLE))
    corpus = _build_enriched_corpus(monkeypatch=monkeypatch, mapping=mapping)
    doc = next(doc for doc in corpus.docs if any(len(ent.wd_ids) > 1 for ent in doc.entities))

    mention_counts = dict(get_doc_wdid_mention_counts(doc))
    assert mention_counts
    assert sum(mention_counts.values()) == pytest.approx(
        sum(ent.mention_count for ent in doc.entities if ent.wd_ids),
    )

    multi_ent = next(ent for ent in doc.entities if len(ent.wd_ids) > 1)
    expected_split = multi_ent.mention_count / len(multi_ent.wd_ids)
    for wd_id in multi_ent.wd_ids:
        assert mention_counts[wd_id] >= expected_split


@pytest.mark.parametrize('min_relevance', [0.0, 1.0, 25.0, 50.0, 90.0])
def test_attach_entities_min_relevance_matches_csv_ground_truth(
    monkeypatch: pytest.MonkeyPatch,
    min_relevance: float,
) -> None:
    """Thresholding against real CSV must match a hand-computed reference per article."""
    mapping = load_wdid_map(wdid_mapping_tsv=str(WDID_MAPPING_SAMPLE))
    corpus = _build_enriched_corpus(monkeypatch=monkeypatch, mapping=mapping, min_relevance=min_relevance)
    expected = _expected_relevances_per_article(min_relevance=min_relevance)

    for doc in corpus.docs[:-1]:
        kept = sorted(ent.relevance for ent in doc.entities)
        assert kept == sorted(expected[doc.id]), (
            f'mismatch for article {doc.id} at threshold={min_relevance}: '
            f'got {len(kept)} kept, expected {len(expected[doc.id])}'
        )
        assert all(ent.relevance >= min_relevance for ent in doc.entities)


def test_attach_entities_min_relevance_monotone_decreasing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Increasing the threshold must never increase the number of kept entities per doc."""
    mapping = load_wdid_map(wdid_mapping_tsv=str(WDID_MAPPING_SAMPLE))
    counts_by_threshold: dict[float, dict[str, int]] = {}
    for thr in (0.0, 10.0, 50.0, 95.0):
        corpus = _build_enriched_corpus(monkeypatch=monkeypatch, mapping=mapping, min_relevance=thr)
        counts_by_threshold[thr] = {doc.id: len(doc.entities) for doc in corpus.docs}

    article_ids = list(counts_by_threshold[0.0])
    sorted_thresholds = sorted(counts_by_threshold)
    for lower, higher in zip(sorted_thresholds, sorted_thresholds[1:]):
        for article_id in article_ids:
            assert counts_by_threshold[higher][article_id] <= counts_by_threshold[lower][article_id]


def test_attach_entities_min_relevance_inclusive_at_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An entity with relevance == threshold must be kept (inclusive boundary)."""
    mapping = load_wdid_map(wdid_mapping_tsv=str(WDID_MAPPING_SAMPLE))
    baseline = _expected_relevances_per_article(min_relevance=0.0)
    boundary_value = next(
        rel for rels in baseline.values() for rel in rels if rel > 0.0
    )

    corpus = _build_enriched_corpus(monkeypatch=monkeypatch, mapping=mapping, min_relevance=boundary_value)
    kept_relevances = [ent.relevance for doc in corpus.docs for ent in doc.entities]

    assert boundary_value in kept_relevances
    assert all(rel >= boundary_value for rel in kept_relevances)


def test_attach_entities_min_relevance_above_max_yields_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A threshold above any observed relevance must drop all entities."""
    mapping = load_wdid_map(wdid_mapping_tsv=str(WDID_MAPPING_SAMPLE))
    corpus = _build_enriched_corpus(monkeypatch=monkeypatch, mapping=mapping, min_relevance=10_000.0)

    for doc in corpus.docs:
        assert doc.entities == []


def test_attach_entities_supports_inline_wdid_format(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Gold-origin entities carry ``wdId`` directly instead of ``gkbId``."""
    monkeypatch.setattr(DocWithEntities, 'from_doc', _fake_from_doc)
    entities_json = json.dumps(
        [
            {
                'id': 'e0',
                'wdId': 'Q162990',
                'type': 'organization',
                'mentions': [{'id': 'm0', 'text': 'Cleveland'}],
            }
        ]
    )
    csv_path = tmp_path / 'inline_wdid_entities.csv'
    with csv_path.open(mode='w', encoding='utf-8', newline='') as tmp:
        writer = csv.writer(tmp)
        writer.writerow(['id', 'entities'])
        writer.writerow(['article_1', entities_json])

    corpus = FakeCorpus(docs=[SimpleNamespace(id='article_1')])
    attach_entities(corpus=corpus, csv_path=str(csv_path), wdid_mapping={}, min_relevance=0.0)

    assert len(corpus.docs[0].entities) == 1
    ent = corpus.docs[0].entities[0]
    assert ent.gkb_id == 'Q162990'
    assert ent.wd_ids == ('Q162990',)
    assert ent.mention_count == 1
    assert ent.entity_type == EntityType.ORGANIZATION


@pytest.mark.parametrize(
    ('entities_payload', 'expected_type'),
    [
        (
            [{'gkbId': 'G1', 'type': 'location', 'feats': {'relevance': '50'}}],
            'location',
        ),
        (
            [{'gkbId': 'G1', 'type': 'other', 'feats': {'detectedType': 'organization', 'relevance': '50'}}],
            'organization',
        ),
        (
            [{'gkbId': 'G1', 'type': 'weird', 'feats': {'relevance': '50'}}],
            'other',
        ),
        (
            [{'gkbId': 'G1', 'type': 'other', 'feats': {'detectedType': 'weird', 'relevance': '50'}}],
            'other',
        ),
        (
            [{'gkbId': 'G1', 'feats': {'relevance': '50'}}],
            'other',
        ),
    ],
)
def test_resolve_entity_type(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    entities_payload: list[Mapping[str, object]],
    expected_type: str,
) -> None:
    """Entity type uses known labels, detectedType fallback, otherwise other."""
    monkeypatch.setattr(DocWithEntities, 'from_doc', _fake_from_doc)
    entities_json = json.dumps(entities_payload)
    csv_path = tmp_path / 'entities.csv'
    with csv_path.open(mode='w', encoding='utf-8', newline='') as tmp:
        writer = csv.writer(tmp)
        writer.writerow(['id', 'entities'])
        writer.writerow(['article_1', entities_json])

    corpus = FakeCorpus(docs=[SimpleNamespace(id='article_1')])
    attach_entities(
        corpus=corpus,
        csv_path=str(csv_path),
        wdid_mapping={'G1': ['Q1']},
        min_relevance=0.0,
    )

    assert corpus.docs[0].entities[0].entity_type == EntityType(expected_type)


def test_filter_linked_entities_by_type_removes_matching_types() -> None:
    entities = [
        LinkedEntity(gkb_id='g1', wd_ids=('Q1',), relevance=1.0, entity_type=EntityType.LOCATION),
        LinkedEntity(gkb_id='g2', wd_ids=('Q2',), relevance=1.0, entity_type=EntityType.ORGANIZATION),
        LinkedEntity(gkb_id='g3', wd_ids=('Q3',), relevance=1.0, entity_type=EntityType.OTHER),
    ]

    filtered = filter_linked_entities_by_type(
        entities=entities,
        remove_types=frozenset({EntityType.ORGANIZATION, EntityType.OTHER}),
    )

    assert len(filtered) == 1
    assert filtered[0].gkb_id == 'g1'
    assert filtered[0].entity_type == EntityType.LOCATION


def test_attach_entities_logs_entity_type_counts_without_remove_types(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """Entity type counts are logged even when remove_types is empty."""
    monkeypatch.setattr(DocWithEntities, 'from_doc', _fake_from_doc)
    entities_json = json.dumps(
        [
            {'gkbId': 'G1', 'type': 'location', 'feats': {'relevance': '50'}},
            {'gkbId': 'G2', 'type': 'organization', 'feats': {'relevance': '50'}},
        ]
    )
    csv_path = tmp_path / 'entities.csv'
    with csv_path.open(mode='w', encoding='utf-8', newline='') as tmp:
        writer = csv.writer(tmp)
        writer.writerow(['id', 'entities'])
        writer.writerow(['article_1', entities_json])

    corpus = FakeCorpus(docs=[SimpleNamespace(id='article_1')])
    with caplog.at_level(logging.INFO):
        attach_entities(
            corpus=corpus,
            csv_path=str(csv_path),
            wdid_mapping={'G1': ['Q1'], 'G2': ['Q2']},
            min_relevance=0.0,
        )

    assert any('Entity type counts:' in record.message for record in caplog.records)
    assert any('location=1' in record.message for record in caplog.records)
    assert any('organization=1' in record.message for record in caplog.records)


def test_attach_entities_remove_types_filters_entities(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Configured remove_types drops linked entities before corpus attachment."""
    monkeypatch.setattr(DocWithEntities, 'from_doc', _fake_from_doc)
    entities_json = json.dumps(
        [
            {'gkbId': 'G1', 'type': 'location', 'feats': {'relevance': '50'}},
            {'gkbId': 'G2', 'type': 'organization', 'feats': {'relevance': '50'}},
            {'gkbId': 'G3', 'type': 'other', 'feats': {'detectedType': 'person', 'relevance': '50'}},
        ]
    )
    csv_path = tmp_path / 'entities.csv'
    with csv_path.open(mode='w', encoding='utf-8', newline='') as tmp:
        writer = csv.writer(tmp)
        writer.writerow(['id', 'entities'])
        writer.writerow(['article_1', entities_json])

    corpus = FakeCorpus(docs=[SimpleNamespace(id='article_1')])
    attach_entities(
        corpus=corpus,
        csv_path=str(csv_path),
        wdid_mapping={'G1': ['Q1'], 'G2': ['Q2'], 'G3': ['Q3']},
        min_relevance=0.0,
        remove_types=('organization', 'person'),
    )

    kept_types = {entity.entity_type for entity in corpus.docs[0].entities}
    assert kept_types == {EntityType.LOCATION}


def test_remove_types_except_keeps_single_type() -> None:
    remove_types = remove_types_except(keep_type='person')

    assert 'person' not in remove_types
    assert set(remove_types) == {entity_type.value for entity_type in EntityType if entity_type != EntityType.PERSON}


# ---------------------------------------------------------------------------
# count_unmapped_entities
# ---------------------------------------------------------------------------

def test_count_unmapped_entities_counts_only_entities_without_wdids() -> None:
    doc = SimpleNamespace(entities=[
        LinkedEntity(gkb_id='g1', wd_ids=('Q1',), relevance=1.0),
        LinkedEntity(gkb_id='g2', wd_ids=(), relevance=1.0),
        LinkedEntity(gkb_id='g3', wd_ids=('Q2', 'Q3'), relevance=1.0),
        LinkedEntity(gkb_id='g4', wd_ids=(), relevance=1.0),
    ])

    assert count_unmapped_entities(doc) == 2


def test_count_unmapped_entities_on_doc_without_entities_returns_zero() -> None:
    doc = SimpleNamespace(entities=[])
    assert count_unmapped_entities(doc) == 0


# ---------------------------------------------------------------------------
# parse_remove_types
# ---------------------------------------------------------------------------

def test_parse_remove_types_parses_known_labels() -> None:
    result = parse_remove_types(remove_types=('person', 'organization'))

    assert result == frozenset({EntityType.PERSON, EntityType.ORGANIZATION})


def test_parse_remove_types_strips_whitespace_and_dedupes() -> None:
    result = parse_remove_types(remove_types=('  person  ', 'person', 'location'))

    assert result == frozenset({EntityType.PERSON, EntityType.LOCATION})


def test_parse_remove_types_skips_unknown_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger='iptc_entity_pipeline.data_loading'):
        result = parse_remove_types(remove_types=('person', 'weird', 'organization'))

    assert result == frozenset({EntityType.PERSON, EntityType.ORGANIZATION})
    assert any('Ignoring unknown remove_types' in rec.message for rec in caplog.records)


def test_parse_remove_types_ignores_empty_and_non_string_entries() -> None:
    result = parse_remove_types(remove_types=('', '   ', 'person', None))  # type: ignore[arg-type]

    assert result == frozenset({EntityType.PERSON})


def test_parse_remove_types_empty_sequence_returns_empty_frozenset() -> None:
    assert parse_remove_types(remove_types=()) == frozenset()


# ---------------------------------------------------------------------------
# get_article_text
# ---------------------------------------------------------------------------

def test_get_article_text_joins_title_lead_body_with_double_newlines() -> None:
    doc = SimpleNamespace(title='Headline', lead='Lead paragraph.', text='Body text.')

    result = get_article_text(doc)

    assert result == 'Headline\n\nLead paragraph.\n\nBody text.'


def test_get_article_text_skips_blank_and_none_parts() -> None:
    doc = SimpleNamespace(title='Headline', lead=None, text='   ')

    result = get_article_text(doc)

    assert result == 'Headline'


def test_get_article_text_returns_empty_string_when_all_parts_blank() -> None:
    doc = SimpleNamespace(title='', lead=None, text=' \t\n ')

    result = get_article_text(doc)

    assert result == ''


def test_get_article_text_strips_each_part_before_joining() -> None:
    doc = SimpleNamespace(title='  Headline  ', lead='  Lead  ', text='  Body  ')

    result = get_article_text(doc)

    assert result == 'Headline\n\nLead\n\nBody'
