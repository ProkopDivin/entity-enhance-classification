"""Tests for the entity_embeddings compute pipeline (description + text vectorization)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from entity_embeddings.constants import DEFAULT_LANGS
from entity_embeddings.compute import (
    build_description_query,
    build_vectorizer,
    compute_description_embeddings,
    compute_file_embeddings,
    fetch_descriptions_batch,
    iter_text_embedding_items,
    parse_entity_text_stem,
)
from entity_embeddings.jina_embed import JinaTextVectorizer


class _FakeSparqlService:
    def __init__(self, payload_by_qid: dict[str, dict]) -> None:
        self._payload_by_qid = payload_by_qid
        self.query_count = 0

    def query(self, query: str):
        self.query_count += 1
        marker = 'VALUES ?item { '
        start = query.index(marker) + len(marker)
        end = query.index(' }', start)
        items = query[start:end].split()
        qids = [item.replace('wd:', '') for item in items]
        bindings = []
        for qid in qids:
            payload = self._payload_by_qid.get(qid, {'results': {'bindings': []}})
            bindings.extend(payload['results']['bindings'])
        return {'results': {'bindings': bindings}}


class _FakeVectorizer:
    def __init__(self, dim: int = 4) -> None:
        self._dim = dim

    def toMatrix(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self._dim), dtype=np.float64)
        for idx, text in enumerate(texts):
            base = float(len(text))
            matrix[idx] = np.array([base + i for i in range(self._dim)], dtype=np.float64)
        return matrix


def test_build_description_query_contains_qid_and_langs() -> None:
    query = build_description_query(qids=('Q42', 'Q100'), langs=('en', 'cs'))
    assert 'wd:Q42' in query
    assert 'wd:Q100' in query
    assert '"en"' in query
    assert '"cs"' in query
    assert 'schema:description' in query


def test_fetch_descriptions_batch_returns_nested_mapping() -> None:
    payload = {
        'results': {
            'bindings': [
                {
                    'item': {'value': 'http://www.wikidata.org/entity/Q42'},
                    'lang': {'value': 'en'},
                    'description': {'value': 'English text'},
                },
                {
                    'item': {'value': 'http://www.wikidata.org/entity/Q42'},
                    'lang': {'value': 'fr'},
                    'description': {'value': 'French text'},
                },
            ],
        },
    }
    sparql = _FakeSparqlService({'Q42': payload})

    result = fetch_descriptions_batch(sparql=sparql, qids=('Q42',), langs=('en', 'fr'))
    assert result == {'Q42': {'en': 'English text', 'fr': 'French text'}}


def test_compute_description_embeddings_saves_expected_files(tmp_path: Path) -> None:
    payload = {
        'results': {
            'bindings': [
                {
                    'item': {'value': 'http://www.wikidata.org/entity/Q100'},
                    'lang': {'value': 'en'},
                    'description': {'value': 'English description'},
                },
                {
                    'item': {'value': 'http://www.wikidata.org/entity/Q100'},
                    'lang': {'value': 'de'},
                    'description': {'value': 'German description'},
                },
            ],
        },
    }
    sparql = _FakeSparqlService({'Q100': payload})
    vectorizer = _FakeVectorizer(dim=3)

    saved, missing = compute_description_embeddings(
        qids=['Q100'],
        langs=DEFAULT_LANGS,
        out_dir=tmp_path,
        sparql=sparql,
        vectorizer=vectorizer,
        model_name='model-x',
    )

    assert saved == 2
    assert missing == len(DEFAULT_LANGS) - 2

    en_npy = tmp_path / 'Q100_en_1.npy'
    de_npy = tmp_path / 'Q100_de_1.npy'
    cs_npy = tmp_path / 'Q100_cs_1.npy'
    assert en_npy.exists()
    assert de_npy.exists()
    assert not cs_npy.exists()

    en_vec = np.load(en_npy)
    assert en_vec.dtype == np.float32
    assert en_vec.shape == (3,)

    with open(tmp_path / 'Q100_en_1.json', encoding='utf-8') as in_file:
        metadata = json.load(in_file)
    assert metadata['id'] == 'Q100_en_1'
    assert metadata['metadata']['Language'] == 'en'
    assert metadata['metadata']['QID'] == 'Q100'
    assert metadata['metadata']['Model'] == 'model-x'


def test_compute_description_embeddings_uses_batches_of_100(tmp_path: Path) -> None:
    payload = {
        'results': {
            'bindings': [
                {
                    'item': {'value': 'http://www.wikidata.org/entity/Q1'},
                    'lang': {'value': 'en'},
                    'description': {'value': 'one'},
                },
            ],
        },
    }
    qids = [f'Q{idx}' for idx in range(1, 202)]
    sparql = _FakeSparqlService({'Q1': payload})
    vectorizer = _FakeVectorizer(dim=2)

    compute_description_embeddings(
        qids=qids,
        langs=DEFAULT_LANGS,
        out_dir=tmp_path,
        sparql=sparql,
        vectorizer=vectorizer,
        model_name='model-x',
    )

    assert sparql.query_count == 3


def test_build_vectorizer_jina_returns_model_name() -> None:
    vectorizer, model_name = build_vectorizer(
        embed_backend='jina',
        model_path=None,
        embed_svc_url='http://example',
        svc_embed_dim=384,
        jina_variant='jina-v3',
        jina_task='passage',
        jina_embed_dim=1024,
    )
    assert isinstance(vectorizer, JinaTextVectorizer)
    assert model_name == 'jina-v3:passage'


def test_build_vectorizer_svc_requires_model_path() -> None:
    with pytest.raises(ValueError, match='--model-path'):
        build_vectorizer(
            embed_backend='svc',
            model_path=None,
            embed_svc_url='http://example',
            svc_embed_dim=384,
            jina_variant='jina-v3',
            jina_task='passage',
            jina_embed_dim=None,
        )


def test_jina_text_vectorizer_to_matrix(monkeypatch) -> None:
    def _fake_embed_texts(texts, *, variant, task_kind, embedding_dim):
        assert variant == 'jina-v3'
        assert task_kind == 'query'
        assert embedding_dim == 512
        return [[float(idx), float(len(text))] for idx, text in enumerate(texts)]

    monkeypatch.setattr('entity_embeddings.jina_embed.embed_texts', _fake_embed_texts)
    vectorizer = JinaTextVectorizer(variant='jina-v3', task_kind='query', embedding_dim=512)
    matrix = vectorizer.toMatrix(['ab', 'abcd'])
    assert matrix.shape == (2, 2)
    assert matrix.dtype == np.float64
    assert vectorizer.model_name == 'jina-v3:query'


def test_parse_entity_text_stem_parses_expected_pattern() -> None:
    assert parse_entity_text_stem(stem='Q1000033_en_1') == ('Q1000033', 'en', 1)
    assert parse_entity_text_stem(stem='invalid_name') is None


def test_iter_text_embedding_items_filters_by_language(tmp_path: Path) -> None:
    (tmp_path / 'Q1_en_1.txt').write_text('English intro', encoding='utf-8')
    (tmp_path / 'Q1_cs_1.txt').write_text('Cesky uvod', encoding='utf-8')
    (tmp_path / 'Q2_en_1.txt').write_text('Another intro', encoding='utf-8')

    items = list(iter_text_embedding_items(text_dir=tmp_path, langs=('en',)))
    assert len(items) == 2
    assert {(item.qid, item.lang) for item in items} == {('Q1', 'en'), ('Q2', 'en')}


def test_compute_file_embeddings_saves_expected_files(tmp_path: Path) -> None:
    text_dir = tmp_path / 'texts'
    text_dir.mkdir()
    (text_dir / 'Q42_en_1.txt').write_text('English intro text', encoding='utf-8')
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    vectorizer = _FakeVectorizer(dim=3)

    saved, skipped = compute_file_embeddings(
        text_dir=text_dir,
        langs=('en',),
        out_dir=out_dir,
        vectorizer=vectorizer,
        model_name='jina-v3:passage',
        batch_size=8,
    )

    assert saved == 1
    assert skipped == 0
    assert (out_dir / 'Q42_en_1.npy').exists()

    with open(out_dir / 'Q42_en_1.json', encoding='utf-8') as in_file:
        metadata = json.load(in_file)
    assert metadata['metadata']['Source'] == 'text_file'
    assert metadata['metadata']['ChunkID'] == 1
