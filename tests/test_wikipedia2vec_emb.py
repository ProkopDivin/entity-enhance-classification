"""Tests for the Wikipedia summary/plain-text fetcher used to feed wikipedia2vec embeddings."""
from __future__ import annotations

import json
from pathlib import Path

from entity_embeddings import wikipedia2vec as wikipedia2vec_emb


def test_extract_plain_text_missing_page_returns_none() -> None:
    payload = {
        'query': {
            'pages': {
                '-1': {
                    'ns': 0,
                    'title': 'Missing',
                    'missing': '',
                },
            },
        },
    }
    assert wikipedia2vec_emb._extract_plain_text(payload=payload) is None


def test_build_summary_uses_first_non_empty_paragraph() -> None:
    text = '\n\nFirst paragraph.\n\nSecond paragraph.'
    assert wikipedia2vec_emb._build_summary(text=text) == 'First paragraph.'


def test_fetch_page_texts_writes_summary_and_plain_text(tmp_path: Path, monkeypatch) -> None:
    def _fake_fetch_page_plain_text(*, lang: str, title: str) -> str | None:
        assert lang == 'en'
        assert title == 'Douglas Adams'
        return 'Douglas Adams was an English author.\n\nHe wrote The Hitchhiker series.'

    monkeypatch.setattr(wikipedia2vec_emb, '_fetch_page_plain_text', _fake_fetch_page_plain_text)

    counts = wikipedia2vec_emb.fetch_page_texts(
        qids=['Q42'],
        langs=('en',),
        qid_to_titles={'Q42': {'en': 'Douglas Adams'}},
        out_dir=tmp_path,
        overwrite=True,
        sleep_s=0.0,
    )

    assert counts['saved'] == 1
    out_path = tmp_path / 'Q42_en_wiki_text.json'
    assert out_path.exists()
    with open(out_path, encoding='utf-8') as in_file:
        payload = json.load(in_file)
    assert payload['status'] == wikipedia2vec_emb.CACHE_STATUS_OK
    assert payload['summary'] == 'Douglas Adams was an English author.'
    assert payload['plain_text'].startswith('Douglas Adams')
