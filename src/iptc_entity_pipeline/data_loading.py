"""Load corpora and article-to-entity mappings for IPTC experiments."""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


def _require_geneea():
    try:
        from geneea.catlib.data import Corpus, CorpusGroup  # type: ignore
        from geneea.mediacats import iptc  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise ImportError(
            'Missing geneea dependencies. Install internal geneea.* packages to run the pipeline.'
        ) from exc
    return Corpus, CorpusGroup, iptc


def load_and_normalize_corpora(
    *,
    train_csv: str,
    dev_csv: str,
    test_csv: str,
    removed_cat_ids: Iterable[str],
) -> Any:
    """
    Load and normalize corpora similarly to the reference pipeline.

    :param train_csv: Training CSV path.
    :param dev_csv: Development CSV path.
    :param test_csv: Test CSV path.
    :param removed_cat_ids: Category IDs to remove after normalization.
    :return: ``CorpusGroup`` instance with normalized categories.
    """
    Corpus, CorpusGroup, iptc = _require_geneea()
    iptc_cats = iptc.IptcTopics.load()
    removed_cat_ids_set = frozenset(removed_cat_ids)

    def norm_corpus(corpus: Any, remove_empty_cats: bool = True) -> Any:
        def new_docs():
            for doc in corpus:
                if doc.cats:
                    norm_cats = iptc_cats.normalizeCategories([iptc_cats.getCategory(cat) for cat in doc.cats])
                    doc.cats.clear()
                    doc.cats.extend(sorted(cat.id for cat in norm_cats if cat.id and cat.id not in removed_cat_ids_set))
                    if 'catScores' in doc.metadata:
                        doc.metadata['catScores'] = {
                            cat: score
                            for cat, score in doc.metadata['catScores'].items()
                            if cat not in removed_cat_ids_set
                        }
                elif remove_empty_cats:
                    continue
                yield doc

        return Corpus(new_docs())

    train = norm_corpus(Corpus.fromCsv(train_csv))
    dev = norm_corpus(Corpus.fromCsv(dev_csv))
    test = norm_corpus(Corpus.fromCsv(test_csv))
    LOGGER.info('Loaded corpora: train=%s dev=%s test=%s', len(train), len(dev), len(test))
    return CorpusGroup(train=train, dev=dev, test=test)


def load_article_wdids(*, article_entities_tsv: str) -> dict[str, list[str]]:
    """
    Parse article to entity mapping and keep only entities with ``wdId``.

    :param article_entities_tsv: Path to ``article_2_entities.tsv``.
    :return: Mapping ``article_id -> list[wdId]``.
    """
    mapping: dict[str, list[str]] = {}
    path = Path(article_entities_tsv)
    with path.open(mode='r', encoding='utf-8', newline='') as in_file:
        reader = csv.DictReader(in_file, delimiter='\t')
        for row in reader:
            article_id = row.get('article_id')
            entities_json = row.get('entities')
            if not article_id or not entities_json:
                continue
            try:
                entities = json.loads(entities_json)
            except json.JSONDecodeError:
                LOGGER.warning('Skipping malformed entities JSON for article_id=%s', article_id)
                continue
            wdids = [entity['wdId'] for entity in entities if isinstance(entity, Mapping) and entity.get('wdId')]
            mapping[article_id] = wdids
    LOGGER.info('Loaded wdId mapping for %s articles', len(mapping))
    return mapping


def get_article_text(doc: Any) -> str:
    """
    Compose deterministic text used for fallback article embeddings.

    :param doc: Corpus document.
    :return: Concatenated text from title, lead, and body.
    """
    title = getattr(doc, 'title', '') or ''
    lead = getattr(doc, 'lead', '') or ''
    text = getattr(doc, 'text', '') or ''
    return '\n\n'.join(part.strip() for part in [title, lead, text] if part and part.strip())

