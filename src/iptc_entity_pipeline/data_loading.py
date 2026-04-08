"""Load corpora and article-to-entity mappings for IPTC experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from iptc_entity_pipeline.utils import DocWithEntities

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LinkedEntity:
    """Resolved entity linked to a corpus article."""

    gkb_id: str
    wd_ids: tuple[str, ...]
    relevance: float


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
    test_csv: str,
    removed_cat_ids: Iterable[str],
    downsample_corpora: Mapping[str, float] | None = None,
    downsampling_order_cache_json: str | None = None,
) -> Any:
    """
    Load and normalize corpora similarly to the reference pipeline.

    :param train_csv: Training CSV path.
    :param test_csv: Test CSV path.
    :param removed_cat_ids: Category IDs to remove after normalization.
    :param downsample_corpora: Optional mapping ``corpus_name -> keep_ratio`` for train/dev downsampling.
    :param downsampling_order_cache_json: Optional path used to resolve persisted per-corpus order files.
    :return: ``CorpusGroup`` with synchronized train/test category space.
    """
    _ensure_csv_field_limit()
    Corpus, CorpusGroup, iptc = _require_geneea()
    iptc_cats = iptc.IptcTopics.load()
    removed_cat_ids_set = frozenset(removed_cat_ids)
    unknown_cat_ids_seen: set[str] = set()

    def norm_corpus(corpus: Any, remove_empty_cats: bool = True) -> Any:
        def new_docs():
            for doc in corpus:
                if doc.cats:
                    valid_topics = []
                    for cat in doc.cats:
                        try:
                            valid_topics.append(iptc_cats.getCategory(cat))
                        except KeyError:
                            if cat not in unknown_cat_ids_seen:
                                unknown_cat_ids_seen.add(cat)
                                LOGGER.warning(
                                    'Unknown IPTC category id encountered and skipped: cat_id=%s doc_id=%s',
                                    cat,
                                    getattr(doc, 'id', ''),
                                )
                    norm_cats = iptc_cats.normalizeCategories(valid_topics)
                    doc.cats.clear()
                    doc.cats.extend(sorted(cat.id for cat in norm_cats if cat.id and cat.id not in removed_cat_ids_set))
                    if 'catScores' in doc.metadata:
                        doc.metadata['catScores'] = {
                            cat: score
                            for cat, score in doc.metadata['catScores'].items()
                            if cat not in removed_cat_ids_set and cat not in unknown_cat_ids_seen
                        }
                elif remove_empty_cats:
                    continue
                yield doc

        return Corpus(new_docs())

    train = norm_corpus(Corpus.fromCsv(train_csv))
    test = norm_corpus(Corpus.fromCsv(test_csv))

    downsample_mapping = {corpus_name: float(ratio) for corpus_name, ratio in (downsample_corpora or {}).items()}
    cache_dir = _resolve_downsampling_cache_dir(
        cache_path=Path(downsampling_order_cache_json) if downsampling_order_cache_json else None,
    )
    if downsample_mapping:
        if cache_dir is None:
            LOGGER.warning('Downsampling cache path is not configured; order files will not be persisted')
        train = _downsample_split_corpus(
            corpus=train,
            split_name='train',
            downsample_corpora=downsample_mapping,
            cache_dir=cache_dir,
            corpus_factory=Corpus,
        )
        LOGGER.info('Downsampling skipped for test split by design')

    corpora = CorpusGroup(train=train, test=test)
    LOGGER.info('Loaded corpora: train=%s test=%s cat_count=%s', len(corpora.train), len(corpora.test), corpora.catCnt)
    return corpora


def _stable_seed_from_keys(*, split_name: str, corpus_name: str) -> int:
    """Build deterministic random seed from split/corpus keys."""
    seed_key = f'{split_name}|{corpus_name}'
    return int(hashlib.sha256(seed_key.encode(encoding='utf-8')).hexdigest()[:16], 16)


def _resolve_downsampling_cache_dir(*, cache_path: Path | None) -> Path | None:
    """Resolve cache directory where per-corpus ID order files are stored."""
    if cache_path is None:
        return None
    if cache_path.suffix:
        return cache_path.parent
    return cache_path


def _sanitize_filename_part(*, value: str) -> str:
    """Create filesystem-safe fragment for order cache filenames."""
    return ''.join(ch if ch.isalnum() or ch in {'-', '_'} else '_' for ch in value)


def _order_file_path(*, cache_dir: Path, corpus_name: str, split_name: str) -> Path:
    """Build path to persisted order file for split/corpus pair."""
    safe_corpus_name = _sanitize_filename_part(value=corpus_name)
    safe_split_name = _sanitize_filename_part(value=split_name)
    return cache_dir / f'{safe_corpus_name}_{safe_split_name}.txt'


def _load_order_ids(*, order_file: Path) -> list[str]:
    """Load persisted order ids from text file."""
    if not order_file.is_file():
        return []
    with order_file.open(mode='r', encoding='utf-8') as in_file:
        return [line.strip() for line in in_file if line.strip()]


def _save_order_ids(*, order_file: Path, ordered_ids: list[str]) -> None:
    """Persist order ids to text file, one id per line."""
    order_file.parent.mkdir(parents=True, exist_ok=True)
    with order_file.open(mode='w', encoding='utf-8') as out_file:
        out_file.write('\n'.join(ordered_ids))
        out_file.write('\n')


def _build_persisted_order(
    *,
    split_name: str,
    corpus_name: str,
    current_doc_ids: list[str],
    cached_order: list[str],
) -> list[str]:
    """Return deterministic persisted random order compatible with current documents."""
    current_doc_ids_set = set(current_doc_ids)
    merged_order = [doc_id for doc_id in cached_order if doc_id in current_doc_ids_set]
    known_doc_ids = set(merged_order)
    missing_doc_ids = sorted(doc_id for doc_id in current_doc_ids if doc_id not in known_doc_ids)
    if missing_doc_ids:
        rng = random.Random(_stable_seed_from_keys(split_name=split_name, corpus_name=corpus_name))
        rng.shuffle(missing_doc_ids)
        merged_order.extend(missing_doc_ids)
    return merged_order


def _downsample_split_corpus(
    *,
    corpus: Any,
    split_name: str,
    downsample_corpora: Mapping[str, float],
    cache_dir: Path | None,
    corpus_factory: Any,
) -> Any:
    """Downsample selected corpora in split with deterministic persisted per-ID ordering."""
    docs = list(corpus)
    split_doc_ids: dict[str, list[str]] = {}
    for doc in docs:
        corpus_name = str(doc.metadata.get('corpusName', ''))
        split_doc_ids.setdefault(corpus_name, []).append(doc.id)

    wrong_ids: set[str] = set()
    for corpus_name, ratio in downsample_corpora.items():
        doc_ids = split_doc_ids.get(corpus_name, [])
        doc_count_before = len(doc_ids)
        if doc_count_before == 0:
            LOGGER.info(
                'Downsampling split=%s corpus=%s ratio=%.4f before=0 after=0',
                split_name,
                corpus_name,
                ratio,
            )
            continue

        ratio_clamped = min(max(float(ratio), 0.0), 1.0)
        keep_count = int(round(doc_count_before * ratio_clamped))
        cached_order: list[str] = []
        order_file: Path | None = None
        if cache_dir is not None:
            order_file = _order_file_path(cache_dir=cache_dir, corpus_name=corpus_name, split_name=split_name)
            cached_order = _load_order_ids(order_file=order_file)
        persisted_order = _build_persisted_order(
            split_name=split_name,
            corpus_name=corpus_name,
            current_doc_ids=doc_ids,
            cached_order=cached_order,
        )
        if order_file is not None and persisted_order != cached_order:
            _save_order_ids(order_file=order_file, ordered_ids=persisted_order)
        kept_ids = set(persisted_order[:keep_count])
        wrong_ids.update(doc_id for doc_id in doc_ids if doc_id not in kept_ids)
        LOGGER.info(
            'Downsampling split=%s corpus=%s ratio=%.4f before=%s after=%s',
            split_name,
            corpus_name,
            ratio_clamped,
            doc_count_before,
            len(kept_ids),
        )

    def filtered_docs():
        for doc in docs:
            if doc.id not in wrong_ids:
                yield doc

    return corpus_factory(filtered_docs())


def _ensure_csv_field_limit() -> None:
    """Raise CSV parser field-size limit to handle large JSON entity columns."""
    field_limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(field_limit)
            break
        except OverflowError:
            field_limit = field_limit // 10


def load_wdid_mapping(*, wdid_mapping_tsv: str) -> dict[str, list[str]]:
    """
    Parse wdId mapping TSV into gkbId to wdId lookup.

    :param wdid_mapping_tsv: Path to ``wdId_mapping.tsv`` with columns ``gkb_id`` and ``wikidata_ids``.
    :return: Mapping ``gkbId -> list[wdId]``.
    """
    mapping: dict[str, list[str]] = {}
    path = Path(wdid_mapping_tsv)
    with path.open(mode='r', encoding='utf-8', newline='') as in_file:
        reader = csv.DictReader(in_file, delimiter='\t')
        for row in reader:
            gkb_id = (row.get('gkb_id') or '').strip()
            wikidata_ids = (row.get('wikidata_ids') or '').strip()
            if not gkb_id:
                continue
            wdids = [wid.strip() for wid in wikidata_ids.split('|') if wid.strip()] if wikidata_ids else []
            mapping[gkb_id] = wdids
    LOGGER.info('Loaded wdId mapping for %s gkb entities', len(mapping))
    return mapping


def attach_entities_to_corpus(
    *,
    corpus: Any,
    csv_path: str,
    wdid_mapping: Mapping[str, Sequence[str]],
) -> None:
    """
    Parse entities from CSV and attach resolved :class:`LinkedEntity` objects to each doc.

    Sets ``doc.entities`` to a list of :class:`LinkedEntity` for every document in the
    corpus. Documents not present in the CSV get an empty list.

    :param corpus: Corpus whose documents will be enriched in-place.
    :param csv_path: Path to a corpus CSV containing an ``entities`` JSON column.
    :param wdid_mapping: Pre-loaded gkbId-to-wdId mapping from :func:`load_wdid_mapping`.
    """
    _ensure_csv_field_limit()
    article_entities: dict[str, list[LinkedEntity]] = {}
    path = Path(csv_path)
    with path.open(mode='r', encoding='utf-8', newline='') as in_file:
        reader = csv.DictReader(in_file)
        for row in reader:
            article_id = (row.get('id') or '').strip()
            entities_json = (row.get('entities') or '').strip()
            if not article_id or not entities_json:
                continue
            try:
                raw_entities = json.loads(entities_json)
            except json.JSONDecodeError:
                LOGGER.warning('Skipping malformed entities JSON for article_id=%s', article_id)
                continue
            linked: list[LinkedEntity] = []
            if isinstance(raw_entities, list):
                for ent in raw_entities:
                    if not isinstance(ent, Mapping):
                        continue
                    gkb_id = ent.get('gkbId', '')
                    if not gkb_id:
                        continue
                    wd_ids = tuple(wdid_mapping.get(gkb_id, ()))
                    relevance = float(ent.get('relevance', 0.0))
                    linked.append(LinkedEntity(gkb_id=gkb_id, wd_ids=wd_ids, relevance=relevance))
            if linked:
                article_entities[article_id] = linked

    attached = 0
    for i, doc in enumerate(corpus.docs):
        enriched = DocWithEntities.from_doc(doc=doc, entities=article_entities.get(doc.id, []))
        corpus.docs[i] = enriched
        if enriched.entities:
            attached += 1
    LOGGER.info(
        'Attached entities to corpus: %s/%s articles have entities (from %s)',
        attached,
        len(corpus),
        csv_path,
    )


def get_doc_wdids(doc: Any) -> list[str]:
    """
    Extract flat wdId list from ``doc.entities``.

    :param doc: Corpus document with attached :class:`LinkedEntity` objects.
    :return: List of all wdIds across the document's linked entities.
    """
    return [wd_id for ent in getattr(doc, 'entities', []) for wd_id in ent.wd_ids]


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

