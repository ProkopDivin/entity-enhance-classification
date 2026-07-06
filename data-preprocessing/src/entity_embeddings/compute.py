'''Compute per-entity embeddings compatible with ``EntityEmbeddingStore``.'''

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Protocol, Sequence

import numpy as np
from geneea.catlib.vec.vectorizer import SvcTextVectorizer
from geneea.core import logutil
from geneea.kb.tools.rdfutil import UrlSparqlService

from entity_embeddings.constants import (
    DEFAULT_EMBED_BATCH_SIZE,
    DEFAULT_LANGS,
    DEFAULT_SPARQL_BATCH_SIZE,
)
from entity_embeddings.jina_embed import JinaTextVectorizer, TaskKind

LOG = logutil.getLogger(__package__, __file__)

InputSource = Literal['wikidata', 'files']
TextSource = Literal['wikidata_description', 'text_file']


@dataclass(frozen=True)
class TextEmbeddingItem:
    '''One local text file ready for embedding.'''

    qid: str
    lang: str
    chunk_id: int
    text: str
    source_path: Path


class TextMatrixVectorizer(Protocol):
    '''Minimal vectorizer interface used for batch text embedding.'''

    def toMatrix(self, texts: Sequence[str]) -> np.ndarray:
        ...


def load_qids(*, ids_path: Path) -> list[str]:
    '''
    Load QIDs from a text file.

    :param ids_path: Input path with one QID per line.
    :return: Ordered QIDs without empty lines.
    '''
    qids: list[str] = []
    with open(ids_path, encoding='utf-8') as in_file:
        for line in in_file:
            qid = line.strip()
            if qid:
                qids.append(qid)
    return qids


def _lang_values(*, langs: Sequence[str]) -> str:
    return ' '.join(f'"{lang}"' for lang in langs)


def _item_values(*, qids: Sequence[str]) -> str:
    return ' '.join(f'wd:{qid}' for qid in qids)


def build_description_query(*, qids: Sequence[str], langs: Sequence[str]) -> str:
    '''
    Build SPARQL query for multilingual descriptions of multiple Wikidata entities.

    :param qids: Wikidata QIDs.
    :param langs: Languages to fetch.
    :return: SPARQL SELECT query.
    '''
    return f'''
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX schema: <http://schema.org/>

SELECT ?item ?lang ?description
WHERE {{
  VALUES ?item {{ {_item_values(qids=qids)} }}
  VALUES ?wantedLang {{ {_lang_values(langs=langs)} }}

  ?item schema:description ?description .
  BIND(LANG(?description) AS ?lang)
  FILTER(?lang IN (?wantedLang))
}}
ORDER BY ?item ?lang
'''


def _qid_from_item_uri(*, item_uri: str) -> str | None:
    marker = 'http://www.wikidata.org/entity/'
    if not item_uri.startswith(marker):
        return None
    qid = item_uri[len(marker):]
    return qid if qid else None


def fetch_descriptions_batch(
    *,
    sparql: UrlSparqlService,
    qids: Sequence[str],
    langs: Sequence[str],
) -> dict[str, dict[str, str]]:
    '''
    Fetch descriptions for entity batch and selected languages.

    :param sparql: SPARQL service client.
    :param qids: Wikidata QIDs.
    :param langs: Requested language codes.
    :return: Mapping ``qid -> (lang -> description)``.
    '''
    query = build_description_query(qids=qids, langs=langs)
    payload = sparql.query(query)
    by_qid = {qid: {} for qid in qids}
    for row in payload.get('results', {}).get('bindings', []):
        item_uri = row.get('item', {}).get('value')
        lang = row.get('lang', {}).get('value')
        description = row.get('description', {}).get('value')
        if not item_uri or not lang or description is None:
            continue
        qid = _qid_from_item_uri(item_uri=str(item_uri))
        if not qid or qid not in by_qid:
            continue
        by_qid[qid][lang] = str(description)
    return by_qid


def iter_qid_batches(*, qids: Sequence[str], batch_size: int) -> Sequence[Sequence[str]]:
    '''
    Split QID list into fixed-size batches.

    :param qids: Ordered Wikidata QIDs.
    :param batch_size: Maximum batch size.
    :return: Sequence of QID batches.
    '''
    return [qids[idx: idx + batch_size] for idx in range(0, len(qids), batch_size)]


def parse_entity_text_stem(*, stem: str) -> tuple[str, str, int] | None:
    '''
    Parse ``{QID}_{lang}_{chunk}`` file stem.

    :param stem: Filename without extension.
    :return: Tuple ``(qid, lang, chunk_id)`` or ``None`` when the stem is invalid.
    '''
    parts = stem.rsplit('_', maxsplit=2)
    if len(parts) != 3:
        return None
    qid, lang, chunk_raw = parts
    if not qid.startswith('Q') or not lang or not chunk_raw.isdigit():
        return None
    return qid, lang, int(chunk_raw)


def iter_text_embedding_items(
    *,
    text_dir: Path,
    langs: Sequence[str],
    qids: Sequence[str] | None = None,
) -> Iterator[TextEmbeddingItem]:
    '''
    Yield text files from a directory filtered by language and optional QID list.

    Expected filename pattern: ``{QID}_{lang}_{chunk}.txt``.

    :param text_dir: Directory with entity text files.
    :param langs: Language codes to include.
    :param qids: Optional QID whitelist; when omitted, all matching files are used.
    :return: Iterator of parsed text items.
    '''
    selected_langs = set(langs)
    selected_qids = set(qids) if qids is not None else None

    for path in sorted(text_dir.glob('*.txt')):
        parsed = parse_entity_text_stem(stem=path.stem)
        if parsed is None:
            LOG.warning(f'Skipping file with unexpected name, path={path}')
            continue
        qid, lang, chunk_id = parsed
        if lang not in selected_langs:
            continue
        if selected_qids is not None and qid not in selected_qids:
            continue
        text = path.read_text(encoding='utf-8').strip()
        if not text:
            LOG.warning(f'Skipping empty text file, path={path}')
            continue
        yield TextEmbeddingItem(
            qid=qid,
            lang=lang,
            chunk_id=chunk_id,
            text=text,
            source_path=path,
        )


def _build_metadata(
    *,
    qid: str,
    lang: str,
    chunk_id: int,
    model_name: str,
    source: TextSource,
) -> dict:
    return {
        'id': f'{qid}_{lang}_{chunk_id}',
        'metadata': {
            'Language': lang,
            'QID': qid,
            'Source': source,
            'Model': model_name,
            'DumpDate': 'n/a',
            'WikipediaTitle': '',
            'ChunkID': chunk_id,
        },
    }


def save_entity_embedding(
    *,
    out_dir: Path,
    qid: str,
    lang: str,
    chunk_id: int,
    vector: np.ndarray,
    model_name: str,
    source: TextSource = 'wikidata_description',
) -> None:
    '''
    Persist one entity embedding and metadata sidecar.

    :param out_dir: Output directory.
    :param qid: Wikidata QID.
    :param lang: Language code.
    :param chunk_id: Chunk index from the source filename.
    :param vector: Embedding vector.
    :param model_name: Embedding model identifier.
    :param source: Text provenance label stored in metadata.
    '''
    stem = f'{qid}_{lang}_{chunk_id}'
    npy_path = out_dir / f'{stem}.npy'
    json_path = out_dir / f'{stem}.json'
    np.save(npy_path, np.asarray(vector, dtype=np.float32))
    metadata = _build_metadata(
        qid=qid,
        lang=lang,
        chunk_id=chunk_id,
        model_name=model_name,
        source=source,
    )
    with open(json_path, 'w', encoding='utf-8') as out_file:
        json.dump(metadata, out_file, ensure_ascii=False, indent=2)


def compute_description_embeddings(
    *,
    qids: Sequence[str],
    langs: Sequence[str],
    out_dir: Path,
    sparql: UrlSparqlService,
    vectorizer: TextMatrixVectorizer,
    model_name: str,
    batch_size: int = DEFAULT_SPARQL_BATCH_SIZE,
) -> tuple[int, int]:
    '''
    Compute and save per-language description embeddings.

    :param qids: Wikidata IDs.
    :param langs: Languages to fetch and embed.
    :param out_dir: Output directory.
    :param sparql: SPARQL service.
    :param vectorizer: Description text vectorizer.
    :param model_name: Embedding model identifier.
    :return: Tuple ``(saved_count, missing_count)``.
    '''
    saved_count = 0
    missing_count = 0
    for qid_batch in iter_qid_batches(qids=qids, batch_size=batch_size):
        descriptions_by_qid = fetch_descriptions_batch(sparql=sparql, qids=qid_batch, langs=langs)
        for qid in qid_batch:
            descriptions = descriptions_by_qid.get(qid, {})
            batch_langs = [lang for lang in langs if lang in descriptions]
            if not batch_langs:
                missing_count += len(langs)
                continue

            texts = [descriptions[lang] for lang in batch_langs]
            matrix = vectorizer.toMatrix(texts)
            for lang, vector in zip(batch_langs, matrix):
                save_entity_embedding(
                    out_dir=out_dir,
                    qid=qid,
                    lang=lang,
                    chunk_id=1,
                    vector=np.asarray(vector, dtype=np.float32),
                    model_name=model_name,
                )
                saved_count += 1
            missing_count += len(langs) - len(batch_langs)
    return saved_count, missing_count


def iter_item_batches(
    *,
    items: Sequence[TextEmbeddingItem],
    batch_size: int,
) -> Iterator[Sequence[TextEmbeddingItem]]:
    '''
    Split text items into fixed-size batches.

    :param items: Ordered text items.
    :param batch_size: Maximum batch size.
    :return: Iterator of item batches.
    '''
    for idx in range(0, len(items), batch_size):
        yield items[idx: idx + batch_size]


def compute_file_embeddings(
    *,
    text_dir: Path,
    langs: Sequence[str],
    out_dir: Path,
    vectorizer: TextMatrixVectorizer,
    model_name: str,
    qids: Sequence[str] | None = None,
    batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    skip_existing: bool = False,
) -> tuple[int, int]:
    '''
    Compute and save embeddings from local text files.

    :param text_dir: Directory with ``{QID}_{lang}_{chunk}.txt`` files.
    :param langs: Language codes to include.
    :param out_dir: Output directory.
    :param vectorizer: Text vectorizer.
    :param model_name: Embedding model identifier.
    :param qids: Optional QID whitelist.
    :param batch_size: Number of texts embedded per vectorizer call.
    :param skip_existing: Skip items whose ``.npy`` output already exists.
    :return: Tuple ``(saved_count, skipped_count)``.
    '''
    items = list(iter_text_embedding_items(text_dir=text_dir, langs=langs, qids=qids))
    if not items:
        LOG.warning(f'No text files matched text_dir={text_dir}, langs={langs}')
        return 0, 0

    to_embed: list[TextEmbeddingItem] = []
    skipped_count = 0
    for item in items:
        out_path = out_dir / f'{item.qid}_{item.lang}_{item.chunk_id}.npy'
        if skip_existing and out_path.is_file():
            skipped_count += 1
            continue
        to_embed.append(item)

    saved_count = 0
    for batch in iter_item_batches(items=to_embed, batch_size=batch_size):
        texts = [item.text for item in batch]
        matrix = vectorizer.toMatrix(texts)
        for item, vector in zip(batch, matrix):
            save_entity_embedding(
                out_dir=out_dir,
                qid=item.qid,
                lang=item.lang,
                chunk_id=item.chunk_id,
                vector=np.asarray(vector, dtype=np.float32),
                model_name=model_name,
                source='text_file',
            )
            saved_count += 1
        LOG.info(f'Embedded file batch, saved_so_far={saved_count}, skipped={skipped_count}')
    return saved_count, skipped_count


def parse_langs(*, raw_langs: Sequence[str] | None) -> tuple[str, ...]:
    '''
    Parse language codes from repeated CLI values or comma-separated strings.

    :param raw_langs: Raw language arguments from CLI.
    :return: Ordered unique language codes.
    '''
    if not raw_langs:
        return DEFAULT_LANGS

    langs: list[str] = []
    seen: set[str] = set()
    for raw in raw_langs:
        for lang in raw.split(','):
            lang = lang.strip()
            if not lang or lang in seen:
                continue
            seen.add(lang)
            langs.append(lang)
    if not langs:
        raise ValueError('At least one language code must be provided')
    return tuple(langs)


def build_vectorizer(
    *,
    embed_backend: str,
    model_path: str | None,
    embed_svc_url: str,
    svc_embed_dim: int,
    jina_variant: str,
    jina_task: TaskKind,
    jina_embed_dim: int | None,
) -> tuple[TextMatrixVectorizer, str]:
    '''
    Build the configured text vectorizer and metadata model name.

    :param embed_backend: ``svc`` for remote embedding service, ``jina`` for local Jina models
    :param model_path: Remote service model id (required for ``svc``)
    :param embed_svc_url: Embedding service URL for ``svc`` backend
    :param svc_embed_dim: Output dimension for ``svc`` backend
    :param jina_variant: Jina model variant key
    :param jina_task: Jina task kind (passage for descriptions, query for search text)
    :param jina_embed_dim: Output dimension for Jina backend; defaults to variant config
    :return: Tuple ``(vectorizer, model_name)``
    '''
    if embed_backend == 'jina':
        vectorizer = JinaTextVectorizer(
            variant=jina_variant,
            task_kind=jina_task,
            embedding_dim=jina_embed_dim,
        )
        return vectorizer, vectorizer.model_name

    if not model_path:
        raise ValueError('--model-path is required when --embed-backend=svc')

    vectorizer = SvcTextVectorizer(
        embedSvcUrl=embed_svc_url,
        modelId=model_path,
        embedDim=svc_embed_dim,
    )
    return vectorizer, model_path
