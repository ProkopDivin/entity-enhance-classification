'''
Compute per-entity embeddings for the IPTC entity-enhanced pipeline.

This script builds entity embedding files compatible with
``EntityEmbeddingStore`` (``{QID}_{lang}_{chunk}.npy`` plus JSON sidecar).

Input sources
-------------

**Wikidata descriptions** (``--input-source wikidata``, default):

  Fetches multilingual ``schema:description`` texts from a SPARQL endpoint for
  QIDs listed in ``--ids``. Use this for short Wikidata entity descriptions.

**Local text files** (``--input-source files``):

  Reads pre-cut entity texts from a directory such as ``data/cuted-articles``.
  Expected filename pattern: ``{QID}_{lang}_{chunk}.txt`` (e.g. Wikipedia intro
  snippets). Filter by language with ``--langs``; optionally restrict QIDs with
  ``--ids``.

Embedding backends and models
-----------------------------

**Jina** (``--embed-backend jina``, default) — local Hugging Face models:

  - ``jina-v3`` — retrieval embeddings via ``retrieval.passage`` / ``retrieval.query``
    (use ``--task passage`` for entity/document text, ``--task query`` for search text)
  - ``jina-v3-classification`` — classification task on v3 (``--task classification``)
  - ``jina-v5`` — retrieval with query/document prompts (same roles as v3)
  - ``jina-v5-classification`` — classification embeddings on v5

  Select variant with ``--variant`` and task with ``--task``. Default output
  dimension is 1024 (override with ``--embedding-dim``).

**Remote service** (``--embed-backend svc``):

  Uses ``SvcTextVectorizer`` against a Geneea embedding service
  (``--embed-svc-url``, ``--model-path``, ``--svc-embedding-dim``). Same
  interface as article embeddings in the main pipeline.

Examples
--------

Wikidata descriptions with Jina-v3::

  python -m iptc_entity_pipeline.custom_embeddings \\
    --input-source wikidata \\
    --ids data/gold-chrono-per-dataset/wdId_ids.txt \\
    --out-dir data/entity_embeddings/WikidataDescription_jina \\
    --variant jina-v3 --task passage --langs en,cs,de

Wikipedia intro texts from cuted-articles (English only)::

  python -m iptc_entity_pipeline.custom_embeddings \\
    --input-source files \\
    --text-dir data/cuted-articles \\
    --out-dir data/entity_embeddings/cuted-article-embeddings-jina-v3 \\
    --variant jina-v3 --task passage --langs en --skip-existing
'''

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Protocol, Sequence

import numpy as np
from geneea.catlib.vec.vectorizer import SvcTextVectorizer
from geneea.core import logutil
from geneea.kb.tools.rdfutil import UrlSparqlService

from iptc_entity_pipeline.jina_embed import JinaModelVariant, JinaTextVectorizer, TaskKind
from iptc_entity_pipeline.wikipedia2vec_emb import DEFAULT_IDS_PATH

LOG = logutil.getLogger(__package__, __file__)

DEFAULT_OUT_DIR = 'data/entity_embeddings/WikidataProject'
DEFAULT_LANGS = ('en', 'cs', 'nl', 'fr', 'de', 'es')
DEFAULT_SPARQL_URL = 'http://psi.g:9999/bigdata/sparql'
DEFAULT_EMBED_SVC_URL = 'http://tau.g:5533'
DEFAULT_SVC_EMBED_DIM = 384
DEFAULT_JINA_EMBED_DIM = 1024
DEFAULT_SPARQL_BATCH_SIZE = 100
DEFAULT_EMBED_BACKEND = 'jina'
DEFAULT_JINA_VARIANT = JinaModelVariant.JINA_V3.value
DEFAULT_JINA_TASK: TaskKind = 'passage'
DEFAULT_TEXT_DIR = 'data/cuted-articles'
DEFAULT_INPUT_SOURCE = 'wikidata'
DEFAULT_EMBED_BATCH_SIZE = 32

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


def build_arg_parser() -> argparse.ArgumentParser:
    '''
    Build CLI argument parser.

    :return: Configured parser.
    '''
    jina_variants = [variant.value for variant in JinaModelVariant]
    epilog = '''
Input sources:
  wikidata  Fetch multilingual descriptions from Wikidata via SPARQL (--ids required)
  files     Read local {QID}_{lang}_{chunk}.txt files (e.g. Wikipedia intros in data/cuted-articles)

Embedding backends:
  jina      Local Jina models (--variant, --task, --embedding-dim)
  svc       Remote embedding service (--model-path, --embed-svc-url, --svc-embedding-dim)

Jina variants (--variant, with --embed-backend jina):
  jina-v3                   retrieval.passage / retrieval.query (default)
  jina-v3-classification    classification embeddings on v3
  jina-v5                   retrieval query/document prompts (same roles as v3)
  jina-v5-classification    classification embeddings on v5

Jina tasks (--task):
  passage         Entity/document text (Wikidata descriptions, Wikipedia intros)
  query           Search-query embeddings (asymmetric retrieval)
  classification  Classification-style embeddings (v3/v5 classification variants)
'''
    argparser = argparse.ArgumentParser(
        description=(
            'Compute entity embeddings from Wikidata descriptions or local text files '
            '(e.g. Wikipedia article snippets).'
        ),
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    argparser.add_argument('--out-dir', default=DEFAULT_OUT_DIR, help='Output directory for per-entity embedding files.')
    argparser.add_argument(
        '--input-source',
        choices=('wikidata', 'files'),
        default=DEFAULT_INPUT_SOURCE,
        help='wikidata: SPARQL descriptions; files: local {QID}_{lang}_{chunk}.txt texts.',
    )
    argparser.add_argument(
        '--text-dir',
        default=DEFAULT_TEXT_DIR,
        help='Text directory for --input-source=files (default: data/cuted-articles).',
    )
    argparser.add_argument(
        '--embed-batch-size',
        type=int,
        default=DEFAULT_EMBED_BATCH_SIZE,
        help='Batch size for file-based embedding (default: 32).',
    )
    argparser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip file inputs whose output .npy already exists (files mode only).',
    )
    argparser.add_argument(
        '--embed-backend',
        choices=('svc', 'jina'),
        default=DEFAULT_EMBED_BACKEND,
        help='jina: local Jina model; svc: remote Geneea embedding service.',
    )
    argparser.add_argument(
        '--model-path',
        help='Embedding service model id (required when --embed-backend=svc).',
    )
    argparser.add_argument(
        '--ids',
        default=None,
        help='QID list file (one QID per line). Required for wikidata mode; optional filter for files mode.',
    )
    argparser.add_argument('--sparql-url', default=DEFAULT_SPARQL_URL, help='SPARQL endpoint URL.')
    argparser.add_argument('--embed-svc-url', default=DEFAULT_EMBED_SVC_URL, help='Embedding service URL.')
    argparser.add_argument(
        '--svc-embedding-dim',
        type=int,
        default=DEFAULT_SVC_EMBED_DIM,
        help='Embedding dimension for svc backend.',
    )
    argparser.add_argument(
        '--variant',
        choices=jina_variants,
        default=DEFAULT_JINA_VARIANT,
        help='Jina model: jina-v3, jina-v3-classification, jina-v5, jina-v5-classification.',
    )
    argparser.add_argument(
        '--task',
        choices=('passage', 'query', 'classification'),
        default=DEFAULT_JINA_TASK,
        help='Jina task: passage (documents), query (search), classification.',
    )
    argparser.add_argument(
        '--embedding-dim',
        type=int,
        default=None,
        help='Jina output dimension when --embed-backend=jina (default: 1024).',
    )
    argparser.add_argument(
        '--langs',
        action='append',
        help='Language code(s) to fetch and embed. Repeat or use comma-separated values (default: en,cs,nl,fr,de).',
    )
    return argparser


def main() -> int:
    '''
    Run CLI entry point.

    :return: Process return code.
    '''
    argparser = build_arg_parser()
    logutil.addLogArguments(argparser)
    args = argparser.parse_args()
    logutil.configureFromArgs(args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    langs = parse_langs(raw_langs=args.langs)
    vectorizer, model_name = build_vectorizer(
        embed_backend=args.embed_backend,
        model_path=args.model_path,
        embed_svc_url=args.embed_svc_url,
        svc_embed_dim=args.svc_embedding_dim,
        jina_variant=args.variant,
        jina_task=args.task,
        jina_embed_dim=args.embedding_dim,
    )

    if args.input_source == 'files':
        qids = load_qids(ids_path=Path(args.ids)) if args.ids else None
        text_dir = Path(args.text_dir)
        if not text_dir.is_dir():
            raise FileNotFoundError(f'Text directory does not exist: {text_dir}')
        LOG.info(
            f'File input mode, text_dir={text_dir}, langs={langs}, '
            f'qid_filter={len(qids) if qids else "all"}, embed_backend={args.embed_backend}'
        )
        saved_count, skipped_count = compute_file_embeddings(
            text_dir=text_dir,
            langs=langs,
            out_dir=out_dir,
            vectorizer=vectorizer,
            model_name=model_name,
            qids=qids,
            batch_size=args.embed_batch_size,
            skip_existing=args.skip_existing,
        )
        LOG.info(
            f'Finished file embeddings. saved={saved_count} skipped={skipped_count} out_dir={out_dir}'
        )
        return 0

    ids_path = Path(args.ids or DEFAULT_IDS_PATH)
    qids = load_qids(ids_path=ids_path)
    LOG.info(
        f'Wikidata input mode, qids={len(qids)} from ids_path={ids_path}, langs={langs}, '
        f'embed_backend={args.embed_backend}'
    )

    sparql = UrlSparqlService(args.sparql_url)
    saved_count, missing_count = compute_description_embeddings(
        qids=qids,
        langs=langs,
        out_dir=out_dir,
        sparql=sparql,
        vectorizer=vectorizer,
        model_name=model_name,
    )
    LOG.info(
        f'Finished description embeddings. saved_pairs={saved_count} missing_pairs={missing_count} out_dir={out_dir}'
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())