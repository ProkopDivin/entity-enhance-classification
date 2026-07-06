'''
Compute per-entity embeddings for the IPTC entity-enhanced pipeline.

This module builds entity embedding files compatible with
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

  PYTHONPATH=data-preprocessing/src:src python -m entity_embeddings \\
    --input-source wikidata \\
    --ids data/gold-chrono-per-dataset/wdId_ids.txt \\
    --out-dir data/entity_embeddings/WikidataDescription_jina \\
    --variant jina-v3 --task passage --langs en,cs,de

Wikipedia intro texts from cuted-articles (English only)::

  PYTHONPATH=data-preprocessing/src:src python -m entity_embeddings \\
    --input-source files \\
    --text-dir data/cuted-articles \\
    --out-dir data/entity_embeddings/cuted-article-embeddings-jina-v3 \\
    --variant jina-v3 --task passage --langs en --skip-existing
'''

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from geneea.core import logutil
from geneea.kb.tools.rdfutil import UrlSparqlService
from entity_embeddings.jina_embed import JinaModelVariant

from entity_embeddings.compute import (
    build_vectorizer,
    compute_description_embeddings,
    compute_file_embeddings,
    load_qids,
    parse_langs,
)
from entity_embeddings.constants import (
    DEFAULT_EMBED_BACKEND,
    DEFAULT_EMBED_BATCH_SIZE,
    DEFAULT_EMBED_SVC_URL,
    DEFAULT_IDS_PATH,
    DEFAULT_INPUT_SOURCE,
    DEFAULT_JINA_TASK,
    DEFAULT_JINA_VARIANT,
    DEFAULT_OUT_DIR,
    DEFAULT_SPARQL_URL,
    DEFAULT_SVC_EMBED_DIM,
    DEFAULT_TEXT_DIR,
)

LOG = logutil.getLogger(__package__, __file__)


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
