'''
Compute description-based entity embeddings for selected Wikidata languages.
'''

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from geneea.catlib.vec.vectorizer import SvcTextVectorizer
from geneea.core import logutil
from geneea.kb.tools.rdfutil import UrlSparqlService

from iptc_entity_pipeline.wikipedia2vec_emb import DEFAULT_IDS_PATH

LOG = logutil.getLogger(__package__, __file__)

DEFAULT_OUT_DIR = 'data/entity_embeddings/WikidataProject'
DEFAULT_LANGS = ('en', 'cs', 'nl', 'fr', 'de')
DEFAULT_SPARQL_URL = 'http://psi.g:9999/bigdata/sparql'
DEFAULT_EMBED_SVC_URL = 'http://tau.g:5533'
DEFAULT_EMBED_DIM = 384
DEFAULT_SPARQL_BATCH_SIZE = 100


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


def _build_metadata(*, qid: str, lang: str, model_name: str) -> dict:
    return {
        'id': f'{qid}_{lang}_1',
        'metadata': {
            'Language': lang,
            'QID': qid,
            'Source': 'wikidata_description',
            'Model': model_name,
            'DumpDate': 'n/a',
            'WikipediaTitle': '',
            'ChunkID': 1,
        },
    }


def save_entity_embedding(
    *,
    out_dir: Path,
    qid: str,
    lang: str,
    vector: np.ndarray,
    model_name: str,
) -> None:
    '''
    Persist one entity embedding and metadata sidecar.

    :param out_dir: Output directory.
    :param qid: Wikidata QID.
    :param lang: Language code.
    :param vector: Embedding vector.
    :param model_name: Embedding model identifier.
    '''
    stem = f'{qid}_{lang}_1'
    npy_path = out_dir / f'{stem}.npy'
    json_path = out_dir / f'{stem}.json'
    np.save(npy_path, np.asarray(vector, dtype=np.float32))
    metadata = _build_metadata(qid=qid, lang=lang, model_name=model_name)
    with open(json_path, 'w', encoding='utf-8') as out_file:
        json.dump(metadata, out_file, ensure_ascii=False, indent=2)


def compute_description_embeddings(
    *,
    qids: Sequence[str],
    langs: Sequence[str],
    out_dir: Path,
    sparql: UrlSparqlService,
    vectorizer: SvcTextVectorizer,
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
                    vector=np.asarray(vector, dtype=np.float32),
                    model_name=model_name,
                )
                saved_count += 1
            missing_count += len(langs) - len(batch_langs)
    return saved_count, missing_count


def build_arg_parser() -> argparse.ArgumentParser:
    '''
    Build CLI argument parser.

    :return: Configured parser.
    '''
    argparser = argparse.ArgumentParser(description='Compute description entity embeddings.')
    argparser.add_argument('--out-dir', default=DEFAULT_OUT_DIR, help='Output directory for per-entity embedding files.')
    argparser.add_argument('--model-path', required=True, help='Embedding service model id used for descriptions.')
    argparser.add_argument('--ids', default=DEFAULT_IDS_PATH, help='QID list file (one QID per line).')
    argparser.add_argument('--sparql-url', default=DEFAULT_SPARQL_URL, help='SPARQL endpoint URL.')
    argparser.add_argument('--embed-svc-url', default=DEFAULT_EMBED_SVC_URL, help='Embedding service URL.')
    argparser.add_argument('--embedding-dim', type=int, default=DEFAULT_EMBED_DIM, help='Embedding dimension.')
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

    qids = load_qids(ids_path=Path(args.ids))
    LOG.info(f'Loaded qids={len(qids)} from ids_path={args.ids}')

    sparql = UrlSparqlService(args.sparql_url)
    vectorizer = SvcTextVectorizer(
        embedSvcUrl=args.embed_svc_url,
        modelId=args.model_path,
        embedDim=args.embedding_dim,
    )
    saved_count, missing_count = compute_description_embeddings(
        qids=qids,
        langs=DEFAULT_LANGS,
        out_dir=out_dir,
        sparql=sparql,
        vectorizer=vectorizer,
        model_name=args.model_path,
    )
    LOG.info(
        f'Finished description embeddings. saved_pairs={saved_count} missing_pairs={missing_count} out_dir={out_dir}'
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())