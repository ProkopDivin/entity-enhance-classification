"""
Precompute Wikipedia2Vec entity embeddings for a list of Wikidata IDs.

Pipeline:
    1. Download pretrained Wikipedia2Vec model(s) (``.pkl.bz2``) if missing.
    2. Resolve each Wikidata QID to requested-language Wikipedia article titles using
       ``wbgetentities`` API (batches of 50, resumable TSV cache).
    3. For each requested language, load the matching model once and fetch entity vectors.
    4. Persist vectors as ``{QID}_{lang}_1.npy`` (float32) plus ``{QID}_{lang}_1.json``
       metadata so files are drop-in compatible with ``EntityEmbeddingStore``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

DEFAULT_IDS_PATH = 'data/gold-chrono-per-dataset/wdId_ids.txt'
DEFAULT_OUT_DIR = 'data/entity_embeddings/wikipedia2vec'
DEFAULT_MODEL_DIR = 'data/wikipedia2vec_models'
DEFAULT_MODEL_NAME = 'enwiki_20180420_win10_500d'
DEFAULT_MODEL_URL = (
    'http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/'
    'enwiki_20180420_win10_500d.pkl.bz2'
)
DEFAULT_DUMP_DATE = '2018-04-20'
DEFAULT_LANG = 'en'
CACHE_STATUS_OK = 'ok'
CACHE_STATUS_NO_SITELINK = 'no_sitelink'

WBGETENTITIES_URL = 'https://www.wikidata.org/w/api.php'
WBGETENTITIES_BATCH = 50
HTTP_TIMEOUT_S = 30
HTTP_RETRY = 8
HTTP_BACKOFF_S = 2.0
HTTP_RATE_LIMIT_DEFAULT_WAIT_S = 15.0
HTTP_RATE_LIMIT_MAX_WAIT_S = 90.0
DEFAULT_BATCH_SLEEP_S = 3.0

LOGGER = logging.getLogger(__name__)


def _flatten_lang_args(*, raw_langs: list[list[str]] | None) -> tuple[str, ...]:
    """
    Normalize ``--lang`` values from argparse into an ordered unique tuple.

    :param raw_langs: Nested list produced by argparse.
    :return: Ordered language codes.
    """
    if not raw_langs:
        return (DEFAULT_LANG,)

    seen: set[str] = set()
    langs: list[str] = []
    for group in raw_langs:
        for raw_lang in group:
            lang = raw_lang.strip().lower()
            if not lang or lang in seen:
                continue
            seen.add(lang)
            langs.append(lang)
    if not langs:
        raise ValueError('No valid language code provided via --lang.')
    return tuple(langs)


def _parse_lang_value_entries(*, entries: list[str], option_name: str) -> dict[str, str]:
    """
    Parse repeated ``lang=value`` CLI entries into a mapping.

    :param entries: Raw CLI values.
    :param option_name: Option name used in validation messages.
    :return: Language -> value mapping.
    """
    mapping: dict[str, str] = {}
    for entry in entries:
        text = entry.strip()
        if not text:
            continue
        if '=' not in text:
            raise ValueError(f'Invalid {option_name} entry "{entry}", expected LANG=VALUE.')
        lang_raw, value_raw = text.split('=', 1)
        lang = lang_raw.strip().lower()
        value = value_raw.strip()
        if not lang or not value:
            raise ValueError(f'Invalid {option_name} entry "{entry}", expected LANG=VALUE.')
        mapping[lang] = value
    return mapping


def _resolve_lang_values(
    *,
    langs: tuple[str, ...],
    default_value: str,
    mapped_values: dict[str, str],
    value_name: str,
) -> dict[str, str]:
    """
    Resolve per-language values with validation.

    For a single language, scalar defaults remain supported. For multiple languages,
    explicit ``lang=value`` mappings are required for every requested language.

    :param langs: Requested languages.
    :param default_value: Scalar CLI value.
    :param mapped_values: ``lang=value`` mapping from CLI.
    :param value_name: Human-readable value name for errors.
    :return: Language -> value mapping.
    """
    if len(langs) == 1 and not mapped_values:
        return {langs[0]: default_value}

    missing = [lang for lang in langs if lang not in mapped_values]
    if missing:
        raise ValueError(
            f'Missing {value_name} mapping for languages: {", ".join(missing)}. '
            f'Provide --{value_name}-map LANG=VALUE for each requested --lang.'
        )
    return {lang: mapped_values[lang] for lang in langs}


def _http_get_json(*, url: str, params: dict[str, str]) -> dict:
    """
    Perform a GET request returning parsed JSON with exponential-backoff retry.

    :param url: Request URL.
    :param params: Query parameters.
    :return: Decoded JSON response.
    """
    from urllib.parse import urlencode

    full_url = f'{url}?{urlencode(params)}'
    headers = {
        'User-Agent': (
            'iptc-entity-enhance-classification/0.1 '
            '(master-thesis; wikipedia2vec entity prefetch)'
        ),
    }
    last_err: Exception | None = None
    for attempt in range(1, HTTP_RETRY + 1):
        try:
            request = Request(full_url, headers=headers)
            with urlopen(request, timeout=HTTP_TIMEOUT_S) as response:
                return json.loads(response.read().decode('utf-8'))
        except HTTPError as err:
            last_err = err
            if err.code == 429:
                retry_after = err.headers.get('Retry-After') if err.headers else None
                try:
                    wait_s = float(retry_after) if retry_after else HTTP_RATE_LIMIT_DEFAULT_WAIT_S
                except ValueError:
                    wait_s = HTTP_RATE_LIMIT_DEFAULT_WAIT_S
                wait_s = min(
                    max(wait_s, HTTP_RATE_LIMIT_DEFAULT_WAIT_S) * min(attempt, 4),
                    HTTP_RATE_LIMIT_MAX_WAIT_S,
                )
                LOGGER.warning('Rate limited (attempt %d), sleeping %.1fs', attempt, wait_s)
            else:
                wait_s = HTTP_BACKOFF_S * attempt
                LOGGER.warning('HTTP %s on attempt %d, retrying in %.1fs', err.code, attempt, wait_s)
            time.sleep(wait_s)
        except Exception as err:
            last_err = err
            wait_s = HTTP_BACKOFF_S * attempt
            LOGGER.warning('HTTP attempt %d failed (%s), retrying in %.1fs', attempt, err, wait_s)
            time.sleep(wait_s)
    raise RuntimeError(f'HTTP GET failed after {HTTP_RETRY} attempts: {last_err}')


def _resolve_batch_titles(*, batch: list[str], langs: tuple[str, ...]) -> dict[str, dict[str, str | None]]:
    """
    Resolve a single batch of QIDs, transparently handling ``no-such-entity`` errors.

    The Wikidata ``wbgetentities`` endpoint returns a **top-level** error (and
    drops the whole ``entities`` payload) whenever any single ID in the batch is
    invalid / deleted. To avoid losing the 49 valid IDs accompanying one bad one,
    we peel off the offending ID reported in ``error.id`` and retry the rest.

    :param batch: List of QIDs to resolve in one call.
    :param langs: Requested language codes.
    :return: Mapping of every input QID to per-language title (or ``None``).
    """
    remaining = list(batch)
    resolved: dict[str, dict[str, str | None]] = {}
    sites = '|'.join(f'{lang}wiki' for lang in langs)
    guard = 0
    max_rounds = len(batch) + 2
    while remaining and guard < max_rounds:
        guard += 1
        params = {
            'action': 'wbgetentities',
            'ids': '|'.join(remaining),
            'props': 'sitelinks',
            'sitefilter': sites,
            'format': 'json',
            'maxlag': '5',
        }
        payload = _http_get_json(url=WBGETENTITIES_URL, params=params)
        error = payload.get('error')
        if error and error.get('code') == 'maxlag':
            wait_s = HTTP_RATE_LIMIT_DEFAULT_WAIT_S
            LOGGER.warning('Wikidata maxlag=%s, sleeping %.1fs', error.get('info'), wait_s)
            time.sleep(wait_s)
            continue
        if error and error.get('code') == 'no-such-entity':
            bad_id = error.get('id')
            if bad_id in remaining:
                resolved[bad_id] = {lang: None for lang in langs}
                remaining = [q for q in remaining if q != bad_id]
                LOGGER.debug('Skipping unknown QID %s', bad_id)
                continue
            LOGGER.warning('Unhandled no-such-entity error (id=%s); marking batch as missing', bad_id)
            for qid in remaining:
                resolved.setdefault(qid, {lang: None for lang in langs})
            return resolved
        if error:
            raise RuntimeError(f'Wikidata API error: {error}')

        entities = payload.get('entities') or {}
        for qid in remaining:
            entity = entities.get(qid) or {}
            sitelinks = entity.get('sitelinks') or {}
            resolved[qid] = {
                lang: ((sitelinks.get(f'{lang}wiki') or {}).get('title') or None)
                for lang in langs
            }
        remaining = []

    for qid in batch:
        resolved.setdefault(qid, {lang: None for lang in langs})
    return resolved


def _ensure_decompressed(*, model_path: Path) -> Path:
    """
    Return a path to the decompressed ``.pkl`` file, creating it if needed.

    Wikipedia2Vec's ``load()`` uses ``joblib`` which requires memory-mappable
    (uncompressed) files. Streaming decompression is not supported.

    :param model_path: Path to a ``.pkl.bz2`` file.
    :return: Path to the decompressed ``.pkl`` file.
    """
    import bz2
    import shutil

    if model_path.suffix != '.bz2':
        return model_path
    decompressed = model_path.with_suffix('')
    if decompressed.exists() and decompressed.stat().st_size > 0:
        LOGGER.info('Using already-decompressed model at %s', decompressed)
        return decompressed

    LOGGER.info('Decompressing %s -> %s (one-off, may take several minutes)', model_path, decompressed)
    tmp_path = decompressed.with_suffix(decompressed.suffix + '.part')
    with bz2.open(model_path, 'rb') as src, open(tmp_path, 'wb') as dst:
        shutil.copyfileobj(src, dst, length=1 << 20)
    tmp_path.rename(decompressed)
    LOGGER.info('Decompressed model size: %d bytes', decompressed.stat().st_size)
    return decompressed


def download_model(*, url: str, dest_dir: Path) -> Path:
    """
    Download the Wikipedia2Vec model ``.pkl.bz2`` to ``dest_dir`` if absent.

    :param url: Source URL of the compressed pickle.
    :param dest_dir: Directory to store the downloaded file.
    :return: Local path to the downloaded file.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    file_name = url.rsplit('/', 1)[-1]
    dest_path = dest_dir / file_name
    if dest_path.exists() and dest_path.stat().st_size > 0:
        LOGGER.info('Model already present at %s (%d bytes)', dest_path, dest_path.stat().st_size)
        return dest_path

    LOGGER.info('Downloading model from %s -> %s', url, dest_path)
    request = Request(url, headers={'User-Agent': 'iptc-entity-enhance-classification/0.1'})
    tmp_path = dest_path.with_suffix(dest_path.suffix + '.part')
    with urlopen(request, timeout=HTTP_TIMEOUT_S) as response, open(tmp_path, 'wb') as out_file:
        total = int(response.headers.get('Content-Length') or 0)
        written = 0
        chunk_size = 1 << 20
        next_log = 100 * chunk_size
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            written += len(chunk)
            if written >= next_log:
                pct = f'{100 * written / total:.1f}%' if total else 'n/a'
                LOGGER.info('Downloaded %d MB (%s)', written >> 20, pct)
                next_log += 100 * chunk_size
    tmp_path.rename(dest_path)
    LOGGER.info('Model downloaded: %s', dest_path)
    return dest_path


def load_qids(*, ids_path: Path) -> list[str]:
    """
    Read QIDs from a newline-delimited file, preserving order and de-duplicating.

    :param ids_path: Path to the QID list file.
    :return: Ordered list of unique QIDs.
    """
    seen: set[str] = set()
    qids: list[str] = []
    with open(ids_path, encoding='utf-8') as in_file:
        for line in in_file:
            qid = line.strip()
            if not qid or qid in seen:
                continue
            seen.add(qid)
            qids.append(qid)
    LOGGER.info('Loaded %d unique QIDs from %s', len(qids), ids_path)
    return qids


def load_title_cache(*, cache_path: Path) -> dict[str, dict[str, str | None]]:
    """
    Load the QID -> title TSV cache if it exists.

    A missing title is mapped to ``None``.
    Expected schema:
      - ``QID<TAB>lang<TAB>status<TAB>title``
      - ``status`` is ``ok`` or ``no_sitelink``.

    :param cache_path: Path to the cache TSV.
    :return: Mapping of QID to per-language title.
    """
    cache: dict[str, dict[str, str | None]] = {}
    if not cache_path.exists():
        return cache
    rows = 0
    with open(cache_path, encoding='utf-8') as in_file:
        for line in in_file:
            parts = line.rstrip('\n').split('\t')
            if len(parts) != 4 or not parts[0]:
                continue
            qid = parts[0]
            lang = parts[1].strip().lower()
            status = parts[2].strip().lower()
            title = parts[3]
            if not lang:
                continue
            if status == CACHE_STATUS_NO_SITELINK:
                cache.setdefault(qid, {})[lang] = None
                rows += 1
                continue
            if status != CACHE_STATUS_OK:
                continue
            cache.setdefault(qid, {})[lang] = title if title else None
            rows += 1
    LOGGER.info('Loaded %d cached QID/lang/title rows for %d QIDs from %s', rows, len(cache), cache_path)
    return cache


def _batched(items: list[str], *, size: int) -> Iterator[list[str]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def fetch_titles(
    *,
    qids: Iterable[str],
    langs: tuple[str, ...],
    cache_path: Path,
    batch_size: int = WBGETENTITIES_BATCH,
    sleep_s: float = DEFAULT_BATCH_SLEEP_S,
) -> dict[str, dict[str, str | None]]:
    """
    Resolve each QID to requested-language Wikipedia titles, using a resumable TSV cache.

    :param qids: Iterable of Wikidata QIDs.
    :param langs: Requested language codes.
    :param cache_path: Destination TSV; appended line-by-line.
    :param batch_size: API batch size (max 50).
    :param sleep_s: Politeness sleep between batches.
    :return: Full mapping of QID to per-language title or ``None``.
    """
    cache = load_title_cache(cache_path=cache_path)
    missing_langs_by_qid: dict[str, tuple[str, ...]] = {}
    for qid in qids:
        missing_langs = tuple(lang for lang in langs if lang not in cache.get(qid, {}))
        if missing_langs:
            missing_langs_by_qid[qid] = missing_langs

    todo_qids = list(missing_langs_by_qid)
    LOGGER.info(
        'QID -> title: cached_qids=%d requested_langs=%s qids_to_fetch=%d',
        len(cache),
        ','.join(langs),
        len(todo_qids),
    )
    if not todo_qids:
        return cache

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'a', encoding='utf-8') as cache_file:
        for batch_idx, batch in enumerate(_batched(todo_qids, size=batch_size), start=1):
            batch_langs = tuple(sorted({lang for qid in batch for lang in missing_langs_by_qid[qid]}))
            resolved = _resolve_batch_titles(batch=batch, langs=batch_langs)
            for qid in batch:
                qid_missing_langs = missing_langs_by_qid[qid]
                qid_lang_titles = resolved.get(qid) or {lang: None for lang in qid_missing_langs}
                cache.setdefault(qid, {})
                for lang in qid_missing_langs:
                    title = qid_lang_titles.get(lang)
                    cache[qid][lang] = title
                    status = CACHE_STATUS_OK if title else CACHE_STATUS_NO_SITELINK
                    cache_file.write(f'{qid}\t{lang}\t{status}\t{title or ""}\n')
            cache_file.flush()
            if batch_idx % 20 == 0:
                LOGGER.info('Fetched titles for %d / %d QIDs', batch_idx * batch_size, len(todo_qids))
            if sleep_s > 0:
                time.sleep(sleep_s)
    LOGGER.info('Finished fetching titles. Total cached QIDs: %d', len(cache))
    return cache


def _build_metadata(*, qid: str, lang: str, title: str, model_name: str, dump_date: str) -> dict:
    return {
        'id': f'{qid}_{lang}_1',
        'metadata': {
            'Language': lang,
            'QID': qid,
            'Source': 'wikipedia2vec',
            'Model': model_name,
            'DumpDate': dump_date,
            'WikipediaTitle': title,
            'ChunkID': 1,
        },
    }


def compute_embeddings(
    *,
    model_path: Path,
    qid_to_title: dict[str, str | None],
    out_dir: Path,
    lang: str,
    model_name: str,
    dump_date: str,
    missing_log_path: Path,
    missing_log_mode: str = 'a',
    overwrite: bool = False,
) -> tuple[int, int]:
    """
    Compute entity vectors and write ``.npy`` + ``.json`` for each resolvable QID.

    :param model_path: Local path to the Wikipedia2Vec ``.pkl.bz2`` file.
    :param qid_to_title: Mapping of QID to Wikipedia title or ``None``.
    :param out_dir: Directory where per-entity files are written.
    :param lang: Language code used in output file names.
    :param model_name: Model identifier embedded in the metadata.
    :param dump_date: Wikipedia dump date embedded in the metadata.
    :param missing_log_path: File where QIDs without an available vector are recorded.
    :param missing_log_mode: File mode for the missing-QID log.
    :param overwrite: If False, skip QIDs whose ``.npy`` already exists.
    :return: Pair ``(hits, misses)``.
    """
    from wikipedia2vec import Wikipedia2Vec

    LOGGER.info('Loading Wikipedia2Vec model from %s', model_path)
    if str(model_path).endswith('.bz2'):
        decompressed = _ensure_decompressed(model_path=model_path)
        wiki2vec = Wikipedia2Vec.load(str(decompressed))
    else:
        wiki2vec = Wikipedia2Vec.load(str(model_path))

    out_dir.mkdir(parents=True, exist_ok=True)
    missing_log_path.parent.mkdir(parents=True, exist_ok=True)

    hits = 0
    misses = 0
    total = len(qid_to_title)
    with open(missing_log_path, missing_log_mode, encoding='utf-8') as miss_file:
        for idx, (qid, title) in enumerate(qid_to_title.items(), start=1):
            npy_path = out_dir / f'{qid}_{lang}_1.npy'
            json_path = out_dir / f'{qid}_{lang}_1.json'
            if not overwrite and npy_path.exists() and json_path.exists():
                hits += 1
                continue

            if not title:
                miss_file.write(f'{qid}\t{lang}\tno_{lang}wiki_sitelink\n')
                misses += 1
                continue

            entity = wiki2vec.get_entity(title)
            if entity is None:
                miss_file.write(f'{qid}\t{lang}\tnot_in_model\t{title}\n')
                misses += 1
                continue

            vector = np.asarray(wiki2vec.get_vector(entity), dtype=np.float32)
            np.save(npy_path, vector)
            metadata = _build_metadata(
                qid=qid, lang=lang, title=title, model_name=model_name, dump_date=dump_date,
            )
            with open(json_path, 'w', encoding='utf-8') as json_file:
                json.dump(metadata, json_file, ensure_ascii=False, indent=2)
            hits += 1

            if idx % 5000 == 0:
                LOGGER.info('Embeddings progress: %d / %d (hits=%d, misses=%d)', idx, total, hits, misses)

    LOGGER.info('Embedding stage complete. Hits=%d, misses=%d, total=%d', hits, misses, total)
    return hits, misses


def recount_embeddings(
    *,
    cache_path: Path,
    out_dir: Path,
    langs: tuple[str, ...],
) -> dict[str, int]:
    """
    Recount already-materialized embeddings without fetching titles or loading the model.

    :param cache_path: Path to the QID -> title TSV cache.
    :param out_dir: Output directory with per-entity ``.npy`` + ``.json`` files.
    :param langs: Language codes used in output file names.
    :return: Counters keyed by category.
    """
    cache = load_title_cache(cache_path=cache_path)

    counts: dict[str, int] = {
        'total_pairs': 0,
        'cache_qids': 0,
        'pairs_with_title': 0,
        'embeddings_present': 0,
        'missing_files': 0,
        'title_known_but_missing_files': 0,
    }

    counts['cache_qids'] = len(cache)

    for qid, title_map in cache.items():
        for lang in langs:
            counts['total_pairs'] += 1
            title = title_map.get(lang)
            if not title:
                continue

            counts['pairs_with_title'] += 1
            npy_path = out_dir / f'{qid}_{lang}_1.npy'
            json_path = out_dir / f'{qid}_{lang}_1.json'
            if npy_path.exists() and json_path.exists():
                counts['embeddings_present'] += 1
                continue

            counts['missing_files'] += 1
            counts['title_known_but_missing_files'] += 1

    return counts


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build CLI parser for the download script.

    :return: Configured argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('--ids', default=DEFAULT_IDS_PATH, help='QID list file (one QID per line).')
    parser.add_argument('--out-dir', default=DEFAULT_OUT_DIR, help='Output directory for per-entity files.')
    parser.add_argument('--model-dir', default=DEFAULT_MODEL_DIR, help='Directory caching the downloaded model.')
    parser.add_argument(
        '--lang',
        dest='langs',
        action='append',
        nargs='+',
        default=None,
        help='Language code(s); repeat flag or pass multiple values, e.g. --lang en de --lang cs.',
    )
    parser.add_argument('--model-url', default=DEFAULT_MODEL_URL, help='Default model URL for single-language runs.')
    parser.add_argument(
        '--model-url-map',
        action='append',
        default=[],
        metavar='LANG=URL',
        help='Per-language model URL mapping; repeat for each language.',
    )
    parser.add_argument('--model-name', default=DEFAULT_MODEL_NAME, help='Default model name for single-language runs.')
    parser.add_argument(
        '--model-name-map',
        action='append',
        default=[],
        metavar='LANG=NAME',
        help='Per-language model name mapping; repeat for each language.',
    )
    parser.add_argument('--dump-date', default=DEFAULT_DUMP_DATE, help='Default dump date for single-language runs.')
    parser.add_argument(
        '--dump-date-map',
        action='append',
        default=[],
        metavar='LANG=DATE',
        help='Per-language dump date mapping; repeat for each language.',
    )
    parser.add_argument('--batch-size', type=int, default=WBGETENTITIES_BATCH, help='Wikidata API batch size (<=50).')
    parser.add_argument(
        '--sleep-s', type=float, default=DEFAULT_BATCH_SLEEP_S,
        help='Politeness delay between API batches (seconds).',
    )
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing per-entity files.')
    parser.add_argument('--skip-download', action='store_true', help='Do not download/check the model, assume it exists.')
    parser.add_argument('--titles-only', action='store_true', help='Only build the QID->title cache, skip embedding step.')
    parser.add_argument(
        '--recount-only',
        action='store_true',
        help='Only recount already written embeddings on disk; no HTTP calls, no model loading.',
    )
    parser.add_argument(
        '--compute-from-cache',
        action='store_true',
        help='Compute embeddings directly from cached titled QIDs in _qid_to_title.tsv (no Wikidata calls).',
    )
    parser.add_argument('--log-level', default='INFO', help='Logging level (DEBUG, INFO, WARNING, ...).')
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Entry point.

    :param argv: Optional CLI arguments (defaults to ``sys.argv[1:]``).
    :return: Process exit code.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    ids_path = Path(args.ids)
    out_dir = Path(args.out_dir)
    model_dir = Path(args.model_dir)
    cache_path = out_dir / '_qid_to_title.tsv'
    missing_log_path = out_dir / '_missing_qids.txt'
    langs = _flatten_lang_args(raw_langs=args.langs)
    try:
        model_urls = _resolve_lang_values(
            langs=langs,
            default_value=args.model_url,
            mapped_values=_parse_lang_value_entries(entries=args.model_url_map, option_name='--model-url-map'),
            value_name='model-url',
        )
        model_names = _resolve_lang_values(
            langs=langs,
            default_value=args.model_name,
            mapped_values=_parse_lang_value_entries(entries=args.model_name_map, option_name='--model-name-map'),
            value_name='model-name',
        )
        dump_dates = _resolve_lang_values(
            langs=langs,
            default_value=args.dump_date,
            mapped_values=_parse_lang_value_entries(entries=args.dump_date_map, option_name='--dump-date-map'),
            value_name='dump-date',
        )
    except ValueError as err:
        LOGGER.error(str(err))
        return 2

    LOGGER.info('Requested languages=%s', ','.join(langs))

    if args.recount_only:
        counts = recount_embeddings(
            cache_path=cache_path,
            out_dir=out_dir,
            langs=langs,
        )
        LOGGER.info(
            'Recount complete (cache-driven). cache_qids=%d total_pairs=%d pairs_with_title=%d',
            counts['cache_qids'],
            counts['total_pairs'],
            counts['pairs_with_title'],
        )
        LOGGER.info(
            'Recount results. embeddings_present=%d missing_files=%d',
            counts['embeddings_present'],
            counts['missing_files'],
        )
        LOGGER.info(
            'Recount breakdown. title_known_but_missing_files=%d',
            counts['title_known_but_missing_files'],
        )
        return 0

    if args.compute_from_cache:
        qid_to_titles = load_title_cache(cache_path=cache_path)
        LOGGER.info(
            'Compute-from-cache mode. cache_qids=%d requested_langs=%s',
            len(qid_to_titles),
            ','.join(langs),
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        missing_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(missing_log_path, 'w', encoding='utf-8'):
            pass

        total_hits = 0
        total_misses = 0
        for lang in langs:
            if args.skip_download:
                candidate = model_dir / model_urls[lang].rsplit('/', 1)[-1]
                if not candidate.exists():
                    LOGGER.error('Model file %s not found and --skip-download set', candidate)
                    return 2
                model_path = candidate
            else:
                model_path = download_model(url=model_urls[lang], dest_dir=model_dir)

            ordered_titles = {
                qid: title_map.get(lang)
                for qid, title_map in qid_to_titles.items()
                if title_map.get(lang)
            }
            LOGGER.info(
                'Compute-from-cache language=%s titles_with_value=%d',
                lang,
                len(ordered_titles),
            )
            hits, misses = compute_embeddings(
                model_path=model_path,
                qid_to_title=ordered_titles,
                out_dir=out_dir,
                lang=lang,
                model_name=model_names[lang],
                dump_date=dump_dates[lang],
                missing_log_path=missing_log_path,
                missing_log_mode='a',
                overwrite=args.overwrite,
            )
            total_hits += hits
            total_misses += misses
            LOGGER.info('Language complete. lang=%s hits=%d misses=%d', lang, hits, misses)

        LOGGER.info('Done (compute-from-cache). hits=%d misses=%d out_dir=%s', total_hits, total_misses, out_dir)
        return 0

    qids = load_qids(ids_path=ids_path)
    qid_to_titles = fetch_titles(
        qids=qids,
        langs=langs,
        cache_path=cache_path,
        batch_size=min(args.batch_size, WBGETENTITIES_BATCH),
        sleep_s=args.sleep_s,
    )

    for lang in langs:
        resolved = sum(1 for qid in qids if (qid_to_titles.get(qid) or {}).get(lang))
        LOGGER.info('Resolved %d / %d QIDs to %swiki titles', resolved, len(qids), lang)

    if args.titles_only:
        LOGGER.info('--titles-only set, exiting before embedding step')
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    missing_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(missing_log_path, 'w', encoding='utf-8'):
        pass

    total_hits = 0
    total_misses = 0
    for lang in langs:
        if args.skip_download:
            candidate = model_dir / model_urls[lang].rsplit('/', 1)[-1]
            if not candidate.exists():
                LOGGER.error('Model file %s not found and --skip-download set', candidate)
                return 2
            model_path = candidate
        else:
            model_path = download_model(url=model_urls[lang], dest_dir=model_dir)

        ordered_titles: dict[str, str | None] = {qid: (qid_to_titles.get(qid) or {}).get(lang) for qid in qids}
        hits, misses = compute_embeddings(
            model_path=model_path,
            qid_to_title=ordered_titles,
            out_dir=out_dir,
            lang=lang,
            model_name=model_names[lang],
            dump_date=dump_dates[lang],
            missing_log_path=missing_log_path,
            missing_log_mode='a',
            overwrite=args.overwrite,
        )
        total_hits += hits
        total_misses += misses
        LOGGER.info('Language complete. lang=%s hits=%d misses=%d', lang, hits, misses)

    LOGGER.info('Done. hits=%d misses=%d out_dir=%s', total_hits, total_misses, out_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
