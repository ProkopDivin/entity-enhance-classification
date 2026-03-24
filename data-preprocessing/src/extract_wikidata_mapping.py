#!/usr/bin/env python3
"""
Extract Wikidata IDs from GKB items and create mapping.
Reads gkbId values from an article–entity TSV (entities column: JSON array per row)
and extracts Wikidata IDs from source attributes.
Produces two outputs:
1. A file with unique Wikidata IDs
2. A mapping file of GKB IDs to Wikidata IDs (TSV format)
"""
import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from requests import Session

# Since this is a CLI script, print is acceptable for output
# For general-purpose tools, logging should be used instead

# Try to import GKB access methods
HAS_GKB_CLIENT = False
try:
    from geneea.kb.client import common

    HAS_GKB_CLIENT = True
except ImportError:
    pass

RE_WIKIDATA_ID_PATTERN = re.compile(r"^Q[0-9]+$")


def wdId_from_item(item: dict) -> list[str]:
    """
    Extract Wikidata IDs from source values in a GKB item.
    Extracts Q numbers from URLs like "http://www.wikidata.org/entity/Q98918715".

    :param item: GKB item dictionary
    :return: List of Wikidata IDs (e.g., ["Q98918715"])
    """
    source_value = item.get("source")
    if isinstance(source_value, str):
        sources = [source_value]
    elif isinstance(source_value, list):
        sources = [source for source in source_value if isinstance(source, str)]
    else:
        return []
    wikidata_ids: list[str] = []
    for source in sources:
        if "wikidata.org" in source.lower() and "/entity/" in source:
            candidate = source.split("/entity/", maxsplit=1)[-1].split("?", maxsplit=1)[0].split("#", maxsplit=1)[0]
        else:
            candidate = source
        if candidate.startswith("Q") and candidate[1:].isdigit():
            wikidata_ids.append(candidate)
    return wikidata_ids


def gkbId_from_map(path: Path) -> list[str]:
    """
    Collect unique GKB IDs from the ``entities`` column (JSON array of objects with ``gkbId``).

    :param path: TSV path with header including ``article_id`` and ``entities``
    :return: Unique non-empty gkbId strings (order not guaranteed)
    """
    gkb_ids: set[str] = set()
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames or "entities" not in reader.fieldnames:
            raise ValueError(f"Expected TSV header with an 'entities' column; got {reader.fieldnames!r}")
        for row in reader:
            raw = (row.get("entities") or "").strip()
            if not raw:
                continue
            entities = json.loads(raw)
            for ent in entities:
                if not isinstance(ent, dict):
                    continue
                gid = ent.get("gkbId")
                if gid is None:
                    continue
                s = str(gid).strip()
                if s:
                    gkb_ids.add(s)
    return list(gkb_ids)


@dataclass(frozen=True)
class FetchStats:
    """Aggregated results from GKB item processing."""

    wikidata_ids: set[str]
    gid_to_wdid_mapping: dict[str, list[str]]
    processed: int
    failed: int
    items_without_source: int


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.

    :return: Parsed CLI arguments
    """
    argparser = argparse.ArgumentParser(
        description=(
            "Extract Wikidata IDs from GKB items and create mapping. "
            "Reads gkbId values from the entities column of an article-entity TSV "
            "and extracts Wikidata IDs from source attributes."
        )
    )
    argparser.add_argument(
        "article_entity_tsv",
        type=str,
        help=(
            "TSV with header including 'entities': JSON array of objects; "
            "each object may contain 'gkbId' (e.g. new-ar_2_ent.tsv)."
        ),
    )
    argparser.add_argument(
        "-o",
        "--output-prefix",
        type=str,
        default="wikidata",
        help="Prefix for output files (default: wikidata). " "Will create {prefix}_ids.txt and {prefix}_mapping.tsv",
    )
    argparser.add_argument(
        "--url",
        type=str,
        default=None,
        help=(
            "GKB service URL. If omitted, the script does not call GKB and instead "
            "converts GKB IDs to Wikidata IDs by replacing the leading G with Q."
        ),
    )
    argparser.add_argument(
        "--bucket",
        type=str,
        default="generic",
        help="GKB bucket name (default: generic)",
    )
    return argparser.parse_args()


def load_gkb_ids(article_entity_path: Path) -> list[str]:
    print(f"Reading GKB IDs from entities column in {article_entity_path}...")
    gkb_ids = gkbId_from_map(path=article_entity_path)
    print(f"Found {len(gkb_ids)} unique GKB IDs to process")
    if not gkb_ids:
        raise ValueError("No GKB IDs found.")
    return gkb_ids


def gkb_to_wikidata_id(gkb_id: str) -> str | None:

    if not gkb_id.startswith("G"):
        return None
    wikidata_id = "Q" + gkb_id[1:]
    if RE_WIKIDATA_ID_PATTERN.match(wikidata_id):
        return wikidata_id
    return None


def maping_no_url(gkb_ids: list[str]) -> FetchStats:
    """
    Build Wikidata mappings directly from GKB IDs without calling GKB.

    :param gkb_ids: GKB IDs to convert
    :return: Aggregated processing statistics and mappings
    """
    wikidata_ids: set[str] = set()
    gid_to_wdid_mapping: dict[str, list[str]] = {}
    processed = 0
    items_without_source = 0

    for gid in gkb_ids:
        wikidata_id = gkb_to_wikidata_id(gkb_id=gid)
        if wikidata_id is None:
            items_without_source += 1
            continue
        wikidata_ids.add(wikidata_id)
        gid_to_wdid_mapping[gid] = [wikidata_id]
        processed += 1

    return FetchStats(
        wikidata_ids=wikidata_ids,
        gid_to_wdid_mapping=gid_to_wdid_mapping,
        processed=processed,
        failed=0,
        items_without_source=items_without_source,
    )


def get_item_def(gkb_item: object) -> dict:

    if hasattr(gkb_item, "definition"):
        return gkb_item.definition
    if hasattr(gkb_item, "__iter__"):
        return dict(gkb_item)
    return gkb_item


def fetch_mapping(gkb_ids: list[str], url: str) -> FetchStats:
    """
    Fetch GKB items and extract Wikidata mappings from their source attributes.

    :param gkb_ids: GKB IDs to process
    :param url: GKB service URL
    :return: Aggregated processing statistics and mappings
    """
    wikidata_ids: set[str] = set()
    gid_to_wdid_mapping: dict[str, list[str]] = {}
    processed = 0
    failed = 0
    items_without_source = 0

    session = Session()
    try:
        for gid, gkb_item in common.getGkbItemsStream(gkb_ids, url=url, session=session):
            if gkb_item is None:
                failed += 1
                continue

            item_dict = get_item_def(gkb_item=gkb_item)
            item_wikidata_ids = wdId_from_item(item=item_dict)
            if item_wikidata_ids:
                wikidata_ids.update(item_wikidata_ids)
                gid_to_wdid_mapping[gid] = item_wikidata_ids
                processed += 1
            else:
                items_without_source += 1
                print(f"  Item {gid} has no source attribute")

            if processed > 0 and processed % 1000 == 0:
                print(f"  Processed {processed} items, found {len(wikidata_ids)} unique Wikidata IDs so far...")
    finally:
        session.close()

    return FetchStats(
        wikidata_ids=wikidata_ids,
        gid_to_wdid_mapping=gid_to_wdid_mapping,
        processed=processed,
        failed=failed,
        items_without_source=items_without_source,
    )


def write_wd_ids(output_ids_path: Path, wikidata_ids: list[str]) -> None:

    with open(output_ids_path, "w", encoding="utf-8") as f:
        for wdid in wikidata_ids:
            f.write(wdid + "\n")


def write_mapping(output_mapping_path: Path, gid_to_wdid_mapping: dict[str, list[str]]) -> None:

    with open(output_mapping_path, "w", encoding="utf-8") as f:
        f.write("gkb_id\twikidata_ids\n")
        for gid in sorted(gid_to_wdid_mapping.keys()):
            wdids_str = "|".join(sorted(gid_to_wdid_mapping[gid]))
            f.write(f"{gid}\t{wdids_str}\n")


def main() -> None:
    """Main entry point for the script."""
    args = parse_args()

    article_entity_path = Path(args.article_entity_tsv)
    output_ids_path = Path(f"{args.output_prefix}_ids.txt")
    output_mapping_path = Path(f"{args.output_prefix}_mapping.tsv")

    try:
        filtered_gids = load_gkb_ids(article_entity_path=article_entity_path)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.url:
        if not HAS_GKB_CLIENT:
            print(
                "Error: geneea.kb.client.common module not found. Please install geneea.kb.tools package.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"\nFetching GKB items from {args.url} (bucket: {args.bucket})...")
        print("Processing GKB items in batch...")
        try:
            stats = fetch_mapping(gkb_ids=filtered_gids, url=args.url)
        except Exception as e:
            print(f"Error fetching GKB items: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("\nNo --url provided, converting GKB IDs to Wikidata IDs locally...")
        stats = maping_no_url(gkb_ids=filtered_gids)

    print(f"\nProcessed {stats.processed} items successfully")
    if stats.items_without_source > 0:
        print(f"Found {stats.items_without_source} items without source attribute")
    if stats.failed > 0:
        print(f"Failed to process {stats.failed} items", file=sys.stderr)

    sorted_wikidata_ids = sorted(stats.wikidata_ids)
    write_wd_ids(output_ids_path=output_ids_path, wikidata_ids=sorted_wikidata_ids)
    print(f"\nExtracted {len(sorted_wikidata_ids)} unique Wikidata IDs")
    print(f"Output written to: {output_ids_path}")

    write_mapping(
        output_mapping_path=output_mapping_path,
        gid_to_wdid_mapping=stats.gid_to_wdid_mapping,
    )
    print(f"Created mapping for {len(stats.gid_to_wdid_mapping)} GKB items")
    print(f"Mapping written to: {output_mapping_path}")
    print("Done!")


if __name__ == "__main__":
    main()
