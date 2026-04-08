#!/usr/bin/env python3
"""
Describe original corpora and generate LaTeX tables for thesis.
Reads article rows from all-corpora-*.csv, entities from article_2_entities.tsv,
and computes article, entity, category, and time statistics per corpus.
"""
import argparse
import csv
import json
import logging
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np  # type: ignore[import-not-found]
import pandas as pd  # type: ignore[import-not-found]

from geneea.mediacats.iptc import IptcTopics  # type: ignore[import-not-found]

LOG = logging.getLogger(__name__)

ALL_CORPORA_FILES = (
    "all-corpora-train.csv",
    "all-corpora-dev.csv",
    "all-corpora-test.csv",
    "all-corpora-train-downsampled.csv",
    "silver-corpora-train-annot-openai.csv",
)
ARTICLE_ENTITIES_TSV = "article_2_entities.tsv"
GKB_WDID_MAPPING_TSV = "wdid_mapping_mapping.tsv"
DATE_STATS_FILE = "stats.md"
OUTPUT_DIR_NAME = "thesis_corpora_tables"
OUTPUT_SUMMARY_FILE = "all_corpora_summary_basic.tex"
OUTPUT_AGGREGATED_FILE = "all_corpora_aggregated.tex"
CORPUS_TABLES_SUFFIX = "_tables.tex"
MISSING_ENTITIES_SUFFIX = "_missing_entities.txt"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AggStats:
    """Aggregated descriptive statistics over a list of values."""

    mean: float = 0.0
    std: float = 0.0
    min: int = 0
    max: int = 0


@dataclass(frozen=True)
class EntityStats:
    """Per-corpus entity statistics."""

    entities_per_doc: AggStats = AggStats()
    entities_with_wdid: AggStats = AggStats()
    entities_without_wdid: AggStats = AggStats()
    articles_without_entities: int = 0


@dataclass(frozen=True)
class ArticleLengthStats:
    """Per-corpus article length statistics (characters and words)."""

    num_docs: int = 0
    article_len: AggStats = AggStats()
    title_len: AggStats = AggStats()
    lead_len: AggStats = AggStats()
    text_len: AggStats = AggStats()


@dataclass(frozen=True)
class CategoryStats:
    """Per-corpus category and date statistics."""

    num_docs: int = 0
    docs_with_date: int = 0
    docs_without_date: int = 0
    cats_per_doc: AggStats = AggStats()
    cat_doc_counts: Counter = field(default_factory=Counter)


@dataclass(frozen=True)
class RenderedTable:
    """Rendered LaTeX table and its tabular data for CSV export."""

    file_stem: str
    latex: str
    df: pd.DataFrame


@dataclass(frozen=True)
class CorpusEntitySummary:
    """Per-corpus entity summary metrics for the global entity summary table."""

    corpus_name: str
    entities_mean: float
    linked_mean: float
    linked_std: float
    linked_relevance_mean: float
    linked_relevance_std: float


# ---------------------------------------------------------------------------
# Standalone utilities
# ---------------------------------------------------------------------------


def extract_gkb_id(entity: Mapping) -> str:
    """Extract normalized gkbId value from an entity."""
    gkb_id = entity.get("gkbId") or entity.get("gkbID")
    if gkb_id is None:
        return ""
    return str(gkb_id).strip()


def parse_wdid_token(token: str) -> str:
    """Return normalized Wikidata ID when valid (Q + digits), else empty string."""
    value = token.strip()
    if not value:
        return ""
    lower = value.lower()
    if lower.startswith("http://www.wikidata.org/entity/") or lower.startswith("https://www.wikidata.org/entity/"):
        value = value.rsplit("/", maxsplit=1)[-1]
    if value.startswith("wd:"):
        value = value.split(":", maxsplit=1)[-1]
    if len(value) < 2 or not value.startswith("Q"):
        return ""
    if not value[1:].isdigit():
        return ""
    return value


def parse_first_wdid(raw_value: object) -> str:
    """Parse first valid wdId from a single or pipe-separated value."""
    if raw_value is None:
        return ""
    for token in str(raw_value).split("|"):
        parsed = parse_wdid_token(token)
        if parsed:
            return parsed
    return ""


def parse_float_maybe(value: object) -> float | None:
    """Try to parse numeric value; return None for invalid values."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and np.isnan(value):
            return None
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def is_entity_linked(entity: Mapping, gkb_to_wdid: Mapping[str, str]) -> bool:
    """Entity is linked only when gkbId/gkbID resolves to a wdId through mapping."""
    gkb_id = extract_gkb_id(entity)
    if not gkb_id:
        return False
    return bool(gkb_to_wdid.get(gkb_id))


def parse_categories(cats_str: str | None) -> list[str]:
    """Parse categories from pipe-separated string."""
    if not cats_str or pd.isna(cats_str):
        return []
    return [cat.strip() for cat in str(cats_str).split("|") if cat.strip()]


def word_count(value: object) -> int:
    """Count words in a value, returning 0 for None/NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0
    return len(str(value).split())


def write_csv_table(output_dir: Path, file_stem: str, df: pd.DataFrame) -> None:
    """Write a table DataFrame to CSV next to LaTeX outputs."""
    path = output_dir / f"{file_stem}.csv"
    df.to_csv(path, index=False)


def pick_first_existing_path(candidates: Iterable[Path]) -> Path | None:
    """Return first existing file path from candidates."""
    for path in candidates:
        if path.is_file():
            return path
    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def iter_all_corpora_csv_rows(origin_dir: Path, *, train_only: bool = False) -> Iterable[dict]:
    """
    Yield raw CSV rows from all-corpora CSV files.

    :param origin_dir: directory containing the all-corpora-*.csv files.
    :param train_only: if True, only read CSV files whose filename contains ``train`` (case-insensitive).
    """
    for name in ALL_CORPORA_FILES:
        if train_only and "train" not in name.lower():
            continue
        path = origin_dir / name
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "id" not in (reader.fieldnames or []):
                continue
            yield from reader


def load_corpus_rows(origin_dir: Path, *, train_only: bool = False) -> dict[str, list[dict]]:
    """
    Load all-corpora CSV rows grouped by corpus name (from metadata.corpusName).

    :param origin_dir: directory containing the all-corpora-*.csv files.
    :param train_only: if True, only load from CSV files whose filename contains ``train``.
        Entity statistics use ``extract_article_ids(rows)`` for each corpus, so they match this split.
    :return: mapping corpus_name -> list of article row dicts.
    """
    rows_by_corpus: dict[str, list[dict]] = {}
    for row in iter_all_corpora_csv_rows(origin_dir=origin_dir, train_only=train_only):
        doc_id = (row.get("id") or "").strip()
        meta_str = row.get("metadata") or ""
        if not doc_id or not meta_str:
            continue
        try:
            meta = json.loads(meta_str)
        except json.JSONDecodeError:
            continue
        corpus_name = meta.get("corpusName") if isinstance(meta, dict) else None
        if not corpus_name:
            continue
        rows_by_corpus.setdefault(corpus_name, []).append(dict(row))
    return rows_by_corpus


def load_date_stats(stats_md_path: Path) -> dict[str, dict[str, int]]:
    """
    Load per-dataset date statistics from stats.md produced by analyze_jsonl_stats.py.

    :return: mapping dataset -> {'total': ..., 'missing': ...}.
    """
    if not stats_md_path.is_file():
        return {}

    stats: dict[str, dict[str, int]] = {}
    with stats_md_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("|") or line.startswith("| Dataset"):
                continue
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if len(parts) < 3 or parts[0] in {"--------", "**TOTAL**"}:
                continue
            dataset, total_str, missing_str = parts[:3]
            try:
                total = int(total_str.replace(",", "").replace("*", ""))
                missing = int(missing_str.replace(",", "").replace("*", ""))
            except ValueError:
                continue
            stats[dataset] = {"total": total, "missing": missing}
    return stats


def load_article_entities_tsv(article_entities_tsv: Path) -> dict[str, list[dict]]:
    """Load article_id -> entities JSON list mapping from TSV."""
    entities_by_article: dict[str, list[dict]] = {}
    if not article_entities_tsv.is_file():
        return entities_by_article

    max_field_limit = 10**9
    try:
        csv.field_size_limit(max_field_limit)
    except OverflowError:
        csv.field_size_limit(2**31 - 1)

    with article_entities_tsv.open(encoding="utf-8") as in_file:
        reader = csv.DictReader(in_file, delimiter="\t")
        if not reader.fieldnames:
            return entities_by_article
        article_key = "article_id" if "article_id" in reader.fieldnames else "id"
        entities_key = "entities" if "entities" in reader.fieldnames else None
        if entities_key is None:
            return entities_by_article

        malformed_rows = 0
        for row in reader:
            article_id = (row.get(article_key) or "").strip()
            if not article_id:
                continue
            entities_raw = row.get(entities_key)
            if not entities_raw:
                entities_by_article[article_id] = []
                continue
            try:
                parsed = json.loads(entities_raw)
            except json.JSONDecodeError:
                malformed_rows += 1
                continue

            entities: list[dict] = []
            if isinstance(parsed, list):
                entities = [item for item in parsed if isinstance(item, Mapping)]
            elif isinstance(parsed, Mapping) and isinstance(parsed.get("entities"), list):
                entities = [item for item in parsed["entities"] if isinstance(item, Mapping)]

            entities_by_article[article_id] = entities

    if malformed_rows > 0:
        LOG.warning(f"Skipped {malformed_rows:,} malformed entity JSON rows from {article_entities_tsv}")
    return entities_by_article


def load_gkb_wdid_mapping_tsv(mapping_tsv: Path) -> dict[str, str]:
    """Load gkb_id -> first valid wdId mapping from TSV."""
    gkb_to_wdid: dict[str, str] = {}
    if not mapping_tsv.is_file():
        return gkb_to_wdid

    with mapping_tsv.open(encoding="utf-8") as in_file:
        reader = csv.DictReader(in_file, delimiter="\t")
        if not reader.fieldnames:
            return gkb_to_wdid

        fieldnames = set(reader.fieldnames)
        gkb_key = "gkb_id" if "gkb_id" in fieldnames else ("gkbId" if "gkbId" in fieldnames else None)
        wdid_key = (
            "wikidata_ids"
            if "wikidata_ids" in fieldnames
            else ("wdid" if "wdid" in fieldnames else ("wdId" if "wdId" in fieldnames else None))
        )
        if gkb_key is None or wdid_key is None:
            return gkb_to_wdid

        for row in reader:
            gkb_id = (row.get(gkb_key) or "").strip()
            if not gkb_id:
                continue
            wdid = parse_first_wdid(row.get(wdid_key))
            if not wdid:
                continue
            gkb_to_wdid[gkb_id] = wdid
    return gkb_to_wdid


def extract_article_ids(rows: list[dict]) -> set[str]:
    """Extract non-empty article IDs from a list of row dicts."""
    return {(r.get("id") or "").strip() for r in rows} - {""}


# ---------------------------------------------------------------------------
# StatsCalculator
# ---------------------------------------------------------------------------


class StatsCalculator:
    """Computes corpus-level article, entity, and category statistics."""

    @staticmethod
    def _agg(values: list[int]) -> AggStats:
        """Compute mean, std, min, max over a list of integer values."""
        if not values:
            return AggStats()
        arr = np.asarray(values, dtype=float)
        return AggStats(
            mean=float(arr.mean()),
            std=float(arr.std()),
            min=int(arr.min()),
            max=int(arr.max()),
        )

    @staticmethod
    def compute_article_length_stats(rows: Iterable[dict]) -> ArticleLengthStats:
        """Compute article length statistics in words from document rows."""
        title_lengths: list[int] = []
        lead_lengths: list[int] = []
        text_lengths: list[int] = []
        article_lengths: list[int] = []

        for doc in rows:
            title_wc = word_count(doc.get("title"))
            lead_wc = word_count(doc.get("lead"))
            text_wc = word_count(doc.get("text"))

            title_lengths.append(title_wc)
            lead_lengths.append(lead_wc)
            text_lengths.append(text_wc)
            article_lengths.append(title_wc + lead_wc + text_wc)

        agg = StatsCalculator._agg
        return ArticleLengthStats(
            num_docs=len(title_lengths),
            article_len=agg(article_lengths),
            title_len=agg(title_lengths),
            lead_len=agg(lead_lengths),
            text_len=agg(text_lengths),
        )

    @staticmethod
    def compute_entity_stats(
        entities_per_article: Iterable[list[dict]],
        gkb_to_wdid: Mapping[str, str],
    ) -> EntityStats:
        """
        Compute entity-related stats from per-article entity lists.

        :param entities_per_article: iterable where each element is a list of entity dicts for one article.
        """
        counts_per_doc: list[int] = []
        counts_with_wdid: list[int] = []
        counts_without_wdid: list[int] = []

        for entities in entities_per_article:
            counts_per_doc.append(len(entities))
            with_wdid = sum(1 for ent in entities if isinstance(ent, Mapping) and is_entity_linked(ent, gkb_to_wdid))
            counts_with_wdid.append(with_wdid)
            counts_without_wdid.append(len(entities) - with_wdid)

        agg = StatsCalculator._agg
        return EntityStats(
            entities_per_doc=agg(counts_per_doc),
            entities_with_wdid=agg(counts_with_wdid),
            entities_without_wdid=agg(counts_without_wdid),
            articles_without_entities=sum(1 for n in counts_per_doc if n == 0),
        )

    @staticmethod
    def compute_entity_stats_from_mapping(
        article_ids: Iterable[str],
        entities_by_article: Mapping[str, list[dict]],
        gkb_to_wdid: Mapping[str, str],
    ) -> tuple[EntityStats, list[str], list[str]]:
        """
        Compute entity stats using article_id -> entities mapping.

        :return: tuple of (entity stats, IDs not in mapping, IDs with zero entities).
        """
        ids_list = list(article_ids)
        missing_ids: list[str] = []
        empty_ids: list[str] = []
        entities_per_article: list[list[dict]] = []

        for doc_id in ids_list:
            entities = entities_by_article.get(str(doc_id))
            if entities is None:
                missing_ids.append(doc_id)
                entities_per_article.append([])
            elif len(entities) == 0:
                empty_ids.append(doc_id)
                entities_per_article.append([])
            else:
                entities_per_article.append(entities)

        stats = StatsCalculator.compute_entity_stats(entities_per_article=entities_per_article, gkb_to_wdid=gkb_to_wdid)
        return stats, missing_ids, empty_ids

    @staticmethod
    def compute_category_stats(rows: Iterable[dict]) -> CategoryStats:
        """Compute document-level category statistics for a corpus."""
        doc_count = 0
        docs_with_date = 0
        cats_per_doc: list[int] = []
        cat_doc_counts: Counter[str] = Counter()

        for doc in rows:
            doc_count += 1
            metadata = doc.get("metadata") or {}
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}
            if isinstance(metadata, Mapping) and metadata.get("date"):
                docs_with_date += 1

            cats = parse_categories(str(doc.get("cats")))
            cats_per_doc.append(len(cats))
            for cat in set(cats):
                cat_doc_counts[cat] += 1

        return CategoryStats(
            num_docs=doc_count,
            docs_with_date=docs_with_date,
            docs_without_date=doc_count - docs_with_date,
            cats_per_doc=StatsCalculator._agg(cats_per_doc),
            cat_doc_counts=cat_doc_counts,
        )

    @staticmethod
    def compute_entity_type_distribution(
        article_ids: Iterable[str],
        entities_by_article: Mapping[str, list[dict]],
        gkb_to_wdid: Mapping[str, str],
    ) -> tuple[Counter[str], int, dict[str, float]]:
        """
        Count entity types among linked entities.

        :return: tuple of (type_counts Counter, total_linked count, type -> average relevance).
        """
        type_counts: Counter[str] = Counter()
        total_linked = 0
        relevance_values_by_type: dict[str, list[float]] = {}

        for doc_id in article_ids:
            entities = entities_by_article.get(str(doc_id), [])
            for ent in entities:
                if not isinstance(ent, Mapping):
                    continue
                if not is_entity_linked(ent, gkb_to_wdid):
                    continue
                ent_type = str(ent.get("type") or "unknown")
                type_counts[ent_type] += 1
                total_linked += 1
                feats = ent.get("feats")
                if isinstance(feats, Mapping):
                    relevance = parse_float_maybe(feats.get("relevance"))
                    if relevance is not None:
                        relevance_values_by_type.setdefault(ent_type, []).append(relevance)

        avg_relevance_by_type: dict[str, float] = {}
        for ent_type, values in relevance_values_by_type.items():
            avg_relevance_by_type[ent_type] = float(np.mean(values)) if values else 0.0

        return type_counts, total_linked, avg_relevance_by_type

    @staticmethod
    def compute_linked_relevance_stats(
        article_ids: Iterable[str],
        entities_by_article: Mapping[str, list[dict]],
        gkb_to_wdid: Mapping[str, str],
    ) -> AggStats:
        """Compute relevance statistics for linked entities."""
        relevance_values: list[float] = []
        for doc_id in article_ids:
            entities = entities_by_article.get(str(doc_id), [])
            for ent in entities:
                if not isinstance(ent, Mapping):
                    continue
                if not is_entity_linked(ent, gkb_to_wdid):
                    continue
                feats = ent.get("feats")
                if not isinstance(feats, Mapping):
                    continue
                relevance = parse_float_maybe(feats.get("relevance"))
                if relevance is None:
                    continue
                relevance_values.append(relevance)
        if not relevance_values:
            return AggStats()
        arr = np.asarray(relevance_values, dtype=float)
        return AggStats(
            mean=float(arr.mean()),
            std=float(arr.std()),
            min=int(arr.min()),
            max=int(arr.max()),
        )

    @staticmethod
    def compute_entity_summary(
        corpus_name: str,
        article_ids: Iterable[str],
        entities_by_article: Mapping[str, list[dict]],
        gkb_to_wdid: Mapping[str, str],
    ) -> CorpusEntitySummary:
        """Compute per-corpus entity summary metrics for global summary table."""
        per_doc_entities: list[int] = []
        per_doc_linked: list[int] = []
        relevance_stats = StatsCalculator.compute_linked_relevance_stats(
            article_ids=article_ids,
            entities_by_article=entities_by_article,
            gkb_to_wdid=gkb_to_wdid,
        )

        for doc_id in article_ids:
            entities = entities_by_article.get(str(doc_id))
            entities_list = entities if entities is not None else []
            per_doc_entities.append(len(entities_list))
            linked_count = 0

            for ent in entities_list:
                if not isinstance(ent, Mapping):
                    continue
                if not is_entity_linked(ent, gkb_to_wdid):
                    continue
                linked_count += 1

            per_doc_linked.append(linked_count)

        entity_agg = StatsCalculator._agg(per_doc_entities)
        linked_agg = StatsCalculator._agg(per_doc_linked)
        return CorpusEntitySummary(
            corpus_name=corpus_name,
            entities_mean=entity_agg.mean,
            linked_mean=linked_agg.mean,
            linked_std=linked_agg.std,
            linked_relevance_mean=relevance_stats.mean,
            linked_relevance_std=relevance_stats.std,
        )


# ---------------------------------------------------------------------------
# Macro-averaged globals
# ---------------------------------------------------------------------------


def compute_macro_globals(
    corpus_rows: dict[str, list[dict]],
    entities_by_article: Mapping[str, list[dict]],
    gkb_to_wdid: Mapping[str, str],
) -> tuple[CategoryStats, EntityStats, Counter, int]:
    """
    Compute macro-averaged global stats (equal weight per corpus).

    :return: (macro_cat_stats, macro_entity_stats, macro_type_counts, n_corpora).
    """
    n = len(corpus_rows)
    if n == 0:
        return CategoryStats(), EntityStats(), Counter(), 0

    per_corpus_entity_stats: list[EntityStats] = []
    macro_cat_counter: Counter = Counter()
    macro_type_counter: Counter = Counter()

    for rows in corpus_rows.values():
        article_ids = extract_article_ids(rows)

        cat_stats = StatsCalculator.compute_category_stats(rows=rows)
        if cat_stats.num_docs > 0:
            for cat_id, count in cat_stats.cat_doc_counts.items():
                macro_cat_counter[cat_id] += count / cat_stats.num_docs

        entity_stats, _, _ = StatsCalculator.compute_entity_stats_from_mapping(
            article_ids=article_ids,
            entities_by_article=entities_by_article,
            gkb_to_wdid=gkb_to_wdid,
        )
        per_corpus_entity_stats.append(entity_stats)

        type_counts, total_linked, _ = StatsCalculator.compute_entity_type_distribution(
            article_ids=article_ids,
            entities_by_article=entities_by_article,
            gkb_to_wdid=gkb_to_wdid,
        )
        if total_linked > 0:
            for type_name, count in type_counts.items():
                macro_type_counter[type_name] += count / total_linked

    macro_cat_stats = CategoryStats(num_docs=n, cat_doc_counts=macro_cat_counter)

    macro_entity_stats = EntityStats(
        entities_per_doc=AggStats(mean=sum(s.entities_per_doc.mean for s in per_corpus_entity_stats) / n),
        entities_with_wdid=AggStats(mean=sum(s.entities_with_wdid.mean for s in per_corpus_entity_stats) / n),
        entities_without_wdid=AggStats(mean=sum(s.entities_without_wdid.mean for s in per_corpus_entity_stats) / n),
    )

    return macro_cat_stats, macro_entity_stats, macro_type_counter, n


# ---------------------------------------------------------------------------
# LatexTableBuilder
# ---------------------------------------------------------------------------


class LatexTableBuilder:
    """Builds LaTeX tabular snippets from computed statistics."""

    def __init__(self, iptc: IptcTopics, *, diff_mode: str = "micro", scope_label: str = "whole corpus"):
        self._iptc = iptc
        self._diff_mode = diff_mode
        self._scope_label = scope_label

    # -- formatting helpers --------------------------------------------------

    @staticmethod
    def format_percentage(part: int, total: int) -> str:
        if total <= 0:
            return "0.0\\%"
        return f"{(part / total) * 100.0:.1f}\\%"

    @staticmethod
    def latex_escape_name(name: str) -> str:
        """Escape underscores for LaTeX (e.g. cs_mafra_iptc -> cs\\_mafra\\_iptc)."""
        return name.replace("_", "\\_")

    def _resolve_iptc_label(self, cat_id: str) -> str:
        """Resolve IPTC category ID to display label."""
        try:
            cat_obj = self._iptc.getCategory(cat_id)
            name = str(cat_obj.getLongName())
            suffix = f"({cat_id})"
            if name.rstrip().endswith(suffix):
                return name
            return f"{name} ({cat_id})"
        except Exception:
            return cat_id

    def _diff_label(self) -> str:
        """Describe baseline used for Diff columns."""
        if self._diff_mode == "macro":
            return "macro average over all corpora"
        return "micro average over all corpora"

    def _scope_suffix(self) -> str:
        """Standard suffix stating whether stats use train-only or full corpora."""
        return f"Statistics computed from the {self._scope_label}."

    def scope_suffix(self) -> str:
        """Public accessor for standard split-scope caption suffix."""
        return self._scope_suffix()

    @staticmethod
    def _parse_int_cell(value: object) -> int:
        """Parse integer from table cell text."""
        text = str(value).strip().replace(",", "")
        return int(text) if text else 0

    @staticmethod
    def _parse_float_cell(value: object) -> float:
        """Parse leading float from table cell text."""
        text = str(value).strip().replace(",", "")
        if not text:
            return 0.0
        if "$\\pm$" in text:
            text = text.split("$\\pm$", maxsplit=1)[0].strip()
        return float(text)

    @staticmethod
    def _parse_mean_std_cell(value: object) -> tuple[float, float]:
        """Parse ``mean ± std`` from table cell text."""
        text = str(value).strip().replace(",", "")
        if not text:
            return 0.0, 0.0
        if "$\\pm$" not in text:
            return float(text), 0.0
        mean_str, std_str = text.split("$\\pm$", maxsplit=1)
        return float(mean_str.strip()), float(std_str.strip())

    # -- low-level renderer --------------------------------------------------

    @staticmethod
    def df_to_latex_tabular(
        df: pd.DataFrame,
        column_format: str,
        caption: str | None = None,
        label: str | None = None,
        hline_edges: bool = True,
        resizebox: bool = False,
    ) -> str:
        """Render DataFrame as LaTeX tabular environment."""
        lines: list[str] = []
        if caption or label:
            lines.append("\\begin{table}[htbp]")
        if resizebox:
            lines.append("\\resizebox{\\textwidth}{!}{")
        lines.append("\\begin{tabular}{" + column_format + "}")
        header_cells = [str(col) for col in df.columns]
        lines.append(" & ".join(header_cells) + " \\\\")
        lines.append("\\hline")
        for _, row in df.iterrows():
            cells = [str(value) for value in row.to_list()]
            lines.append(" & ".join(cells) + " \\\\")
        if hline_edges:
            lines.append("\\hline")
        lines.append("\\end{tabular}")
        if resizebox:
            lines.append("}")
        if caption:
            lines.append(f"\\caption{{{caption}}}")
        if label:
            lines.append(f"\\label{{{label}}}")
        if caption or label:
            lines.append("\\end{table}")
        return "\n".join(lines)

    # -- table builders ------------------------------------------------------

    def format_basic_row(
        self,
        corpus_name: str,
        length_stats: ArticleLengthStats,
        cat_stats: CategoryStats,
    ) -> dict:
        """Build a basic-statistics row dict used in per-corpus basic tables."""
        esc = self.latex_escape_name
        ls = length_stats
        return {
            "Corpus": esc(corpus_name),
            "Articles": f"{cat_stats.num_docs:,}",
            "Avg title words": f"{ls.title_len.mean:.0f} $\\pm$ {ls.title_len.std:.0f}",
            "Avg lead words": f"{ls.lead_len.mean:.0f} $\\pm$ {ls.lead_len.std:.0f}",
            "Avg text words": f"{ls.text_len.mean:.0f} $\\pm$ {ls.text_len.std:.0f}",
            "Avg article words": f"{ls.article_len.mean:.0f} $\\pm$ {ls.article_len.std:.0f}",
        }

    def format_corpora_overview_row(
        self,
        corpus_name: str,
        length_stats: ArticleLengthStats,
        cat_stats: CategoryStats,
        entity_stats: EntityStats,
    ) -> dict:
        """Build a row for the multi-corpus overview table (lengths + entities per article)."""
        esc = self.latex_escape_name
        ls = length_stats
        ent = entity_stats
        pm = "$\\pm$"
        return {
            "Corpus": esc(corpus_name),
            "Articles": f"{cat_stats.num_docs:,}",
            "Title": f"{ls.title_len.mean:.0f}",
            "Lead": f"{ls.lead_len.mean:.0f}",
            "Text": f"{ls.text_len.mean:.0f}",
            "Article $\\pm$ Std.": f"{ls.article_len.mean:.0f} {pm} {ls.article_len.std:.0f}",
            "entities": f"{ent.entities_per_doc.mean:.1f}",
            "wdId entities": f"{ent.entities_with_wdid.mean:.1f} {pm} {ent.entities_with_wdid.std:.1f}",
        }

    def build_basic_table(
        self,
        corpus_name: str,
        length_stats: ArticleLengthStats,
        cat_stats: CategoryStats,
    ) -> RenderedTable:
        """One-row table summarizing basic corpus statistics."""
        data = [self.format_basic_row(corpus_name=corpus_name, length_stats=length_stats, cat_stats=cat_stats)]
        df = pd.DataFrame(data)
        return RenderedTable(
            file_stem=f"{corpus_name}_basic",
            latex=self.df_to_latex_tabular(
                df=df,
                column_format="lrrrrr",
                caption=f"Basic statistics for {self.latex_escape_name(corpus_name)}. {self._scope_suffix()}",
                label=f"tab:corpus-basic-{corpus_name}",
                resizebox=True,
            ),
            df=df,
        )

    def build_entity_stats_table(
        self,
        entity_stats: EntityStats,
        corpus_name: str,
        global_entity_stats: EntityStats | None = None,
    ) -> RenderedTable:
        """Table with entity statistics (mean, std, min, max); no top/bottom hline.

        :param global_entity_stats: all-corpora entity stats; when provided, a Diff column is added.
        """
        metrics = [
            (
                "Entities",
                entity_stats.entities_per_doc,
                global_entity_stats.entities_per_doc if global_entity_stats else None,
            ),
            (
                "Entities with wdId",
                entity_stats.entities_with_wdid,
                global_entity_stats.entities_with_wdid if global_entity_stats else None,
            ),
            (
                "Entities without wdId",
                entity_stats.entities_without_wdid,
                global_entity_stats.entities_without_wdid if global_entity_stats else None,
            ),
        ]
        data: list[dict] = []
        for name, agg, g_agg in metrics:
            row_dict: dict = {"Metric": name, "Mean": f"{agg.mean:.1f}"}
            if global_entity_stats is not None:
                diff = agg.mean - g_agg.mean
                sign = "+" if diff >= 0 else ""
                row_dict["Diff"] = f"{sign}{diff:.1f}"
            row_dict.update({"Std": f"{agg.std:.1f}", "Min": f"{agg.min:,}", "Max": f"{agg.max:,}"})
            data.append(row_dict)

        no_ent_row: dict = {
            "Metric": "Articles without entities",
            "Mean": f"{entity_stats.articles_without_entities:,}",
        }
        if global_entity_stats is not None:
            no_ent_row["Diff"] = "---"
        no_ent_row.update({"Std": "---", "Min": "---", "Max": "---"})
        data.append(no_ent_row)

        df = pd.DataFrame(data)
        col_fmt = "lrrrrr" if global_entity_stats is not None else "lrrrr"
        return RenderedTable(
            file_stem=f"{corpus_name}_entity_stats",
            latex=self.df_to_latex_tabular(
                df=df,
                column_format=col_fmt,
                caption=(
                    f"Entity statistics and difference from {self._diff_label()} "
                    f"for {self.latex_escape_name(corpus_name)}. {self._scope_suffix()}"
                ),
                label=f"tab:entity-stats-{corpus_name}",
                hline_edges=False,
            ),
            df=df,
        )

    def build_category_top_table(
        self,
        corpus_name: str,
        cat_stats: CategoryStats,
        top_n: int = 20,
        caption: str | None = None,
        label: str | None = None,
        global_cat_stats: CategoryStats | None = None,
    ) -> RenderedTable:
        """
        Top-N categories by percentage of documents.

        :param caption: custom caption; defaults to a per-corpus description.
        :param label: custom LaTeX label; defaults to ``tab:cats-top-{corpus_name}``.
        :param global_cat_stats: all-corpora category stats; when provided, a diff column is added.
        """
        total_docs = cat_stats.num_docs
        items = cat_stats.cat_doc_counts.most_common(top_n)
        rows = []
        for rank, (cat_id, doc_count) in enumerate(items, start=1):
            corpus_pct = (doc_count / total_docs) * 100.0 if total_docs > 0 else 0.0
            row_dict: dict = {
                "Rank": rank,
                "Category": self._resolve_iptc_label(cat_id),
                "Docs": f"{doc_count:,}",
                "\\%": self.format_percentage(doc_count, total_docs),
            }
            if global_cat_stats is not None:
                global_total = global_cat_stats.num_docs
                global_count = global_cat_stats.cat_doc_counts.get(cat_id, 0)
                global_pct = (global_count / global_total) * 100.0 if global_total > 0 else 0.0
                diff = corpus_pct - global_pct
                sign = "+" if diff >= 0 else ""
                row_dict["Diff"] = f"{sign}{diff:.1f}\\%"
            rows.append(row_dict)

        df = pd.DataFrame(rows)
        name_tex = self.latex_escape_name(corpus_name)
        final_caption = (
            caption
            if caption is not None
            else (
                f"Top {top_n} IPTC categories by document coverage and difference from "
                f"{self._diff_label()} for {name_tex}. {self._scope_suffix()}"
            )
        )
        final_label = label if label is not None else f"tab:cats-top-{corpus_name}"
        col_fmt = "rp{0.60\\textwidth}rrr" if global_cat_stats is not None else "rp{0.60\\textwidth}rr"
        return RenderedTable(
            file_stem=f"{corpus_name}_categories_top",
            latex=self.df_to_latex_tabular(
                df=df,
                column_format=col_fmt,
                caption=final_caption,
                label=final_label,
                resizebox=True,
            ),
            df=df,
        )

    def build_category_count_table(
        self,
        corpus_name: str,
        cat_stats: CategoryStats,
        caption: str | None = None,
        label: str | None = None,
    ) -> RenderedTable:
        """Table with all IPTC categories sorted by document count."""
        rows: list[dict] = []
        total_docs = cat_stats.num_docs
        for cat_id, doc_count in cat_stats.cat_doc_counts.most_common():
            rows.append(
                {
                    "Category": self._resolve_iptc_label(cat_id),
                    "Docs": f"{doc_count:,}",
                    "\\%": self.format_percentage(doc_count, total_docs),
                }
            )

        df = pd.DataFrame(rows)
        final_caption = (
            caption
            if caption is not None
            else f"IPTC categories sorted by document count for {self.latex_escape_name(corpus_name)}. {self._scope_suffix()}"
        )
        final_label = label if label is not None else f"tab:cats-count-{corpus_name}"
        return RenderedTable(
            file_stem=f"{corpus_name}_categories_count",
            latex=self.df_to_latex_tabular(
                df=df,
                column_format="p{0.70\\textwidth}rr",
                caption=final_caption,
                label=final_label,
                resizebox=True,
            ),
            df=df,
        )

    def build_entity_type_table(
        self,
        corpus_name: str,
        type_counts: Counter[str],
        total_linked: int,
        avg_relevance_by_type: Mapping[str, float] | None = None,
        global_type_counts: Counter[str] | None = None,
        global_total_linked: int = 0,
    ) -> RenderedTable:
        """Table with distribution of entity types among linked entities.

        :param global_type_counts: all-corpora type counts; when provided, a Diff column is added.
        :param global_total_linked: all-corpora total linked count.
        """
        rows_out: list[dict] = []
        for ent_type, count in type_counts.most_common():
            corpus_pct = (count / total_linked) * 100.0 if total_linked > 0 else 0.0
            row_dict: dict = {
                "Type": ent_type,
                "Entities with wdId": f"{count:,}",
                "\\% ": self.format_percentage(count, total_linked),
                "Avg relevance": f"{(avg_relevance_by_type or {}).get(ent_type, 0.0):.1f}",
            }
            if global_type_counts is not None:
                global_count = global_type_counts.get(ent_type, 0)
                global_pct = (global_count / global_total_linked) * 100.0 if global_total_linked > 0 else 0.0
                diff = corpus_pct - global_pct
                sign = "+" if diff >= 0 else ""
                row_dict["Diff"] = f"{sign}{diff:.1f}\\%"
            rows_out.append(row_dict)

        if not rows_out:
            return RenderedTable(
                file_stem=f"{corpus_name}_entity_types",
                latex=f"% No wdId entities with types for {corpus_name}.\n",
                df=pd.DataFrame(columns=["Type", "Entities with wdId", "\\% ", "Avg relevance"]),
            )

        df = pd.DataFrame(rows_out)
        name_tex = self.latex_escape_name(corpus_name)
        col_fmt = "lrrrr" if global_type_counts is not None else "lrrr"
        return RenderedTable(
            file_stem=f"{corpus_name}_entity_types",
            latex=self.df_to_latex_tabular(
                df=df,
                column_format=col_fmt,
                caption=(
                    "Distribution of linked-entity types (wdId), including average relevance, "
                    f"and difference from {self._diff_label()} for {name_tex}. {self._scope_suffix()}"
                ),
                label=f"tab:entity-types-{corpus_name}",
            ),
            df=df,
        )

    def build_summary_basic_table(self, basic_rows: list[dict]) -> RenderedTable:
        """One table with one row per corpus (corpora overview: lengths and entity averages)."""
        if not basic_rows:
            return RenderedTable(
                file_stem="all_corpora_overview",
                latex="% No basic rows to summarize.\n",
                df=pd.DataFrame(),
            )
        df = pd.DataFrame(basic_rows)

        articles = [self._parse_int_cell(v) for v in df["Articles"]]
        title_means = [self._parse_float_cell(v) for v in df["Title"]]
        lead_means = [self._parse_float_cell(v) for v in df["Lead"]]
        text_means = [self._parse_float_cell(v) for v in df["Text"]]
        article_stats = [self._parse_mean_std_cell(v) for v in df["Article $\\pm$ Std."]]
        entities_means = [self._parse_float_cell(v) for v in df["entities"]]
        wdid_stats = [self._parse_mean_std_cell(v) for v in df["wdId entities"]]

        total_articles = sum(articles)
        n_rows = len(df)
        if n_rows > 0:
            macro_row = {
                "Corpus": "Mean (macro)",
                "Articles": "---",
                "Title": f"{(sum(title_means) / n_rows):.0f}",
                "Lead": f"{(sum(lead_means) / n_rows):.0f}",
                "Text": f"{(sum(text_means) / n_rows):.0f}",
                "Article $\\pm$ Std.": (
                    f"{(sum(m for m, _ in article_stats) / n_rows):.0f} "
                    f"$\\pm$ {(sum(s for _, s in article_stats) / n_rows):.0f}"
                ),
                "entities": f"{(sum(entities_means) / n_rows):.1f}",
                "wdId entities": (
                    f"{(sum(m for m, _ in wdid_stats) / n_rows):.1f} "
                    f"$\\pm$ {(sum(s for _, s in wdid_stats) / n_rows):.1f}"
                ),
            }
            if total_articles > 0:
                micro_row = {
                    "Corpus": "Mean (micro)",
                    "Articles": "---",
                    "Title": f"{(sum(v * w for v, w in zip(title_means, articles)) / total_articles):.0f}",
                    "Lead": f"{(sum(v * w for v, w in zip(lead_means, articles)) / total_articles):.0f}",
                    "Text": f"{(sum(v * w for v, w in zip(text_means, articles)) / total_articles):.0f}",
                    "Article $\\pm$ Std.": (
                        f"{(sum(m * w for (m, _), w in zip(article_stats, articles)) / total_articles):.0f} "
                        f"$\\pm$ {(sum(s * w for (_, s), w in zip(article_stats, articles)) / total_articles):.0f}"
                    ),
                    "entities": f"{(sum(v * w for v, w in zip(entities_means, articles)) / total_articles):.1f}",
                    "wdId entities": (
                        f"{(sum(m * w for (m, _), w in zip(wdid_stats, articles)) / total_articles):.1f} "
                        f"$\\pm$ {(sum(s * w for (_, s), w in zip(wdid_stats, articles)) / total_articles):.1f}"
                    ),
                }
            else:
                micro_row = {
                    "Corpus": "Mean (micro)",
                    "Articles": "---",
                    "Title": "0",
                    "Lead": "0",
                    "Text": "0",
                    "Article $\\pm$ Std.": "0 $\\pm$ 0",
                    "entities": "0.0",
                    "wdId entities": "0.0 $\\pm$ 0.0",
                }
            total_row = {
                "Corpus": "Total",
                "Articles": f"{total_articles:,}",
                "Title": "---",
                "Lead": "---",
                "Text": "---",
                "Article $\\pm$ Std.": "---",
                "entities": "---",
                "wdId entities": "---",
            }
            df = pd.concat([df, pd.DataFrame([macro_row, micro_row, total_row])], ignore_index=True)

        return RenderedTable(
            file_stem="all_corpora_overview",
            latex=self.df_to_latex_tabular(
                df=df,
                column_format="lrrrrrrr",
                caption=(
                    "Per-corpus means for article lengths (in words) and entity density. "
                    f"{self._scope_suffix()}"
                ),
                label="tab:corpora_overview",
                resizebox=True,
            ),
            df=df,
        )

    def build_corpus_entity_summary_table(self, rows: list[CorpusEntitySummary]) -> RenderedTable:
        """One table with per-corpus entity means and linked relevance summary."""
        if not rows:
            return RenderedTable(
                file_stem="all_corpora_entity_summary",
                latex="% No entity summary rows to render.\n",
                df=pd.DataFrame(),
            )

        pm = "$\\pm$"
        data = [
            {
                "Corpus": self.latex_escape_name(row.corpus_name),
                "Entities": f"{row.entities_mean:.1f}",
                "Linked": f"{row.linked_mean:.1f} {pm} {row.linked_std:.1f}",
                "Relevance": f"{row.linked_relevance_mean:.1f} {pm} {row.linked_relevance_std:.1f}",
            }
            for row in rows
        ]
        df = pd.DataFrame(data)
        return RenderedTable(
            file_stem="all_corpora_entity_summary",
            latex=self.df_to_latex_tabular(
                df=df,
                column_format="lrrr",
                caption=(
                    "Per-corpus entity statistics per article: total entities, linked entities, and linked-entity "
                    f"relevance (mean $\\pm$ std). {self._scope_suffix()}"
                ),
                label="tab:corpora_entity_summary",
                resizebox=True,
            ),
            df=df,
        )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def describe_corpus(
    corpus_name: str,
    rows: list[dict],
    article_ids: set[str],
    builder: LatexTableBuilder,
    entities_by_article: Mapping[str, list[dict]],
    gkb_to_wdid: Mapping[str, str],
    date_stats: Mapping[str, dict[str, int]] | None,
    output_dir: Path | None = None,
    global_cat_stats: CategoryStats | None = None,
    global_entity_stats: EntityStats | None = None,
    global_type_counts: Counter[str] | None = None,
    global_total_linked: int = 0,
    *,
    train_only: bool = False,
) -> tuple[str, dict, dict]:
    """
    Compute statistics for a single corpus.

    :return: LaTeX snippet, basic row for per-corpus basic table, overview row for corpora overview table.
    """
    length_stats = StatsCalculator.compute_article_length_stats(rows=rows)
    cat_stats = StatsCalculator.compute_category_stats(rows=rows)
    entity_stats, missing_entity_ids, empty_entity_ids = StatsCalculator.compute_entity_stats_from_mapping(
        article_ids=article_ids,
        entities_by_article=entities_by_article,
        gkb_to_wdid=gkb_to_wdid,
    )

    if missing_entity_ids or empty_entity_ids:
        LOG.info(
            f"{corpus_name}: {len(missing_entity_ids)} articles not in entity mapping, "
            f"{len(empty_entity_ids)} articles with zero entities"
        )
        if output_dir is not None:
            rows_by_id = {(r.get("id") or "").strip(): r for r in rows}
            missing_path = output_dir / f"{corpus_name}{MISSING_ENTITIES_SUFFIX}"
            header = "id\treason\tword_count\tcats\ttitle\tlead\ttext"
            lines: list[str] = [header]
            for doc_id, reason in sorted(
                [(did, "not_in_mapping") for did in missing_entity_ids]
                + [(did, "empty_entities") for did in empty_entity_ids]
            ):
                row = rows_by_id.get(doc_id, {})
                title = str(row.get("title") or "")
                lead = str(row.get("lead") or "")
                text = str(row.get("text") or "")
                wc = len(f"{title} {lead} {text}".strip().split())
                cats = str(row.get("cats") or "")
                lines.append(f"{doc_id}\t{reason}\t{wc}\t{cats}\t{title}\t{lead}\t{text}")
            missing_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            LOG.info(f"  Written to {missing_path}")

    # stats.md totals are usually full split sets; do not apply when train_only so doc counts match
    # ``rows`` / ``article_ids`` and stay aligned with entity stats (also keyed by those IDs).
    if date_stats is not None and not train_only:
        corpus_date = date_stats.get(corpus_name)
        if corpus_date is not None:
            total = corpus_date["total"]
            missing = corpus_date["missing"]
            cat_stats = replace(cat_stats, num_docs=total, docs_with_date=total - missing, docs_without_date=missing)

    type_counts, total_linked, avg_relevance_by_type = StatsCalculator.compute_entity_type_distribution(
        article_ids=article_ids,
        entities_by_article=entities_by_article,
        gkb_to_wdid=gkb_to_wdid,
    )

    basic_table = builder.build_basic_table(corpus_name=corpus_name, length_stats=length_stats, cat_stats=cat_stats)
    categories_table = builder.build_category_top_table(
        corpus_name=corpus_name,
        cat_stats=cat_stats,
        top_n=20,
        global_cat_stats=global_cat_stats,
    )
    entity_stats_table = builder.build_entity_stats_table(
        entity_stats=entity_stats,
        corpus_name=corpus_name,
        global_entity_stats=global_entity_stats,
    )
    entity_types_table = builder.build_entity_type_table(
        corpus_name=corpus_name,
        type_counts=type_counts,
        total_linked=total_linked,
        avg_relevance_by_type=avg_relevance_by_type,
        global_type_counts=global_type_counts,
        global_total_linked=global_total_linked,
    )

    if output_dir is not None:
        for table in (basic_table, categories_table, entity_stats_table, entity_types_table):
            write_csv_table(output_dir=output_dir, file_stem=table.file_stem, df=table.df)

    parts: list[str] = []
    parts.append(f"% Corpus: {corpus_name}")
    parts.append(basic_table.latex)
    parts.append("")
    parts.append(categories_table.latex)
    parts.append("")
    parts.append(entity_stats_table.latex)
    parts.append("")
    parts.append(entity_types_table.latex)
    parts.append("")

    basic_row = builder.format_basic_row(corpus_name=corpus_name, length_stats=length_stats, cat_stats=cat_stats)
    overview_row = builder.format_corpora_overview_row(
        corpus_name=corpus_name,
        length_stats=length_stats,
        cat_stats=cat_stats,
        entity_stats=entity_stats,
    )
    return "\n".join(parts), basic_row, overview_row


def build_all_corpora_aggregated_tables(
    all_rows: list[dict],
    all_article_ids: set[str],
    builder: LatexTableBuilder,
    entities_by_article: Mapping[str, list[dict]],
    gkb_to_wdid: Mapping[str, str],
    output_dir: Path,
) -> str:
    """Build tables for the whole dataset (all corpora combined)."""
    if not all_rows:
        return "% No data for aggregated tables.\n"

    cat_stats = StatsCalculator.compute_category_stats(rows=all_rows)
    entity_stats, _, _ = StatsCalculator.compute_entity_stats_from_mapping(
        article_ids=all_article_ids,
        entities_by_article=entities_by_article,
        gkb_to_wdid=gkb_to_wdid,
    )
    type_counts, total_linked, avg_relevance_by_type = StatsCalculator.compute_entity_type_distribution(
        article_ids=all_article_ids,
        entities_by_article=entities_by_article,
        gkb_to_wdid=gkb_to_wdid,
    )

    parts: list[str] = []
    parts.append(f"% Aggregated statistics over all corpora. {builder.scope_suffix()}")

    # data_basic = [builder.format_basic_row(corpus_name="All corpora", length_stats=length_stats, cat_stats=cat_stats)]
    # df_basic = pd.DataFrame(data_basic)
    # parts.append(
    #    LatexTableBuilder.df_to_latex_tabular(
    #        df=df_basic,
    #        column_format="lrrrrr",
    #        caption="Basic statistics for the whole dataset (all corpora combined).",
    #        label="tab:corpus-basic-all-aggregated",
    #        resizebox=True,
    #    )
    # )
    # parts.append("")

    aggregated_categories = builder.build_category_top_table(
        corpus_name="all-aggregated",
        cat_stats=cat_stats,
        top_n=20,
        caption=f"Top 20 IPTC categories by document coverage for all corpora. {builder.scope_suffix()}",
        label="tab:cats-top-all-aggregated",
    )
    parts.append(aggregated_categories.latex)
    parts.append("")

    aggregated_category_counts = builder.build_category_count_table(
        corpus_name="all-aggregated",
        cat_stats=cat_stats,
        caption=f"All IPTC categories sorted by document count for all corpora. {builder.scope_suffix()}",
        label="tab:cats-count-all-aggregated",
    )
    parts.append(aggregated_category_counts.latex)
    parts.append("")

    aggregated_entity_stats = builder.build_entity_stats_table(
        entity_stats=entity_stats,
        corpus_name="all",
    )
    parts.append(aggregated_entity_stats.latex)
    parts.append("")

    aggregated_entity_types = builder.build_entity_type_table(
        corpus_name="all_corpora",
        type_counts=type_counts,
        total_linked=total_linked,
        avg_relevance_by_type=avg_relevance_by_type,
    )
    parts.append(aggregated_entity_types.latex)
    parts.append("")

    line_plot_path = output_dir / "all_corpora_category_count_line.png"
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]

        sorted_counts = [count for _, count in cat_stats.cat_doc_counts.most_common()]
        if sorted_counts:
            fig, ax = plt.subplots(figsize=(12, 4))
            x_positions = list(range(1, len(sorted_counts) + 1))
            ax.plot(x_positions, sorted_counts, linewidth=1.5)
            plot_title = "Category counts in train set" if "train" in builder.scope_suffix().lower() else "Category counts"
            ax.set_title(plot_title)
            ax.set_xlabel("Categories sorted by document count")
            ax.set_ylabel("Count")
            ax.set_xlim(left=0, right=len(sorted_counts))
            tick_positions = [0] + list(range(100, len(sorted_counts) + 1, 100))
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([str(pos) for pos in tick_positions])
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(line_plot_path, dpi=600)
            plt.close(fig)

            parts.append("\\begin{figure}[htbp]")
            parts.append("\\centering")
            parts.append("\\includegraphics[width=\\textwidth]{all_corpora_category_count_line.png}")
            parts.append(
                f"\\caption{{Line plot of category document counts sorted in descending order. {builder.scope_suffix()}}}"
            )
            parts.append("\\label{fig:all-corpora-category-count-line}")
            parts.append("\\end{figure}")
        else:
            parts.append("% No category counts available for line plot.")
    except Exception as exc:
        LOG.warning(f"Failed to generate category count line plot: {exc}")
        parts.append("% Category count line plot could not be generated.")

    for table in (aggregated_categories, aggregated_category_counts, aggregated_entity_stats, aggregated_entity_types):
        write_csv_table(output_dir=output_dir, file_stem=table.file_stem, df=table.df)
    return "\n".join(parts)


def main() -> None:
    """Entry point for corpus description script."""
    argparser = argparse.ArgumentParser(
        description=(
            "Describe original corpora (aggregated over train/dev/test CSVs, or train-only with --train) "
            "and generate LaTeX tables with article, entity, and category statistics."
        )
    )
    argparser.add_argument(
        "origin_dir",
        type=str,
        help="Directory with all-corpora-*.csv files (e.g. data-preprocessing/origin-corpora)",
    )
    argparser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help=f"Output directory for LaTeX snippets (default: <origin_dir>/{OUTPUT_DIR_NAME})",
    )
    argparser.add_argument(
        "--macro-diff",
        action="store_true",
        help="Use macro-averaged baseline for Diff columns (equal weight per corpus instead of per article)",
    )
    argparser.add_argument(
        "--train",
        action="store_true",
        help=(
            "Only read all-corpora CSV files whose filename contains 'train'. "
            "Article counts, lengths, categories, and entity stats then use only those articles; "
            "stats.md date totals are not applied so counts stay aligned with the train CSV rows."
        ),
    )
    argparser.add_argument(
        "--article-entities-tsv",
        type=str,
        default=None,
        help=(
            "Path to article -> entities TSV (expects columns article_id/id and entities JSON). "
            f"Default: <origin_dir>/../{ARTICLE_ENTITIES_TSV}"
        ),
    )
    argparser.add_argument(
        "--gkb-wdid-mapping-tsv",
        type=str,
        default=None,
        help=(
            "Path to gkb_id -> wikidata_ids TSV mapping (pipe-separated wdIds allowed). "
            f"Default: <origin_dir>/{GKB_WDID_MAPPING_TSV}"
        ),
    )
    args = argparser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    origin_dir = Path(args.origin_dir)
    if not origin_dir.is_dir():
        raise SystemExit(f"Origin directory does not exist: {origin_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else origin_dir / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    LOG.info(f"Loading date statistics from {origin_dir / DATE_STATS_FILE}")
    date_stats = load_date_stats(stats_md_path=origin_dir / DATE_STATS_FILE)

    if args.train:
        LOG.info(
            "Train-only mode: loading rows only from CSV files with 'train' in the filename; "
            "entity statistics use the same article IDs as those rows"
        )
    LOG.info(f"Loading corpus rows from {origin_dir}")
    corpus_rows = load_corpus_rows(origin_dir=origin_dir, train_only=args.train)
    LOG.info(f"Found {len(corpus_rows)} corpora with {sum(len(r) for r in corpus_rows.values()):,} total articles")

    if args.article_entities_tsv:
        article2entities_tsv = Path(args.article_entities_tsv)
    else:
        article_candidates = [
            origin_dir / ARTICLE_ENTITIES_TSV,
            origin_dir / "articles_entities_dates.tsv",
            origin_dir.parent / ARTICLE_ENTITIES_TSV,
            origin_dir.parent / "articles_entities_dates.tsv",
        ]
        article2entities_tsv = pick_first_existing_path(article_candidates)

    if article2entities_tsv is not None and article2entities_tsv.is_file():
        LOG.info(f"Loading article entities from {article2entities_tsv}")
        entities_by_article = load_article_entities_tsv(article_entities_tsv=article2entities_tsv)
    else:
        if args.article_entities_tsv:
            LOG.warning(f"Article-to-entity TSV not found: {args.article_entities_tsv}")
        else:
            LOG.warning("Article-to-entity TSV not found in default locations; entity stats will be empty")
        entities_by_article = {}

    if args.gkb_wdid_mapping_tsv:
        gkb_mapping_tsv = Path(args.gkb_wdid_mapping_tsv)
    else:
        mapping_candidates = [
            origin_dir / GKB_WDID_MAPPING_TSV,
            origin_dir / "wdId_mapping.tsv",
            origin_dir / "wdid_mapping.tsv",
            origin_dir.parent / GKB_WDID_MAPPING_TSV,
            origin_dir.parent / "wdId_mapping.tsv",
            origin_dir.parent / "wdid_mapping.tsv",
        ]
        gkb_mapping_tsv = pick_first_existing_path(mapping_candidates)

    if gkb_mapping_tsv is not None and gkb_mapping_tsv.is_file():
        LOG.info(f"Loading gkb->wdId mapping from {gkb_mapping_tsv}")
        gkb_to_wdid = load_gkb_wdid_mapping_tsv(mapping_tsv=gkb_mapping_tsv)
    else:
        if args.gkb_wdid_mapping_tsv:
            LOG.warning(f"gkb->wdId TSV mapping not found: {args.gkb_wdid_mapping_tsv}")
        else:
            LOG.warning("gkb->wdId TSV mapping not found in default locations; linked stats may be zero")
        gkb_to_wdid = {}

    iptc = IptcTopics.load()
    diff_mode = "macro" if args.macro_diff else "micro"
    scope_label = "train split only" if args.train else "whole corpus"
    builder = LatexTableBuilder(iptc=iptc, diff_mode=diff_mode, scope_label=scope_label)

    all_rows = [row for rows in corpus_rows.values() for row in rows]
    all_article_ids: set[str] = set()
    for rows in corpus_rows.values():
        all_article_ids.update(extract_article_ids(rows))

    if args.macro_diff:
        LOG.info("Computing macro-averaged global stats for diff columns (equal weight per corpus)")
        global_cat_stats, global_entity_stats, global_type_counts, global_total_linked = compute_macro_globals(
            corpus_rows=corpus_rows,
            entities_by_article=entities_by_article,
            gkb_to_wdid=gkb_to_wdid,
        )
    else:
        LOG.info("Computing micro-averaged global stats for diff columns (equal weight per article)")
        global_cat_stats = StatsCalculator.compute_category_stats(rows=all_rows)
        global_entity_stats, _, _ = StatsCalculator.compute_entity_stats_from_mapping(
            article_ids=all_article_ids,
            entities_by_article=entities_by_article,
            gkb_to_wdid=gkb_to_wdid,
        )
        global_type_counts, global_total_linked, _ = StatsCalculator.compute_entity_type_distribution(
            article_ids=all_article_ids,
            entities_by_article=entities_by_article,
            gkb_to_wdid=gkb_to_wdid,
        )

    overview_rows_for_summary: list[dict] = []
    corpus_entity_summary_rows: list[CorpusEntitySummary] = []

    for corpus_name in sorted(corpus_rows):
        rows = corpus_rows[corpus_name]
        article_ids = extract_article_ids(rows)

        LOG.info(
            f"Describing corpus: {corpus_name} ({len(rows):,} articles, {len(article_ids):,} with entities lookup)"
        )
        snippet, _basic_row, overview_row = describe_corpus(
            corpus_name=corpus_name,
            rows=rows,
            article_ids=article_ids,
            builder=builder,
            entities_by_article=entities_by_article,
            gkb_to_wdid=gkb_to_wdid,
            date_stats=date_stats,
            output_dir=output_dir,
            global_cat_stats=global_cat_stats,
            global_entity_stats=global_entity_stats,
            global_type_counts=global_type_counts,
            global_total_linked=global_total_linked,
            train_only=args.train,
        )
        (output_dir / f"{corpus_name}{CORPUS_TABLES_SUFFIX}").write_text(snippet, encoding="utf-8")
        if overview_row:
            overview_rows_for_summary.append(overview_row)
        corpus_entity_summary_rows.append(
            StatsCalculator.compute_entity_summary(
                corpus_name=corpus_name,
                article_ids=article_ids,
                entities_by_article=entities_by_article,
                gkb_to_wdid=gkb_to_wdid,
            )
        )

    summary_basic = builder.build_summary_basic_table(basic_rows=overview_rows_for_summary)
    entity_summary = builder.build_corpus_entity_summary_table(rows=corpus_entity_summary_rows)
    (output_dir / OUTPUT_SUMMARY_FILE).write_text(
        "\n".join([summary_basic.latex, "", entity_summary.latex]),
        encoding="utf-8",
    )
    write_csv_table(output_dir=output_dir, file_stem=summary_basic.file_stem, df=summary_basic.df)
    write_csv_table(output_dir=output_dir, file_stem=entity_summary.file_stem, df=entity_summary.df)

    LOG.info("Building aggregated tables for all corpora")
    aggregated = build_all_corpora_aggregated_tables(
        all_rows=all_rows,
        all_article_ids=all_article_ids,
        builder=builder,
        entities_by_article=entities_by_article,
        gkb_to_wdid=gkb_to_wdid,
        output_dir=output_dir,
    )
    (output_dir / OUTPUT_AGGREGATED_FILE).write_text(aggregated, encoding="utf-8")
    LOG.info(f"All tables written to {output_dir}")


if __name__ == "__main__":
    main()
