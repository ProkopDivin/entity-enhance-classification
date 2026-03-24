#!/usr/bin/env python3
"""
Describe original corpora and generate LaTeX tables for thesis.
Computes article, entity, category, and time statistics per corpus.
"""
import argparse
import csv
import json
import logging
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np  # type: ignore[import-not-found]
import pandas as pd  # type: ignore[import-not-found]

from geneea.mediacats.iptc import IptcTopics  # type: ignore[import-not-found]

from iptc_entity_pipeline.data_loading import load_article_entities  # type: ignore[import-not-found]

LOG = logging.getLogger(__name__)

ALL_CORPORA_FILES = ("all-corpora-train.csv", "all-corpora-dev.csv", "all-corpora-test.csv")

ONLY_IMPORTANT = False

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
    """Per-corpus article length statistics."""

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
class CorpusConfig:
    name: str
    split_files: Mapping[str, Path]


# ---------------------------------------------------------------------------
# Standalone utilities
# ---------------------------------------------------------------------------


def is_entity_linked(ent: Mapping) -> bool:
    """Check whether an entity is linked to a knowledge base (wdId, gkbId, or gkbID)."""
    return bool(ent.get("wdId") or ent.get("gkbId") or ent.get("gkbID"))


def extract_corpus_name(filename: str) -> str:
    """Extract corpus name from filename like 'cs_mafra_iptc.train_all-cats.csv'."""
    name = filename
    if name.endswith(".csv"):
        name = name[:-4]
    for suffix in (
        ".train_all-cats",
        ".dev_all-cats",
        ".test_all-cats",
        ".train_smallpp-cats",
        ".dev_smallpp-cats",
        ".test_smallpp-cats",
        ".train_medium-cats",
        ".dev_medium-cats",
        ".test_medium-cats",
        ".train",
        ".dev",
        ".test",
    ):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def parse_categories(cats_str: str | None) -> list[str]:
    """Parse categories from pipe-separated string."""
    if not cats_str or pd.isna(cats_str):
        return []
    return [cat.strip() for cat in str(cats_str).split("|") if cat.strip()]


def parse_entities(entities_str: str | None) -> list[dict]:
    """Parse entities from JSON array stored in CSV column."""
    if not entities_str or pd.isna(entities_str):
        return []
    try:
        if isinstance(entities_str, str):
            return json.loads(entities_str)
        if isinstance(entities_str, list):
            return entities_str
        return []
    except json.JSONDecodeError:
        return []


def safe_len(value: object) -> int:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0
    return len(str(value))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_corpus_configs(directory: Path) -> list[CorpusConfig]:
    """Discover per-corpus CSVs under origin-corpora and group train/dev/test."""
    corpus_map: dict[str, dict[str, Path]] = {}
    for csv_path in sorted(directory.glob("*.csv")):
        fname = csv_path.name.lower()
        if fname.startswith("all-corpora") or fname.startswith("entities_all-corpora"):
            continue

        split: Optional[str] = None
        if ".train" in fname or "_train" in fname:
            split = "train"
        elif ".dev" in fname or "_dev" in fname:
            split = "dev"
        elif ".test" in fname or "_test" in fname:
            split = "test"

        if split is None:
            continue

        corpus_name = extract_corpus_name(csv_path.name)
        corpus_map.setdefault(corpus_name, {})[split] = csv_path

    return [
        CorpusConfig(name=corpus_name, split_files=split_files)
        for corpus_name, split_files in sorted(corpus_map.items())
    ]


def iter_all_corpora_csv_rows(origin_dir: Path) -> Iterable[dict]:
    """
    Yield raw CSV rows from all-corpora CSV files.

    :param origin_dir: directory containing the all-corpora-*.csv files.
    """
    for name in ALL_CORPORA_FILES:
        path = origin_dir / name
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "id" not in (reader.fieldnames or []):
                continue
            yield from reader


def iter_all_corpus_rows(
    origin_dir: Path,
    entities_by_article: Optional[Mapping[str, list[dict]]],
) -> Iterable[dict]:
    """Yield all article rows from all-corpora-*.csv with entities attached."""
    for row in iter_all_corpora_csv_rows(origin_dir=origin_dir):
        doc = dict(row)
        doc_id = (doc.get("id") or "").strip()
        if entities_by_article and doc_id:
            doc["entities"] = entities_by_article.get(doc_id, [])
        yield doc


def load_corpus_article_ids(origin_dir: Path) -> dict[str, set[str]]:
    """
    Build mapping corpus_name -> set of article ids from all-corpora-*.csv.

    Uses metadata.corpusName so article ids match the format in article_2_entities.tsv.
    """
    result: dict[str, set[str]] = {}
    for row in iter_all_corpora_csv_rows(origin_dir=origin_dir):
        doc_id = (row.get("id") or "").strip()
        meta_str = row.get("metadata") or ""
        if not doc_id or not meta_str:
            LOG.warning(f"Skipping row with no id or metadata: {row}")
            continue
        try:
            meta = json.loads(meta_str)
        except json.JSONDecodeError:
            LOG.warning(f"Invalid JSON in metadata: {meta_str}")
            continue
        corpus_name = meta.get("corpusName") if isinstance(meta, dict) else None
        if not corpus_name:
            continue
        result.setdefault(corpus_name, set()).add(doc_id)
    return result


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


def iter_corpus_rows(articles_df: pd.DataFrame, entities_df: Optional[pd.DataFrame]) -> Iterable[dict]:
    """Yield merged per-article rows, optionally enriched with entities."""
    entities_by_id: dict[str, dict] = {}
    if entities_df is not None and "id" in entities_df.columns:
        for _, row in entities_df.iterrows():
            doc_id = str(row.get("id"))
            entities_by_id[doc_id] = row.to_dict()

    for _, row in articles_df.iterrows():
        doc = row.to_dict()
        doc_id = str(doc.get("id"))
        if entities_by_id:
            ent_row = entities_by_id.get(doc_id)
            if ent_row is not None and "entities" in ent_row and "entities" not in doc:
                doc["entities"] = ent_row["entities"]
        yield doc


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
    def compute_entity_stats(entities_per_article: Iterable[list[dict]]) -> EntityStats:
        """
        Compute entity-related stats from per-article entity lists.

        :param entities_per_article: iterable where each element is a list of entity dicts for one article.
        """
        counts_per_doc: list[int] = []
        counts_with_wdid: list[int] = []
        counts_without_wdid: list[int] = []

        for entities in entities_per_article:
            counts_per_doc.append(len(entities))
            with_wdid = sum(1 for ent in entities if isinstance(ent, Mapping) and is_entity_linked(ent))
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
    ) -> EntityStats:
        """Compute entity stats using article_id -> entities mapping."""
        return StatsCalculator.compute_entity_stats(
            entities_per_article=(entities_by_article.get(str(doc_id), []) for doc_id in article_ids)
        )

    @staticmethod
    def compute_article_stats(rows: Iterable[dict]) -> tuple[ArticleLengthStats, EntityStats]:
        """
        Compute article length and entity statistics in a single pass over document rows.

        :return: tuple of (article length stats, entity stats).
        """
        title_lengths: list[int] = []
        lead_lengths: list[int] = []
        text_lengths: list[int] = []
        article_lengths: list[int] = []
        entities_per_article: list[list[dict]] = []

        for doc in rows:
            title_len = safe_len(doc.get("title"))
            lead_len = safe_len(doc.get("lead"))
            text_len = safe_len(doc.get("text"))

            title_lengths.append(title_len)
            lead_lengths.append(lead_len)
            text_lengths.append(text_len)
            article_lengths.append(title_len + lead_len + text_len)

            entities = parse_entities(str(doc.get("entities"))) if "entities" in doc else []
            entities_per_article.append(entities)

        entity_stats = StatsCalculator.compute_entity_stats(entities_per_article=entities_per_article)
        agg = StatsCalculator._agg
        length_stats = ArticleLengthStats(
            num_docs=len(text_lengths),
            article_len=agg(article_lengths),
            title_len=agg(title_lengths),
            lead_len=agg(lead_lengths),
            text_len=agg(text_lengths),
        )
        return length_stats, entity_stats

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
    ) -> tuple[Counter[str], int]:
        """
        Count entity types among linked entities.

        :return: tuple of (type_counts Counter, total_linked count).
        """
        type_counts: Counter[str] = Counter()
        total_linked = 0

        for doc_id in article_ids:
            entities = entities_by_article.get(str(doc_id), [])
            for ent in entities:
                if not isinstance(ent, Mapping):
                    continue
                if not is_entity_linked(ent):
                    continue
                ent_type = str(ent.get("type") or "unknown")
                type_counts[ent_type] += 1
                total_linked += 1

        return type_counts, total_linked


# ---------------------------------------------------------------------------
# LatexTableBuilder
# ---------------------------------------------------------------------------


class LatexTableBuilder:
    """Builds LaTeX tabular snippets from computed statistics."""

    def __init__(self, iptc: IptcTopics):
        self._iptc = iptc

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

    # -- low-level renderer --------------------------------------------------

    @staticmethod
    def df_to_latex_tabular(
        df: pd.DataFrame,
        column_format: str,
        caption: Optional[str] = None,
        label: Optional[str] = None,
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
        if hline_edges:
            lines.append("\\hline")
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
        """Build a basic-statistics row dict used in corpus summary tables."""
        pct = self.format_percentage
        esc = self.latex_escape_name
        ls = length_stats
        output_dict = {
            "Corpus": esc(corpus_name),
            "Articles": f"{cat_stats.num_docs:,}",
            "Avg title len": f"{ls.title_len.mean:.0f} $\\pm$ {ls.title_len.std:.0f}",
            "Avg lead len": f"{ls.lead_len.mean:.0f} $\\pm$ {ls.lead_len.std:.0f}",
            "Avg text len": f"{ls.text_len.mean:.0f} $\\pm$ {ls.text_len.std:.0f}",
            "Avg article len": f"{ls.article_len.mean:.0f} $\\pm$ {ls.article_len.std:.0f}",
        }
        if ONLY_IMPORTANT:
            output_dict["With date"] = (
                f"{cat_stats.docs_with_date:,} ({pct(cat_stats.docs_with_date, cat_stats.num_docs)})"
            )
            output_dict["Without date"] = (
                f"{cat_stats.docs_without_date:,} ({pct(cat_stats.docs_without_date, cat_stats.num_docs)})"
            )
        return output_dict

    def build_basic_table(
        self,
        corpus_name: str,
        length_stats: ArticleLengthStats,
        cat_stats: CategoryStats,
    ) -> str:
        """One-row table summarizing basic corpus statistics."""
        data = [self.format_basic_row(corpus_name=corpus_name, length_stats=length_stats, cat_stats=cat_stats)]
        df = pd.DataFrame(data)
        return self.df_to_latex_tabular(
            df=df,
            column_format="lrrrrrr",
            caption="Basic statistics for the corpus.",
            label=f"tab:corpus-basic-{corpus_name}",
            resizebox=True,
        )

    def build_entity_stats_table(self, entity_stats: EntityStats, corpus_name: str) -> str:
        """Table with entity statistics (mean, std, min, max); no top/bottom hline."""
        epd = entity_stats.entities_per_doc
        eww = entity_stats.entities_with_wdid
        ewo = entity_stats.entities_without_wdid
        data = [
            {
                "Metric": "Entities",
                "Mean": f"{epd.mean:.2f}",
                "Std": f"{epd.std:.2f}",
                "Min": f"{epd.min:,}",
                "Max": f"{epd.max:,}",
            },
            {
                "Metric": "Entities with wdId",
                "Mean": f"{eww.mean:.2f}",
                "Std": f"{eww.std:.2f}",
                "Min": f"{eww.min:,}",
                "Max": f"{eww.max:,}",
            },
            {
                "Metric": "Entities without wdId",
                "Mean": f"{ewo.mean:.2f}",
                "Std": f"{ewo.std:.2f}",
                "Min": f"{ewo.min:,}",
                "Max": f"{ewo.max:,}",
            },
            {
                "Metric": "Articles without entities",
                "Mean": f"{entity_stats.articles_without_entities:,}",
                "Std": "---",
                "Min": "---",
                "Max": "---",
            },
        ]
        df = pd.DataFrame(data)
        return self.df_to_latex_tabular(
            df=df,
            column_format="lrrrr",
            caption=f"Entity statistics for {corpus_name}.",
            label=f"tab:entity-stats-{corpus_name}",
            hline_edges=False,
        )

    def build_category_top_table(
        self,
        corpus_name: str,
        cat_stats: CategoryStats,
        top_n: int = 20,
        caption: Optional[str] = None,
        label: Optional[str] = None,
    ) -> str:
        """
        Top-N categories by percentage of documents.

        :param caption: custom caption; defaults to a per-corpus description.
        :param label: custom LaTeX label; defaults to ``tab:cats-top-{corpus_name}``.
        """
        total_docs = cat_stats.num_docs
        items = cat_stats.cat_doc_counts.most_common(top_n)
        rows = []
        for rank, (cat_id, doc_count) in enumerate(items, start=1):
            rows.append(
                {
                    "Rank": rank,
                    "Category": self._resolve_iptc_label(cat_id),
                    "Docs": f"{doc_count:,}",
                    "\\% of docs": self.format_percentage(doc_count, total_docs),
                }
            )
        df = pd.DataFrame(rows)
        name_tex = self.latex_escape_name(corpus_name)
        final_caption = (
            caption if caption is not None else f"Top {top_n} IPTC categories by document coverage for {name_tex}."
        )
        final_label = label if label is not None else f"tab:cats-top-{corpus_name}"
        return self.df_to_latex_tabular(
            df=df,
            column_format="rp{0.55\\textwidth}rr",
            caption=final_caption,
            label=final_label,
        )

    def build_entity_type_table(
        self,
        corpus_name: str,
        type_counts: Counter[str],
        total_linked: int,
    ) -> str:
        """Table with distribution of entity types among linked entities."""
        rows_out: list[dict] = []
        for ent_type, count in type_counts.most_common():
            rows_out.append(
                {
                    "Type": ent_type,
                    "Entities with wdId": f"{count:,}",
                    "\\% of wdId entities": self.format_percentage(count, total_linked),
                }
            )

        if not rows_out:
            return f"% No wdId entities with types for {corpus_name}.\n"

        df = pd.DataFrame(rows_out)
        name_tex = self.latex_escape_name(corpus_name)
        return self.df_to_latex_tabular(
            df=df,
            column_format="lrr",
            caption=f"Distribution of types among linked entities (with wdId) for {name_tex}.",
            label=f"tab:entity-types-{corpus_name}",
        )

    def build_summary_basic_table(self, basic_rows: list[dict]) -> str:
        """One table with one row per corpus (basic statistics)."""
        if not basic_rows:
            return "% No basic rows to summarize.\n"
        df = pd.DataFrame(basic_rows)
        return self.df_to_latex_tabular(
            df=df,
            column_format="lrrrrrr",
            caption="Basic statistics for all corpora.",
            label="tab:corpus-basic-all-summary",
            resizebox=True,
        )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def describe_corpus(
    config: CorpusConfig,
    builder: LatexTableBuilder,
    entities_dir: Optional[Path],
    date_stats: Optional[Mapping[str, dict[str, int]]],
    entities_by_article: Optional[Mapping[str, list[dict]]],
    corpus_article_ids: Optional[Mapping[str, set[str]]] = None,
) -> tuple[str, dict]:
    """Compute statistics for a single corpus; return LaTeX snippet and basic row for summary table."""
    dfs: list[pd.DataFrame] = []
    for split_name in ("train", "dev", "test"):
        csv_path = config.split_files.get(split_name)
        if csv_path is not None:
            dfs.append(pd.read_csv(csv_path, encoding="utf-8"))
    if not dfs:
        return f"% No data for corpus {config.name}\n", {}
    articles_df = pd.concat(dfs, ignore_index=True)

    entities_df: Optional[pd.DataFrame] = None
    if entities_dir is not None:
        ent_path = entities_dir / "entities_all-corpora-train.csv"
        if ent_path.exists():
            entities_df = pd.read_csv(ent_path, encoding="utf-8")

    rows_for_stats = list(iter_corpus_rows(articles_df=articles_df, entities_df=entities_df))
    length_stats, entity_stats = StatsCalculator.compute_article_stats(rows=rows_for_stats)
    cat_stats = StatsCalculator.compute_category_stats(rows=rows_for_stats)

    if date_stats is not None:
        corpus_date = date_stats.get(config.name)
        if corpus_date is not None:
            total = corpus_date["total"]
            missing = corpus_date["missing"]
            cat_stats = replace(cat_stats, num_docs=total, docs_with_date=total - missing, docs_without_date=missing)

    parts: list[str] = []
    parts.append(f"% Corpus: {config.name}")
    parts.append(builder.build_basic_table(corpus_name=config.name, length_stats=length_stats, cat_stats=cat_stats))
    parts.append("")
    parts.append(builder.build_category_top_table(corpus_name=config.name, cat_stats=cat_stats, top_n=20))
    parts.append("")
    parts.append(builder.build_entity_stats_table(entity_stats=entity_stats, corpus_name=config.name))
    parts.append("")

    if entities_by_article is not None:
        ids_for_corpus = (corpus_article_ids or {}).get(config.name, set())
        type_counts, total_linked = StatsCalculator.compute_entity_type_distribution(
            article_ids=ids_for_corpus,
            entities_by_article=entities_by_article,
        )
        parts.append(
            builder.build_entity_type_table(
                corpus_name=config.name,
                type_counts=type_counts,
                total_linked=total_linked,
            )
        )
    parts.append("")

    basic_row = builder.format_basic_row(corpus_name=config.name, length_stats=length_stats, cat_stats=cat_stats)
    return "\n".join(parts), basic_row


def build_all_corpora_aggregated_tables(
    origin_dir: Path,
    builder: LatexTableBuilder,
    entities_by_article: Optional[Mapping[str, list[dict]]],
    corpus_article_ids: Mapping[str, set[str]],
) -> str:
    """Build tables for the whole dataset (all corpora combined)."""
    all_rows = list(iter_all_corpus_rows(origin_dir=origin_dir, entities_by_article=entities_by_article))
    if not all_rows:
        return "% No data for aggregated tables.\n"

    length_stats, entity_stats = StatsCalculator.compute_article_stats(rows=all_rows)
    cat_stats = StatsCalculator.compute_category_stats(rows=all_rows)

    all_article_ids: set[str] = set()
    for ids in corpus_article_ids.values():
        all_article_ids.update(ids)
    if entities_by_article and all_article_ids:
        entity_stats = StatsCalculator.compute_entity_stats_from_mapping(
            article_ids=all_article_ids,
            entities_by_article=entities_by_article,
        )

    parts: list[str] = []
    parts.append("% Aggregated statistics over all corpora (whole dataset)")

    data_basic = [builder.format_basic_row(corpus_name="All corpora", length_stats=length_stats, cat_stats=cat_stats)]
    df_basic = pd.DataFrame(data_basic)
    parts.append(
        LatexTableBuilder.df_to_latex_tabular(
            df=df_basic,
            column_format="lrrrrrr",
            caption="Basic statistics for the whole dataset (all corpora combined).",
            label="tab:corpus-basic-all-aggregated",
            resizebox=True,
        )
    )
    parts.append("")

    parts.append(
        builder.build_category_top_table(
            corpus_name="all-aggregated",
            cat_stats=cat_stats,
            top_n=20,
            caption="Top 20 IPTC categories by document coverage (whole dataset).",
            label="tab:cats-top-all-aggregated",
        )
    )
    parts.append("")

    parts.append(builder.build_entity_stats_table(entity_stats=entity_stats, corpus_name="all"))
    parts.append("")

    if entities_by_article:
        type_counts, total_linked = StatsCalculator.compute_entity_type_distribution(
            article_ids=all_article_ids,
            entities_by_article=entities_by_article,
        )
        parts.append(
            builder.build_entity_type_table(
                corpus_name="all corpora",
                type_counts=type_counts,
                total_linked=total_linked,
            )
        )
    return "\n".join(parts)


def main() -> None:
    """Entry point for corpus description script."""
    argparser = argparse.ArgumentParser(
        description=(
            "Describe original corpora (aggregated over train/dev/test) and generate LaTeX tables "
            "with article, entity, and category statistics."
        )
    )
    argparser.add_argument(
        "origin_dir",
        type=str,
        help="Directory with per-corpus CSVs (e.g. data-preprocessing/origin-corpora)",
    )
    argparser.add_argument(
        "-e",
        "--entities-dir",
        type=str,
        default=None,
        help="Optional directory with entities_all-corpora-*.csv (default: use origin_dir if present)",
    )
    argparser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for LaTeX snippets (default: <origin_dir>/thesis_corpora_tables)",
    )
    args = argparser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    origin_dir = Path(args.origin_dir)
    if not origin_dir.is_dir():
        raise SystemExit(f"Origin directory does not exist: {origin_dir}")

    entities_dir = Path(args.entities_dir) if args.entities_dir else origin_dir
    output_dir = Path(args.output_dir) if args.output_dir else origin_dir / "thesis_corpora_tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    LOG.info(f'Loading date statistics from {origin_dir / "stats.md"}')
    date_stats = load_date_stats(stats_md_path=origin_dir / "stats.md")

    LOG.info(f"Loading corpus article IDs from {origin_dir}")
    corpus_article_ids = load_corpus_article_ids(origin_dir=origin_dir)
    LOG.info(f"Found {len(corpus_article_ids)} corpora in all-corpora CSVs")

    entities_by_article: Optional[dict[str, list[dict]]] = None
    article2entities_tsv = origin_dir.parent / "article_2_entities.tsv"
    if article2entities_tsv.is_file():
        LOG.info(f"Loading article entities from {article2entities_tsv}")
        entities_by_article = load_article_entities(article_entities_tsv=str(article2entities_tsv))
    else:
        LOG.warning(f"Article-to-entity TSV not found: {article2entities_tsv}")

    iptc = IptcTopics.load()
    builder = LatexTableBuilder(iptc=iptc)

    corpus_configs = load_corpus_configs(directory=origin_dir)
    if not corpus_configs:
        raise SystemExit(f"No per-corpus CSVs found in {origin_dir}")
    LOG.info(f"Discovered {len(corpus_configs)} corpus configurations")

    basic_rows_for_summary: list[dict] = []
    for config in corpus_configs:
        LOG.info(f"Describing corpus: {config.name}")
        snippet, basic_row = describe_corpus(
            config=config,
            builder=builder,
            entities_dir=entities_dir,
            date_stats=date_stats,
            entities_by_article=entities_by_article,
            corpus_article_ids=corpus_article_ids,
        )
        out_path = output_dir / f"{config.name}_tables.tex"
        out_path.write_text(snippet, encoding="utf-8")
        if basic_row:
            basic_rows_for_summary.append(basic_row)

    summary_basic = builder.build_summary_basic_table(basic_rows=basic_rows_for_summary)
    (output_dir / "all_corpora_summary_basic.tex").write_text(summary_basic, encoding="utf-8")

    LOG.info("Building aggregated tables for all corpora")
    aggregated = build_all_corpora_aggregated_tables(
        origin_dir=origin_dir,
        builder=builder,
        entities_by_article=entities_by_article,
        corpus_article_ids=corpus_article_ids,
    )
    (output_dir / "all_corpora_aggregated.tex").write_text(aggregated, encoding="utf-8")
    LOG.info(f"All tables written to {output_dir}")


if __name__ == "__main__":
    main()
