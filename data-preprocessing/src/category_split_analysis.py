#!/usr/bin/env python3
"""
Build train/test IPTC category analysis tables, plots, and comparison CSV files.
"""

import argparse
import csv
import logging
from pathlib import Path
from typing import Iterable, Sequence, cast

import pandas as pd  # type: ignore[import-not-found]
from geneea.mediacats.iptc import IptcTopics  # type: ignore[import-not-found]

from describe_corpora import (
    ALL_CORPORA_FILES,
    LatexTableBuilder,
    StatsCalculator,
    write_csv_table,
)

LOG = logging.getLogger(__name__)
DEFAULT_OUTPUT_DIR_NAME = "category_split_analysis"
OUTPUT_TEX_FILE = "category_split_analysis.tex"


def iter_split_rows(origin_dir: Path, *, split_token: str) -> Iterable[dict]:
    """Yield rows from all-corpora files matching split token in filename."""
    token = split_token.lower()
    for name in ALL_CORPORA_FILES:
        if token not in name.lower():
            continue
        path = origin_dir / name
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as in_file:
            reader = csv.DictReader(in_file)
            if "id" not in (reader.fieldnames or []):
                continue
            yield from reader


def split_category_stats(origin_dir: Path, *, split_token: str):
    """Compute category stats for one split token."""
    rows = list(iter_split_rows(origin_dir=origin_dir, split_token=split_token))
    return StatsCalculator.compute_category_stats(rows=rows)


def build_plot(output_path: Path, counts: Sequence[int], *, title: str) -> None:
    """Build category-rank line plot with sparse x labels."""
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]

    fig, ax = plt.subplots(figsize=(12, 4))
    x_positions = list(range(1, len(counts) + 1))
    ax.plot(x_positions, counts, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Categories sorted by document count")
    ax.set_ylabel("Count")
    ax.set_xlim(left=0, right=len(counts))
    tick_positions = [0] + list(range(100, len(counts) + 1, 100))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(pos) for pos in tick_positions])
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=600)
    plt.close(fig)


def category_rows_for(
    category_ids: Iterable[str],
    *,
    train_counts: dict[str, int],
    test_counts: dict[str, int],
    order_key: str,
    builder: LatexTableBuilder,
) -> list[dict]:
    """Build output rows for category comparison CSV files."""
    rows: list[dict[str, str | int]] = []
    for cat_id in category_ids:
        rows.append(
            {
                "category": builder._resolve_iptc_label(cat_id),
                "test_count": test_counts.get(cat_id, 0),
                "train_count": train_counts.get(cat_id, 0),
            }
        )
    rows.sort(key=lambda row: cast(int, row[order_key]), reverse=True)
    return rows


def write_category_csv(output_dir: Path, filename: str, rows: list[dict]) -> None:
    """Write category comparison rows into CSV."""
    df = pd.DataFrame(rows, columns=["category", "test_count", "train_count"])
    df.to_csv(output_dir / filename, index=False)


def main() -> None:
    """Entry point for train/test category analysis."""
    argparser = argparse.ArgumentParser(
        description=(
            "Generate train/test full-category tables, category-rank plots, "
            "and train-vs-test category comparison CSV files."
        )
    )
    argparser.add_argument("origin_dir", type=str, help="Directory with all-corpora-*.csv files")
    argparser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help=f"Output directory (default: <origin_dir>/{DEFAULT_OUTPUT_DIR_NAME})",
    )
    args = argparser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    origin_dir = Path(args.origin_dir)
    if not origin_dir.is_dir():
        raise SystemExit(f"Origin directory does not exist: {origin_dir}")
    output_dir = Path(args.output_dir) if args.output_dir else origin_dir / DEFAULT_OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    iptc = IptcTopics.load()
    builder = LatexTableBuilder(iptc=iptc, diff_mode="micro", scope_label="selected split")

    LOG.info("Computing train and test category statistics")
    train_stats = split_category_stats(origin_dir=origin_dir, split_token="train")
    test_stats = split_category_stats(origin_dir=origin_dir, split_token="test")

    train_table = builder.build_category_count_table(
        corpus_name="train-set",
        cat_stats=train_stats,
        caption="All IPTC categories in train set, sorted by document count.",
        label="tab:cats-count-train-set",
    )
    test_table = builder.build_category_count_table(
        corpus_name="test-set",
        cat_stats=test_stats,
        caption="All IPTC categories in test set, sorted by document count.",
        label="tab:cats-count-test-set",
    )
    write_csv_table(output_dir=output_dir, file_stem=train_table.file_stem, df=train_table.df)
    write_csv_table(output_dir=output_dir, file_stem=test_table.file_stem, df=test_table.df)

    train_plot_file = "train_category_count_line.png"
    test_plot_file = "test_category_count_line.png"
    build_plot(
        output_path=output_dir / train_plot_file,
        counts=[count for _, count in train_stats.cat_doc_counts.most_common()],
        title="Category counts in train set",
    )
    build_plot(
        output_path=output_dir / test_plot_file,
        counts=[count for _, count in test_stats.cat_doc_counts.most_common()],
        title="Category counts in test set",
    )

    train_counts = dict(train_stats.cat_doc_counts)
    test_counts = dict(test_stats.cat_doc_counts)
    train_set = set(train_counts)
    test_set = set(test_counts)

    rows_in_test = category_rows_for(
        test_set, train_counts=train_counts, test_counts=test_counts, order_key="test_count", builder=builder
    )
    rows_test_not_train = category_rows_for(
        test_set - train_set, train_counts=train_counts, test_counts=test_counts, order_key="test_count", builder=builder
    )
    rows_train_not_test = category_rows_for(
        train_set - test_set,
        train_counts=train_counts,
        test_counts=test_counts,
        order_key="train_count",
        builder=builder,
    )
    rows_in_both = category_rows_for(
        train_set & test_set, train_counts=train_counts, test_counts=test_counts, order_key="train_count", builder=builder
    )
    rows_train_gt100_test_gt10 = category_rows_for(
        [cat for cat in (train_set & test_set) if train_counts.get(cat, 0) > 100 and test_counts.get(cat, 0) > 10],
        train_counts=train_counts,
        test_counts=test_counts,
        order_key="train_count",
        builder=builder,
    )
    rows_train_gt1000_test_gt100 = category_rows_for(
        [cat for cat in (train_set & test_set) if train_counts.get(cat, 0) > 1000 and test_counts.get(cat, 0) > 100],
        train_counts=train_counts,
        test_counts=test_counts,
        order_key="train_count",
        builder=builder,
    )

    write_category_csv(output_dir, "categories_in_test_ordered_by_test.csv", rows_in_test)
    write_category_csv(output_dir, "categories_in_test_not_in_train_ordered_by_test.csv", rows_test_not_train)
    write_category_csv(output_dir, "categories_in_train_not_in_test_ordered_by_train.csv", rows_train_not_test)
    write_category_csv(output_dir, "categories_in_train_and_test_ordered_by_train.csv", rows_in_both)
    write_category_csv(output_dir, "categories_train_gt100_test_gt10_ordered_by_train.csv", rows_train_gt100_test_gt10)
    write_category_csv(
        output_dir, "categories_train_gt1000_test_gt100_ordered_by_train.csv", rows_train_gt1000_test_gt100
    )

    summary_df = pd.DataFrame(
        [
            {"Metric": "Overall category count in train set", "Value": sum(train_counts.values())},
            {"Metric": "Overall category count in test set", "Value": sum(test_counts.values())},
            {"Metric": "Categories in train set", "Value": len(train_set)},
            {"Metric": "Categories in test set", "Value": len(test_set)},
            {"Metric": "Categories in train and test", "Value": len(train_set & test_set)},
            {"Metric": "Categories in train not in test", "Value": len(train_set - test_set)},
            {"Metric": "Categories in test not in train", "Value": len(test_set - train_set)},
            {"Metric": "Categories with train>100 and test>10", "Value": len(rows_train_gt100_test_gt10)},
            {"Metric": "Categories with train>1000 and test>100", "Value": len(rows_train_gt1000_test_gt100)},
        ]
    )
    summary_table_latex = LatexTableBuilder.df_to_latex_tabular(
        df=summary_df,
        column_format="lr",
        caption="Summary of train/test category overlap and threshold-based subsets.",
        label="tab:train-test-category-summary",
    )
    write_csv_table(output_dir=output_dir, file_stem="train_test_category_summary", df=summary_df)

    report_parts = [
        "% Train/test category analysis",
        train_table.latex,
        "",
        "\\begin{figure}[htbp]",
        "\\centering",
        f"\\includegraphics[width=\\textwidth]{{{train_plot_file}}}",
        "\\caption{Line plot of train-set category document counts sorted by descending count.}",
        "\\label{fig:train-category-count-line}",
        "\\end{figure}",
        "",
        test_table.latex,
        "",
        "\\begin{figure}[htbp]",
        "\\centering",
        f"\\includegraphics[width=\\textwidth]{{{test_plot_file}}}",
        "\\caption{Line plot of test-set category document counts sorted by descending count.}",
        "\\label{fig:test-category-count-line}",
        "\\end{figure}",
        "",
        summary_table_latex,
    ]
    (output_dir / OUTPUT_TEX_FILE).write_text("\n".join(report_parts), encoding="utf-8")

    LOG.info(f"Train/test category analysis written to {output_dir}")


if __name__ == "__main__":
    main()
