#!/usr/bin/env python3
"""
Analyze dataset splits and generate statistics and visualizations for thesis.
Computes dataset distribution, category distribution, missing categories,
article statistics, and entity statistics.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Since this is a CLI script, print is acceptable for output
# For general-purpose tools, logging should be used instead

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10


def extract_dataset_name(filename: str) -> Optional[str]:
    """
    Extract dataset name from filename.

    :param filename: Filename (e.g., 'en_bbc_iptc.train_all-cats.csv')
    :return: Dataset name (e.g., 'en_bbc_iptc') or None
    """
    # Remove common suffixes
    name = filename.replace("_all-cats.csv", "").replace(".csv", "")
    for suffix in [".train", ".dev", ".test", "_train", "_dev", "_test"]:
        if suffix in name:
            name = name.split(suffix)[0]
            break
    return name if name else None


def load_csv_files(directory: Path) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Load all CSV files from directory, organized by split type and dataset.
    Prefers files starting with 'entities_' for entity analysis, but loads all CSV files.

    :param directory: Directory containing CSV files
    :return: Dictionary mapping split_type -> dataset_name -> DataFrame
    """
    data: dict[str, dict[str, pd.DataFrame]] = {"train": {}, "dev": {}, "test": {}}

    # Load all CSV files, but prefer entities_* files when available
    csv_files = sorted(directory.glob("*.csv"))

    for csv_file in csv_files:
        filename = csv_file.name.lower()

        # Determine split type
        if ".train" in filename or "_train" in filename:
            split_type = "train"
        elif ".dev" in filename or "_dev" in filename:
            split_type = "dev"
        elif ".test" in filename or "_test" in filename:
            split_type = "test"
        else:
            continue  # Skip files that don't match expected pattern

        # Extract dataset name (remove entities_ prefix if present)
        base_name = csv_file.name
        if base_name.startswith("entities_"):
            base_name = base_name[9:]  # Remove "entities_" prefix

        dataset_name = extract_dataset_name(base_name)
        if not dataset_name:
            continue

        # If we already have data for this split/dataset and current file is not entities_*,
        # skip it (prefer entities_* files)
        if dataset_name in data[split_type] and not csv_file.name.startswith("entities_"):
            continue

        try:
            df = pd.read_csv(csv_file, encoding="utf-8")
            data[split_type][dataset_name] = df
            print(f"Loaded {split_type}: {dataset_name} ({len(df)} articles)")
        except Exception as e:
            print(f"Error loading {csv_file}: {e}", file=sys.stderr)
            continue

    return data


def parse_categories(cats_str: str) -> list[str]:
    """
    Parse categories from CSV column (pipe-separated format).

    :param cats_str: Categories string (e.g., "01000000|04000000|20000002")
    :return: List of category strings
    """
    if not cats_str or pd.isna(cats_str):
        return []
    # Categories are pipe-separated: "01000000|04000000|20000002"
    return [cat.strip() for cat in str(cats_str).split("|") if cat.strip()]


def parse_entities(entities_str: str) -> list[dict]:
    """
    Parse entities from CSV column (JSON array).

    :param entities_str: Entities JSON string
    :return: List of entity dictionaries
    """
    if not entities_str or pd.isna(entities_str):
        return []
    try:
        if isinstance(entities_str, str):
            return json.loads(entities_str)
        return entities_str if isinstance(entities_str, list) else []
    except (json.JSONDecodeError, ValueError):
        return []


def compute_dataset_statistics(data: dict[str, dict[str, pd.DataFrame]]) -> dict:
    """
    Compute dataset distribution statistics.

    :param data: Dictionary mapping split_type -> dataset_name -> DataFrame
    :return: Dictionary with statistics
    """
    stats = {}

    for split_type in ["train", "dev", "test"]:
        datasets = data.get(split_type, {})
        total_articles = sum(len(df) for df in datasets.values())

        dataset_counts = {name: len(df) for name, df in datasets.items()}
        dataset_percentages = {
            name: (count / total_articles * 100) if total_articles > 0 else 0.0
            for name, count in dataset_counts.items()
        }

        stats[split_type] = {
            "total_articles": total_articles,
            "dataset_counts": dataset_counts,
            "dataset_percentages": dataset_percentages,
            "num_datasets": len(datasets),
        }

    return stats


def compute_category_statistics(data: dict[str, dict[str, pd.DataFrame]]) -> dict:
    """
    Compute category distribution statistics.

    :param data: Dictionary mapping split_type -> dataset_name -> DataFrame
    :return: Dictionary with category statistics
    """
    stats = {}

    # Get all categories across all splits
    all_categories: set[str] = set()

    for split_type in ["train", "dev", "test"]:
        datasets = data.get(split_type, {})
        split_categories: Counter[str] = Counter()
        dataset_category_counts: dict[str, Counter[str]] = {}

        for dataset_name, df in datasets.items():
            dataset_cats: Counter[str] = Counter()

            if "cats" in df.columns:
                for cats_str in df["cats"]:
                    categories = parse_categories(cats_str)
                    for cat in categories:
                        split_categories[cat] += 1
                        dataset_cats[cat] += 1
                        all_categories.add(cat)

            dataset_category_counts[dataset_name] = dataset_cats

        total_cat_mentions = sum(split_categories.values())
        category_percentages = {
            cat: (count / total_cat_mentions * 100) if total_cat_mentions > 0 else 0.0
            for cat, count in split_categories.items()
        }

        stats[split_type] = {
            "category_counts": dict(split_categories),
            "category_percentages": category_percentages,
            "dataset_category_counts": {name: dict(cats) for name, cats in dataset_category_counts.items()},
            "total_categories": len(split_categories),
        }

    return stats


def compute_missing_categories(category_stats: dict) -> dict[str, list[str]]:
    """
    Compute missing categories between splits.

    :param category_stats: Category statistics dictionary
    :return: Dictionary with missing category comparisons
    """
    train_cats = set(category_stats["train"]["category_counts"].keys())
    dev_cats = set(category_stats["dev"]["category_counts"].keys())
    test_cats = set(category_stats["test"]["category_counts"].keys())

    return {
        "test_vs_train": sorted(list(test_cats - train_cats)),
        "dev_vs_train": sorted(list(dev_cats - train_cats)),
        "train_vs_test": sorted(list(train_cats - test_cats)),
        "train_vs_dev": sorted(list(train_cats - dev_cats)),
    }


def compute_article_statistics(data: dict[str, dict[str, pd.DataFrame]]) -> dict:
    """
    Compute article-level statistics (length, entities, categories).

    :param data: Dictionary mapping split_type -> dataset_name -> DataFrame
    :return: Dictionary with article statistics
    """
    stats = {}

    for split_type in ["train", "dev", "test"]:
        datasets = data.get(split_type, {})
        article_lengths = []
        num_categories_list = []
        num_entities_list = []
        num_entities_with_gkbid = []
        num_entities_without_gkbid = []

        for df in datasets.values():
            # Article length (text column)
            if "text" in df.columns:
                lengths = df["text"].fillna("").astype(str).str.len()
                article_lengths.extend(lengths.tolist())

            # Number of categories
            if "cats" in df.columns:
                for cats_str in df["cats"]:
                    categories = parse_categories(cats_str)
                    num_categories_list.append(len(categories))

            # Number of entities (only in files starting with 'entities_')
            if "entities" in df.columns:
                for entities_str in df["entities"]:
                    entities = parse_entities(entities_str)
                    num_entities_list.append(len(entities))

                    # Count entities with and without gkbId (note: lowercase 'd')
                    # Check both 'gkbId' and 'gkbID' for compatibility
                    with_gkbid = sum(1 for e in entities if e.get("gkbId") or e.get("gkbID"))
                    num_entities_with_gkbid.append(with_gkbid)
                    num_entities_without_gkbid.append(len(entities) - with_gkbid)

        stats[split_type] = {
            "article_length": {
                "mean": np.mean(article_lengths) if article_lengths else 0.0,
                "std": np.std(article_lengths) if article_lengths else 0.0,
                "count": len(article_lengths),
            },
            "num_categories": {
                "mean": np.mean(num_categories_list) if num_categories_list else 0.0,
                "std": np.std(num_categories_list) if num_categories_list else 0.0,
                "count": len(num_categories_list),
            },
            "num_entities": {
                "mean": np.mean(num_entities_list) if num_entities_list else 0.0,
                "std": np.std(num_entities_list) if num_entities_list else 0.0,
                "count": len(num_entities_list),
            },
            "entities_with_gkbid": {
                "mean": np.mean(num_entities_with_gkbid) if num_entities_with_gkbid else 0.0,
                "std": np.std(num_entities_with_gkbid) if num_entities_with_gkbid else 0.0,
                "total": sum(num_entities_with_gkbid),
            },
            "entities_without_gkbid": {
                "mean": np.mean(num_entities_without_gkbid) if num_entities_without_gkbid else 0.0,
                "std": np.std(num_entities_without_gkbid) if num_entities_without_gkbid else 0.0,
                "total": sum(num_entities_without_gkbid),
            },
        }

    return stats


def plot_dataset_distribution_comparison(
    dataset_stats1: dict, dataset_stats2: dict, name1: str, name2: str, output_dir: Path
) -> None:
    """
    Create comparison plots for dataset distribution between two splits.

    :param dataset_stats1: Dataset statistics for first split
    :param dataset_stats2: Dataset statistics for second split
    :param name1: Name of first split
    :param name2: Name of second split
    :param output_dir: Output directory for plots
    """
    # Plot 1: Compare train splits from both datasets
    for split_type in ["train", "dev", "test"]:
        stats1 = dataset_stats1[split_type]
        stats2 = dataset_stats2[split_type]

        # Get all unique datasets from both
        all_datasets = sorted(set(list(stats1["dataset_counts"].keys()) + list(stats2["dataset_counts"].keys())))

        percentages1 = [stats1["dataset_percentages"].get(d, 0.0) for d in all_datasets]
        percentages2 = [stats2["dataset_percentages"].get(d, 0.0) for d in all_datasets]

        fig, ax = plt.subplots(figsize=(14, 6))
        fig.suptitle(
            f"Dataset Distribution Comparison: {split_type.capitalize()} Split", fontsize=16, fontweight="bold"
        )

        x = range(len(all_datasets))
        width = 0.35

        bars1 = ax.bar(
            [i - width / 2 for i in x],
            percentages1,
            width,
            label=name1,
            color="steelblue",
            edgecolor="black",
            linewidth=1.5,
        )
        bars2 = ax.bar(
            [i + width / 2 for i in x],
            percentages2,
            width,
            label=name2,
            color="coral",
            edgecolor="black",
            linewidth=1.5,
        )

        ax.set_xlabel("Dataset", fontsize=11)
        ax.set_ylabel("Percentage of Articles", fontsize=11)
        ax.set_title(
            f"{name1}: {stats1['total_articles']:,} articles | {name2}: {stats2['total_articles']:,} articles",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(all_datasets, rotation=45, ha="right", fontsize=9)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

        # Add value labels (percentages only)
        for bars, percentages in [(bars1, percentages1), (bars2, percentages2)]:
            for bar, pct in zip(bars, percentages):
                if pct > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{pct:.1f}%",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        plt.tight_layout()
        output_file = output_dir / f"dataset_distribution_comparison_{split_type}.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  -> Saved dataset distribution comparison ({split_type}): {output_file}")


def plot_dataset_distribution_within_split(dataset_stats: dict, name: str, output_dir: Path) -> None:
    """
    Create plots showing train/dev/test distribution within each dataset for one split.

    :param dataset_stats: Dataset statistics dictionary
    :param name: Name of the split
    :param output_dir: Output directory for plots
    """
    # Get all datasets
    all_datasets_set = set()
    for split_type in ["train", "dev", "test"]:
        all_datasets_set.update(dataset_stats[split_type]["dataset_counts"].keys())
    all_datasets = sorted(list(all_datasets_set))

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(f"Train/Dev/Test Distribution by Dataset: {name}", fontsize=16, fontweight="bold")

    x = range(len(all_datasets))
    width = 0.25

    train_pcts = [dataset_stats["train"]["dataset_percentages"].get(d, 0.0) for d in all_datasets]
    dev_pcts = [dataset_stats["dev"]["dataset_percentages"].get(d, 0.0) for d in all_datasets]
    test_pcts = [dataset_stats["test"]["dataset_percentages"].get(d, 0.0) for d in all_datasets]

    bars1 = ax.bar(
        [i - width for i in x],
        train_pcts,
        width,
        label="Train",
        color="steelblue",
        edgecolor="black",
        linewidth=1.5,
    )
    bars2 = ax.bar(x, dev_pcts, width, label="Dev", color="orange", edgecolor="black", linewidth=1.5)
    bars3 = ax.bar(
        [i + width for i in x],
        test_pcts,
        width,
        label="Test",
        color="green",
        edgecolor="black",
        linewidth=1.5,
    )

    ax.set_xlabel("Dataset", fontsize=11)
    ax.set_ylabel("Percentage of Articles", fontsize=11)
    ax.set_title(
        f"Train: {dataset_stats['train']['total_articles']:,} | "
        f"Dev: {dataset_stats['dev']['total_articles']:,} | "
        f"Test: {dataset_stats['test']['total_articles']:,}",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(all_datasets, rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Add value labels (percentages only)
    for bars, percentages in [(bars1, train_pcts), (bars2, dev_pcts), (bars3, test_pcts)]:
        for bar, pct in zip(bars, percentages):
            if pct > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{pct:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    plt.tight_layout()
    output_file = output_dir / f"dataset_distribution_within_split_{name}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved dataset distribution within split: {output_file}")


def plot_category_distribution(category_stats: dict, output_dir: Path) -> None:
    """
    Create plots for category distribution.

    :param category_stats: Category statistics dictionary
    :param output_dir: Output directory for plots
    """
    # Plot 1: Top categories by split
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Top 20 Categories by Split", fontsize=16, fontweight="bold")

    for idx, split_type in enumerate(["train", "dev", "test"]):
        stats = category_stats[split_type]
        # Get top 20 categories
        sorted_cats = sorted(stats["category_counts"].items(), key=lambda x: x[1], reverse=True)[:20]
        categories = [cat for cat, _ in sorted_cats]
        percentages = [stats["category_percentages"][cat] for cat in categories]

        ax = axes[idx]
        bars = ax.barh(range(len(categories)), percentages, color=f"C{idx}", edgecolor="black", linewidth=1.5)
        ax.set_xlabel("Percentage of Category Mentions", fontsize=11)
        ax.set_ylabel("Category", fontsize=11)
        ax.set_title(
            f"{split_type.capitalize()} Split\n({stats['total_categories']} unique categories)",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_yticks(range(len(categories)))
        ax.set_yticklabels(categories, fontsize=9)
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        ax.invert_yaxis()

        # Add value labels
        for bar, pct in zip(bars, percentages):
            if pct > 0:
                ax.text(
                    bar.get_width(),
                    bar.get_y() + bar.get_height() / 2,
                    f" {pct:.2f}%",
                    ha="left",
                    va="center",
                    fontsize=8,
                )

    plt.tight_layout()
    plt.savefig(output_dir / "category_distribution_top20.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved category distribution plot: {output_dir / 'category_distribution_top20.png'}")


def plot_categories_entities_comparison(
    article_stats1: dict, article_stats2: dict, name1: str, name2: str, output_dir: Path
) -> None:
    """
    Create comparison plots for categories and entities statistics.

    :param article_stats1: Article statistics for first split
    :param article_stats2: Article statistics for second split
    :param name1: Name of first split
    :param name2: Name of second split
    :param output_dir: Output directory for plots
    """
    # Plot 1: Categories comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Categories Statistics Comparison", fontsize=16, fontweight="bold")

    for idx, split_type in enumerate(["train", "dev", "test"]):
        stats1 = article_stats1[split_type]["num_categories"]
        stats2 = article_stats2[split_type]["num_categories"]

        ax = axes[idx]
        x = [0, 1]
        width = 0.35

        bars1 = ax.bar(
            [i - width / 2 for i in x],
            [stats1["mean"], stats1["std"]],
            width,
            label=name1,
            color="steelblue",
            edgecolor="black",
            linewidth=1.5,
        )
        bars2 = ax.bar(
            [i + width / 2 for i in x],
            [stats2["mean"], stats2["std"]],
            width,
            label=name2,
            color="coral",
            edgecolor="black",
            linewidth=1.5,
        )

        ax.set_ylabel("Number of Categories", fontsize=11)
        ax.set_title(f"{split_type.capitalize()} Split", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["Mean", "Std"])
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

        # Add value labels
        for bars, stats in [(bars1, stats1), (bars2, stats2)]:
            for bar, value in zip(bars, [stats["mean"], stats["std"]]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

    plt.tight_layout()
    output_file = output_dir / "categories_statistics_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved categories statistics comparison: {output_file}")

    # Plot 2: Entities comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Entities Statistics Comparison", fontsize=16, fontweight="bold")

    for idx, split_type in enumerate(["train", "dev", "test"]):
        stats1 = article_stats1[split_type]["num_entities"]
        stats2 = article_stats2[split_type]["num_entities"]

        ax = axes[idx]
        x = [0, 1]
        width = 0.35

        bars1 = ax.bar(
            [i - width / 2 for i in x],
            [stats1["mean"], stats1["std"]],
            width,
            label=name1,
            color="steelblue",
            edgecolor="black",
            linewidth=1.5,
        )
        bars2 = ax.bar(
            [i + width / 2 for i in x],
            [stats2["mean"], stats2["std"]],
            width,
            label=name2,
            color="coral",
            edgecolor="black",
            linewidth=1.5,
        )

        ax.set_ylabel("Number of Entities", fontsize=11)
        ax.set_title(f"{split_type.capitalize()} Split", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["Mean", "Std"])
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

        # Add value labels
        for bars, stats in [(bars1, stats1), (bars2, stats2)]:
            for bar, value in zip(bars, [stats["mean"], stats["std"]]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

    plt.tight_layout()
    output_file = output_dir / "entities_statistics_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved entities statistics comparison: {output_file}")


def plot_article_statistics(article_stats: dict, output_dir: Path) -> None:
    """
    Create plots for article statistics.

    :param article_stats: Article statistics dictionary
    :param output_dir: Output directory for plots
    """
    # Plot 1: Article length statistics
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Article Length Statistics by Split", fontsize=16, fontweight="bold")

    for idx, split_type in enumerate(["train", "dev", "test"]):
        stats = article_stats[split_type]["article_length"]
        ax = axes[idx]
        ax.bar(
            ["Mean", "Std"],
            [stats["mean"], stats["std"]],
            color=f"C{idx}",
            edgecolor="black",
            linewidth=1.5,
        )
        ax.set_ylabel("Characters", fontsize=11)
        ax.set_title(f"{split_type.capitalize()} Split\n(n={stats['count']:,})", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        # Add value labels
        ax.text(0, stats["mean"], f"{stats['mean']:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.text(1, stats["std"], f"{stats['std']:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_dir / "article_length_statistics.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved article length statistics: {output_dir / 'article_length_statistics.png'}")

    # Plot 3: Entities with/without gkbID
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Entity Statistics: With and Without gkbID", fontsize=16, fontweight="bold")

    for idx, split_type in enumerate(["train", "dev", "test"]):
        stats = article_stats[split_type]
        with_gkbid = stats["entities_with_gkbid"]["total"]
        without_gkbid = stats["entities_without_gkbid"]["total"]
        total = with_gkbid + without_gkbid

        ax = axes[idx]
        bars = ax.bar(
            ["With gkbID", "Without gkbID"],
            [with_gkbid, without_gkbid],
            color=[f"C{idx}", "lightcoral"],
            edgecolor="black",
            linewidth=1.5,
        )
        ax.set_ylabel("Total Entities", fontsize=11)
        ax.set_title(f"{split_type.capitalize()} Split\n(Total: {total:,})", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        # Add value labels and percentages
        for bar, value in zip(bars, [with_gkbid, without_gkbid]):
            pct = (value / total * 100) if total > 0 else 0
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:,}\n({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(output_dir / "entity_gkbid_statistics.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved entity gkbID statistics: {output_dir / 'entity_gkbid_statistics.png'}")


def create_missing_categories_comparison_table(
    category_stats1: dict,
    category_stats2: dict,
    missing_cats1: dict[str, list[str]],
    missing_cats2: dict[str, list[str]],
    name1: str,
    name2: str,
    output_dir: Path,
) -> None:
    """
    Create comparison table for missing categories between splits.

    :param category_stats1: Category statistics for first split
    :param category_stats2: Category statistics for second split
    :param missing_cats1: Missing categories for first split
    :param missing_cats2: Missing categories for second split
    :param name1: Name of first split
    :param name2: Name of second split
    :param output_dir: Output directory
    """
    rows = []

    # Overall categories in each split
    for split_type in ["train", "dev", "test"]:
        total_cats1 = category_stats1[split_type]["total_categories"]
        total_cats2 = category_stats2[split_type]["total_categories"]
        rows.append(
            {
                "Split": split_type.capitalize(),
                "Comparison": "Total Categories",
                f"{name1}": total_cats1,
                f"{name2}": total_cats2,
            }
        )

    # Missing categories comparisons
    comparisons = [
        ("test_vs_train", "Test vs Train"),
        ("dev_vs_train", "Dev vs Train"),
        ("train_vs_test", "Train vs Test"),
        ("train_vs_dev", "Train vs Dev"),
    ]

    for comp_key, comp_name in comparisons:
        missing1 = len(missing_cats1.get(comp_key, []))
        missing2 = len(missing_cats2.get(comp_key, []))
        rows.append(
            {
                "Split": "-",
                "Comparison": comp_name,
                f"{name1}": missing1,
                f"{name2}": missing2,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "missing_categories_comparison.csv", index=False)
    print(f"  -> Saved missing categories comparison table: {output_dir / 'missing_categories_comparison.csv'}")


def create_statistics_tables(
    dataset_stats: dict,
    category_stats: dict,
    missing_cats: dict[str, list[str]],
    article_stats: dict,
    output_dir: Path,
) -> None:
    """
    Create statistics tables in CSV and markdown format.

    :param dataset_stats: Dataset statistics
    :param category_stats: Category statistics
    :param missing_cats: Missing categories dictionary
    :param article_stats: Article statistics
    :param output_dir: Output directory
    """
    # Table 1: Dataset distribution
    rows = []
    for split_type in ["train", "dev", "test"]:
        stats = dataset_stats[split_type]
        for dataset, count in sorted(stats["dataset_counts"].items()):
            rows.append(
                {
                    "Split": split_type.capitalize(),
                    "Dataset": dataset,
                    "Articles": count,
                    "Percentage": f"{stats['dataset_percentages'][dataset]:.2f}%",
                }
            )

    df_dataset = pd.DataFrame(rows)
    df_dataset.to_csv(output_dir / "dataset_distribution.csv", index=False)
    print(f"  -> Saved dataset distribution table: {output_dir / 'dataset_distribution.csv'}")

    # Table 2: Category distribution by split
    rows = []
    all_categories = set()
    for split_type in ["train", "dev", "test"]:
        all_categories.update(category_stats[split_type]["category_counts"].keys())

    for cat in sorted(all_categories):
        row = {"Category": cat}
        for split_type in ["train", "dev", "test"]:
            count = category_stats[split_type]["category_counts"].get(cat, 0)
            pct = category_stats[split_type]["category_percentages"].get(cat, 0.0)
            row[f"{split_type.capitalize()}_Count"] = count
            row[f"{split_type.capitalize()}_Percentage"] = f"{pct:.2f}%"
        rows.append(row)

    df_categories = pd.DataFrame(rows)
    df_categories.to_csv(output_dir / "category_distribution.csv", index=False)
    print(f"  -> Saved category distribution table: {output_dir / 'category_distribution.csv'}")

    # Table 3: Missing categories
    rows = []
    for comparison, missing in missing_cats.items():
        for cat in missing:
            rows.append({"Comparison": comparison, "Missing_Category": cat})

    if rows:
        df_missing = pd.DataFrame(rows)
        df_missing.to_csv(output_dir / "missing_categories.csv", index=False)
        print(f"  -> Saved missing categories table: {output_dir / 'missing_categories.csv'}")

    # Table 4: Article statistics summary
    rows = []
    for split_type in ["train", "dev", "test"]:
        stats = article_stats[split_type]
        rows.append(
            {
                "Split": split_type.capitalize(),
                "Article_Length_Mean": f"{stats['article_length']['mean']:.0f}",
                "Article_Length_Std": f"{stats['article_length']['std']:.0f}",
                "Num_Categories_Mean": f"{stats['num_categories']['mean']:.2f}",
                "Num_Categories_Std": f"{stats['num_categories']['std']:.2f}",
                "Num_Entities_Mean": f"{stats['num_entities']['mean']:.2f}",
                "Num_Entities_Std": f"{stats['num_entities']['std']:.2f}",
                "Entities_With_gkbID_Total": stats["entities_with_gkbid"]["total"],
                "Entities_Without_gkbID_Total": stats["entities_without_gkbid"]["total"],
            }
        )

    df_article_stats = pd.DataFrame(rows)
    df_article_stats.to_csv(output_dir / "article_statistics_summary.csv", index=False)
    print(f"  -> Saved article statistics summary: {output_dir / 'article_statistics_summary.csv'}")


def main() -> None:
    """Main entry point for the script."""
    argparser = argparse.ArgumentParser(
        description=(
            "Analyze dataset splits and generate statistics and visualizations. "
            "Takes directories containing CSV files and creates comprehensive analysis."
        )
    )
    argparser.add_argument(
        "directories",
        nargs=2,
        help="Exactly two directories containing CSV files to compare",
    )
    argparser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output directory for results (default: analysis_results in first input directory)",
    )
    args = argparser.parse_args()

    directories = [Path(d) for d in args.directories]

    for directory in directories:
        if not directory.is_dir():
            print(f"Error: {directory} is not a directory", file=sys.stderr)
            sys.exit(1)

    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = directories[0] / "analysis_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Comparing two data splits...")
    print(f"Output directory: {output_dir}\n")

    # Process both directories
    all_results = {}
    all_data = {}

    for directory in directories:
        print(f"Processing directory: {directory}")
        data = load_csv_files(directory)

        if not any(data.values()):
            print(f"  Warning: No data found in {directory}", file=sys.stderr)
            sys.exit(1)

        # Compute statistics
        print("\nComputing statistics...")
        dataset_stats = compute_dataset_statistics(data)
        category_stats = compute_category_statistics(data)
        missing_cats = compute_missing_categories(category_stats)
        article_stats = compute_article_statistics(data)

        # Store results with directory name as key
        dir_name = directory.name
        all_results[dir_name] = {
            "dataset_stats": dataset_stats,
            "category_stats": category_stats,
            "missing_cats": missing_cats,
            "article_stats": article_stats,
        }
        all_data[dir_name] = data

        # Create output subdirectory for this dataset
        dataset_output_dir = output_dir / dir_name
        dataset_output_dir.mkdir(parents=True, exist_ok=True)

        # Generate individual visualizations
        print("\nGenerating individual visualizations...")
        plot_category_distribution(category_stats, dataset_output_dir)
        plot_article_statistics(article_stats, dataset_output_dir)

        # Generate tables
        print("\nGenerating statistics tables...")
        create_statistics_tables(dataset_stats, category_stats, missing_cats, article_stats, dataset_output_dir)

        print(f"\nAnalysis complete for {dir_name}!\n")

    # Create comparison visualizations
    if len(all_results) == 2:
        print("Creating comparison visualizations...")
        names = list(all_results.keys())
        name1, name2 = names[0], names[1]

        # Comparison plots for dataset distribution
        plot_dataset_distribution_comparison(
            all_results[name1]["dataset_stats"],
            all_results[name2]["dataset_stats"],
            name1,
            name2,
            output_dir,
        )

        # Within-split distribution plots for each dataset
        plot_dataset_distribution_within_split(all_results[name1]["dataset_stats"], name1, output_dir)
        plot_dataset_distribution_within_split(all_results[name2]["dataset_stats"], name2, output_dir)

        # Categories and entities comparison plots
        plot_categories_entities_comparison(
            all_results[name1]["article_stats"],
            all_results[name2]["article_stats"],
            name1,
            name2,
            output_dir,
        )

        # Missing categories comparison table
        create_missing_categories_comparison_table(
            all_results[name1]["category_stats"],
            all_results[name2]["category_stats"],
            all_results[name1]["missing_cats"],
            all_results[name2]["missing_cats"],
            name1,
            name2,
            output_dir,
        )

    print(f"\nAll analyses saved to: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
