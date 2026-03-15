#!/usr/bin/env python3
"""
Analyze dataset splits and generate statistics and visualizations for thesis.
Computes dataset distribution, category distribution, missing categories,
article statistics, and entity statistics.
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
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

    :param directory: Directory containing CSV files
    :return: Dictionary mapping split_type -> dataset_name -> DataFrame
    """
    data: dict[str, dict[str, pd.DataFrame]] = {"train": {}, "dev": {}, "test": {}}

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

        dataset_name = extract_dataset_name(csv_file.name)
        if not dataset_name:
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
    Parse categories from CSV column (can be JSON array or comma-separated).

    :param cats_str: Categories string
    :return: List of category strings
    """
    if not cats_str or pd.isna(cats_str):
        return []
    try:
        # Try JSON first
        if cats_str.strip().startswith("["):
            return json.loads(cats_str)
        # Otherwise, try comma-separated
        return [cat.strip() for cat in str(cats_str).split(",") if cat.strip()]
    except (json.JSONDecodeError, ValueError):
        return []


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


def compute_missing_categories(
    category_stats: dict,
) -> dict[str, dict[str, list[str]]]:
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

            # Number of entities
            if "entities" in df.columns:
                for entities_str in df["entities"]:
                    entities = parse_entities(entities_str)
                    num_entities_list.append(len(entities))

                    # Count entities with and without gkbID
                    with_gkbid = sum(1 for e in entities if e.get("gkbID"))
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


def plot_dataset_distribution(dataset_stats: dict, output_dir: Path) -> None:
    """
    Create plots for dataset distribution.

    :param dataset_stats: Dataset statistics dictionary
    :param output_dir: Output directory for plots
    """
    # Plot 1: Dataset distribution by split
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Dataset Distribution by Split", fontsize=16, fontweight="bold")

    for idx, split_type in enumerate(["train", "dev", "test"]):
        stats = dataset_stats[split_type]
        datasets = sorted(stats["dataset_counts"].keys())
        counts = [stats["dataset_counts"][d] for d in datasets]
        percentages = [stats["dataset_percentages"][d] for d in datasets]

        ax = axes[idx]
        bars = ax.bar(range(len(datasets)), percentages, color=f"C{idx}", edgecolor="black", linewidth=1.5)
        ax.set_xlabel("Dataset", fontsize=11)
        ax.set_ylabel("Percentage of Articles", fontsize=11)
        ax.set_title(
            f"{split_type.capitalize()} Split\n({stats['total_articles']:,} articles)", fontsize=12, fontweight="bold"
        )
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        # Add value labels
        for bar, pct, count in zip(bars, percentages, counts):
            if pct > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{pct:.1f}%\n({count:,})",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    plt.tight_layout()
    plt.savefig(output_dir / "dataset_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved dataset distribution plot: {output_dir / 'dataset_distribution.png'}")


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

    # Plot 2: Number of categories and entities
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Article Statistics: Categories and Entities", fontsize=16, fontweight="bold")

    for idx, split_type in enumerate(["train", "dev", "test"]):
        stats = article_stats[split_type]

        # Categories
        ax = axes[0, idx]
        cat_stats = stats["num_categories"]
        ax.bar(
            ["Mean", "Std"],
            [cat_stats["mean"], cat_stats["std"]],
            color=f"C{idx}",
            edgecolor="black",
            linewidth=1.5,
        )
        ax.set_ylabel("Number of Categories", fontsize=11)
        ax.set_title(f"{split_type.capitalize()} Split", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.text(
            0, cat_stats["mean"], f"{cat_stats['mean']:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold"
        )
        ax.text(
            1, cat_stats["std"], f"{cat_stats['std']:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

        # Entities
        ax = axes[1, idx]
        ent_stats = stats["num_entities"]
        ax.bar(
            ["Mean", "Std"],
            [ent_stats["mean"], ent_stats["std"]],
            color=f"C{idx}",
            edgecolor="black",
            linewidth=1.5,
        )
        ax.set_ylabel("Number of Entities", fontsize=11)
        ax.set_title(f"{split_type.capitalize()} Split", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.text(
            0, ent_stats["mean"], f"{ent_stats['mean']:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold"
        )
        ax.text(
            1, ent_stats["std"], f"{ent_stats['std']:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    plt.tight_layout()
    plt.savefig(output_dir / "categories_entities_statistics.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved categories and entities statistics: {output_dir / 'categories_entities_statistics.png'}")

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


def create_statistics_tables(
    dataset_stats: dict,
    category_stats: dict,
    missing_cats: dict[str, dict[str, list[str]]],
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
        nargs="+",
        help="One or more directories containing CSV files to analyze",
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

    print(f"Analyzing splits from {len(directories)} directory(ies)...")
    print(f"Output directory: {output_dir}\n")

    # Process each directory
    all_results = {}

    for directory in directories:
        print(f"Processing directory: {directory}")
        data = load_csv_files(directory)

        if not any(data.values()):
            print(f"  Warning: No data found in {directory}", file=sys.stderr)
            continue

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

        # Create output subdirectory for this dataset
        dataset_output_dir = output_dir / dir_name
        dataset_output_dir.mkdir(parents=True, exist_ok=True)

        # Generate visualizations
        print("\nGenerating visualizations...")
        plot_dataset_distribution(dataset_stats, dataset_output_dir)
        plot_category_distribution(category_stats, dataset_output_dir)
        plot_article_statistics(article_stats, dataset_output_dir)

        # Generate tables
        print("\nGenerating statistics tables...")
        create_statistics_tables(dataset_stats, category_stats, missing_cats, article_stats, dataset_output_dir)

        print(f"\nAnalysis complete for {dir_name}!\n")

    # Create comparison summary if multiple directories
    if len(all_results) > 1:
        print("Creating comparison summary...")
        # TODO: Add comparison visualizations if needed
        print("  -> Comparison summary can be created from individual results")

    print(f"\nAll analyses saved to: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
