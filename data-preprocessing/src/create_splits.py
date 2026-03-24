#!/usr/bin/env python3
"""
Create new train/dev/test splits from .jsonl.gz files:
- Global chronological split (all articles together)
- Per-dataset chronological split (each dataset independently)
Outputs jsonl.gz files. For per-dataset splits, creates train/test/dev files
for all datasets, even if they have no articles with dates.
"""
import argparse
import gzip
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

# Since this is a CLI script, print is acceptable for output
# For general-purpose tools, logging should be used instead


@dataclass(frozen=True)
class ArticleRecord:
    """Single article record used during split creation."""

    date: datetime | None
    article_id: str
    dataset: str
    json_line: str
    original_split: str


def parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """
    Parse date string to datetime object.

    :param date_str: Date string in format 'YYYY-MM-DD' or ISO format (e.g., '2021-09-19T00:00:00Z')
    :return: Timezone-naive datetime object or None if date_str is None or invalid
    """
    if not date_str:
        return None
    try:
        # Try ISO format first (handles both with and without timezone)
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        # Normalize to timezone-naive for consistent comparison
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return dt
    except (ValueError, AttributeError):
        # Fall back to simple YYYY-MM-DD format
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return None


def extract_dataset_name(filepath: Path) -> Optional[str]:
    """
    Extract dataset name from filepath.

    :param filepath: Path to .jsonl.gz file
    :return: Dataset name (e.g., 'en_bbc_iptc' from
             'en_bbc_iptc.dev_all.analysis.jsonl.gz' or
             'de_dpa_iptc' from 'de_dpa_iptc.train_smallpp.analysis.jsonl.gz')
    """
    # Handle .gz files
    name = filepath.name
    if name.endswith(".gz"):
        name = name[:-3]  # Remove .gz

    # Remove .analysis.jsonl
    name = name.replace(".analysis.jsonl", "")

    # Remove split suffixes: .train_all, .dev_all, .test_all, .train, .dev,
    # .test. Also handle patterns like _smallpp, _medium
    for suffix in [
        ".train_all",
        ".dev_all",
        ".test_all",
        ".train_smallpp",
        ".dev_smallpp",
        ".test_smallpp",
        ".train_medium",
        ".dev_medium",
        ".test_medium",
        ".train",
        ".dev",
        ".test",
    ]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def read_all_articles(
    input_dir: Path,
) -> Tuple[
    list[ArticleRecord],
    list[ArticleRecord],
    dict[str, Counter[str]],
    set[str],
]:
    """
    Read all .jsonl.gz files and extract articles with dates
    and dataset info, preserving the full JSON line content.

    :param input_dir: Directory containing .jsonl.gz files
    :return: Tuple of (articles_with_dates, articles_without_dates,
             original_splits, all_datasets) where:
             - articles_with_dates: list of article records with dates
             - articles_without_dates: list of article records without dates
             - original_splits: dict mapping 'train'/'dev'/'test' to
               Counter of dataset counts
             - all_datasets: set of all unique dataset names found
    """
    articles_with_dates: list[ArticleRecord] = []
    articles_without_dates: list[ArticleRecord] = []
    original_splits: dict[str, Counter[str]] = {"train": Counter(), "dev": Counter(), "test": Counter()}
    all_datasets: set[str] = set()

    # Find .jsonl.gz files
    files = sorted(input_dir.glob("*.jsonl.gz"))

    for filepath in files:
        dataset_name = extract_dataset_name(filepath=filepath)
        if not dataset_name:
            continue

        all_datasets.add(dataset_name)

        # Determine original split from filename
        filename = filepath.name
        if ".train" in filename:
            original_split = "train"
        elif ".dev" in filename:
            original_split = "dev"
        elif ".test" in filename:
            original_split = "test"
        else:
            original_split = "train"  # default

        try:
            with gzip.open(filepath, mode="rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    article = json.loads(line)
                    article_id = article.get("id", "")
                    metadata = article.get("metadata", {})
                    date_str = metadata.get("date") if metadata else None

                    date_obj = parse_date(date_str=date_str) if date_str else None

                    if date_obj:
                        articles_with_dates.append(
                            ArticleRecord(
                                date=date_obj,
                                article_id=article_id,
                                dataset=dataset_name,
                                json_line=line,
                                original_split=original_split,
                            )
                        )
                        original_splits[original_split][dataset_name] += 1
                    else:
                        articles_without_dates.append(
                            ArticleRecord(
                                date=None,
                                article_id=article_id,
                                dataset=dataset_name,
                                json_line=line,
                                original_split=original_split,
                            )
                        )

        except OSError as e:
            print(f"Error reading {filepath}: {e}", file=sys.stderr)

    return articles_with_dates, articles_without_dates, original_splits, all_datasets


def create_chronological_splits(
    articles_with_dates: list[ArticleRecord],
) -> Tuple[list[ArticleRecord], list[ArticleRecord], list[ArticleRecord]]:
    """
    Create new train/dev/test splits based on date ordering (global).

    :param articles_with_dates: List of dated article records
    :return: Tuple of (train_articles, dev_articles, test_articles)
    """
    # Sort by date (oldest first)
    sorted_articles = sorted(articles_with_dates, key=lambda article: article.date or datetime.min)

    total = len(sorted_articles)
    train_size = int(total * 0.8)
    test_size = int(total * 0.1)
    dev_size = total - train_size - test_size

    train_articles = sorted_articles[:train_size]
    dev_articles = sorted_articles[train_size : train_size + dev_size]
    test_articles = sorted_articles[train_size + dev_size :]

    return train_articles, dev_articles, test_articles


def create_per_dataset_splits(
    articles_with_dates: list[ArticleRecord],
    articles_without_dates: list[ArticleRecord],
    all_datasets: set[str],
    *,
    seed: int = 42,
) -> dict[
    str,
    Tuple[
        list[ArticleRecord],
        list[ArticleRecord],
        list[ArticleRecord],
    ],
]:
    """
    Create new train/dev/test splits per dataset based on date ordering.
    If dated articles already produce a non-empty test split, all undated
    articles go to train. Otherwise, only ``nl_eventdna`` datasets use a
    random 8/1/1 train/dev/test split for undated articles; other datasets
    still place undated articles into train.

    :param articles_with_dates: List of dated article records
    :param articles_without_dates: List of undated article records
    :param all_datasets: Set of dataset names
    :param seed: Random seed for undated fallback split
    :return: Dict mapping dataset name to (train_articles, dev_articles, test_articles)
    """
    articles_by_dataset = defaultdict(list)
    for article in articles_with_dates:
        articles_by_dataset[article.dataset].append(article)

    undated_by_dataset = defaultdict(list)
    for article in articles_without_dates:
        undated_by_dataset[article.dataset].append(article)

    per_dataset_splits = {}
    rng = random.Random(seed)

    for dataset in all_datasets:
        dataset_articles = articles_by_dataset.get(dataset, [])
        dataset_undated = undated_by_dataset.get(dataset, [])

        if dataset_articles:
            sorted_articles = sorted(dataset_articles, key=lambda article: article.date or datetime.min)
            total = len(sorted_articles)
            train_size = int(total * 0.8)
            test_size = int(total * 0.1)
            dev_size = total - train_size - test_size
            train_articles = list(sorted_articles[:train_size])
            dev_articles = list(sorted_articles[train_size : train_size + dev_size])
            test_articles = list(sorted_articles[train_size + dev_size :])
        else:
            train_articles = []
            dev_articles = []
            test_articles = []

        if test_articles:
            train_articles.extend(dataset_undated)
        elif dataset_undated:
            if dataset.startswith("nl_eventdna"):
                shuffled_undated = list(dataset_undated)
                rng.shuffle(shuffled_undated)
                undated_total = len(shuffled_undated)
                undated_train_size = int(undated_total * 0.8)
                undated_test_size = int(undated_total * 0.1)
                undated_dev_size = undated_total - undated_train_size - undated_test_size
                train_articles.extend(shuffled_undated[:undated_train_size])
                dev_articles.extend(shuffled_undated[undated_train_size : undated_train_size + undated_dev_size])
                test_articles.extend(shuffled_undated[undated_train_size + undated_dev_size :])
            else:
                train_articles.extend(dataset_undated)

        per_dataset_splits[dataset] = (train_articles, dev_articles, test_articles)

    return per_dataset_splits


def write_split_file(articles: list[ArticleRecord], output_path: Path) -> None:
    """
    Write articles to a gzipped jsonl file.

    :param articles: List of article tuples
    :param output_path: Path where to save the file (.jsonl.gz)
    """
    try:
        with gzip.open(output_path, mode="wt", encoding="utf-8") as f:
            for article in articles:
                f.write(article.json_line + "\n")
        print(f"  -> Wrote {len(articles)} articles to: {output_path}")
    except OSError as e:
        print(f"  -> Error writing {output_path}: {e}", file=sys.stderr)


def write_undated_articles(articles: list[ArticleRecord], output_path: Path) -> None:
    """
    Write list of articles without dates to a file.

    :param articles_without_dates: List of undated article records
    :param output_path: Path where to save the list
    """
    try:
        with open(output_path, mode="w", encoding="utf-8") as f:
            f.write("Articles without dates:\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total: {len(articles)}\n\n")
            f.write("Dataset\tArticle ID\n")
            f.write("-" * 60 + "\n")
            for article in articles:
                f.write(f"{article.dataset}\t{article.article_id}\n")
        print(f"  -> Saved articles without dates to: {output_path}")
    except OSError as e:
        print(f"  -> Error writing {output_path}: {e}", file=sys.stderr)


def main() -> None:
    """Main entry point for the script."""
    argparser = argparse.ArgumentParser(
        description=(
            "Create new train/dev/test splits from .jsonl.gz files. "
            "Supports global chronological split (all articles together), "
            "or per-dataset chronological split (each dataset independently)."
        )
    )
    argparser.add_argument(
        "input",
        nargs="?",
        default=".",
        help=("Input directory containing .jsonl.gz files " "(default: current directory)"),
    )
    argparser.add_argument(
        "-o", "--output", type=str, default=None, help="Output directory for results (default: same as input)"
    )
    argparser.add_argument(
        "--split",
        choices=["global", "per-dataset"],
        default="per-dataset",
        help=(
            'Type of split to create: "global" (all articles together), '
            '"per-dataset" (each dataset independently, default)'
        ),
    )
    args = argparser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir

    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading articles from {input_dir}...")
    articles_with_dates, articles_without_dates, _, all_datasets = read_all_articles(input_dir=input_dir)

    print(f"Found {len(articles_with_dates)} articles with dates")
    print(f"Found {len(articles_without_dates)} articles without dates")
    print(f"Found {len(all_datasets)} unique datasets")

    split_type = args.split_type

    # Create global chronological splits
    if split_type == "global":
        print("\nCreating global chronological splits...")
        train_articles, dev_articles, test_articles = create_chronological_splits(
            articles_with_dates=articles_with_dates
        )

        # Group articles by dataset within each split
        train_by_dataset: dict[str, list[ArticleRecord]] = defaultdict(list)
        dev_by_dataset: dict[str, list[ArticleRecord]] = defaultdict(list)
        test_by_dataset: dict[str, list[ArticleRecord]] = defaultdict(list)

        for article in train_articles:
            train_by_dataset[article.dataset].append(article)

        for article in dev_articles:
            dev_by_dataset[article.dataset].append(article)

        for article in test_articles:
            test_by_dataset[article.dataset].append(article)

        # Write per-dataset files for all datasets (even if empty)
        for dataset in all_datasets:
            train_arts = train_by_dataset.get(dataset, [])
            dev_arts = dev_by_dataset.get(dataset, [])
            test_arts = test_by_dataset.get(dataset, [])

            write_split_file(articles=train_arts, output_path=output_dir / f"{dataset}.train.analysis.jsonl.gz")
            write_split_file(articles=dev_arts, output_path=output_dir / f"{dataset}.dev.analysis.jsonl.gz")
            write_split_file(articles=test_arts, output_path=output_dir / f"{dataset}.test.analysis.jsonl.gz")

    # Create per-dataset chronological splits
    if split_type == "per-dataset":
        print("\nCreating per-dataset chronological splits...")
        per_dataset_splits = create_per_dataset_splits(
            articles_with_dates=articles_with_dates,
            articles_without_dates=articles_without_dates,
            all_datasets=all_datasets,
        )

        total_train = 0
        total_dev = 0
        total_test = 0

        for dataset, (train_arts, dev_arts, test_arts) in per_dataset_splits.items():
            total_train += len(train_arts)
            total_dev += len(dev_arts)
            total_test += len(test_arts)

            write_split_file(articles=train_arts, output_path=output_dir / f"{dataset}.train.analysis.jsonl.gz")
            write_split_file(articles=dev_arts, output_path=output_dir / f"{dataset}.dev.analysis.jsonl.gz")
            write_split_file(articles=test_arts, output_path=output_dir / f"{dataset}.test.analysis.jsonl.gz")

        print("\nPer-dataset splits summary:")
        print(f"  Total train: {total_train} articles")
        print(f"  Total dev: {total_dev} articles")
        print(f"  Total test: {total_test} articles")
        print(f"  Number of datasets: {len(per_dataset_splits)}")

    # Write articles without dates
    no_dates_path = output_dir / "articles_without_dates.txt"
    write_undated_articles(articles=articles_without_dates, output_path=no_dates_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
