#!/usr/bin/env python3
"""
Visualize distributions of train/dev/test splits comparing original and new splits.
"""
import argparse
import gzip
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

# Since this is a CLI script, print is acceptable for output
# For general-purpose tools, logging should be used instead


def parse_date(date_str: str) -> Optional[datetime]:
    """
    Parse date string to datetime object.

    :param date_str: Date string in format 'YYYY-MM-DD' or 'None'
    :return: Datetime object or None if date_str is 'None' or invalid
    """
    if date_str == 'None' or not date_str:
        return None
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        return None


def extract_dataset_name(filepath: Path) -> Optional[str]:
    """
    Extract dataset name from filepath.

    :param filepath: Path to .jsonl.tsv or .jsonl.gz file
    :return: Dataset name (e.g., 'en_bbc_iptc' from
             'en_bbc_iptc.dev_all.analysis.jsonl.tsv' or
             'de_dpa_iptc' from 'de_dpa_iptc.train_smallpp.analysis.jsonl.gz')
    """
    # Handle both .tsv and .gz files
    name = filepath.name
    if name.endswith('.gz'):
        name = name[:-3]  # Remove .gz
    if name.endswith('.tsv'):
        name = name[:-4]  # Remove .tsv

    # Remove .analysis.jsonl
    name = name.replace('.analysis.jsonl', '')

    # Remove split suffixes: .train_all, .dev_all, .test_all, .train, .dev,
    # .test. Also handle patterns like _smallpp, _medium
    for suffix in [
        '.train_all', '.dev_all', '.test_all',
        '.train_smallpp', '.dev_smallpp', '.test_smallpp',
        '.train_medium', '.dev_medium', '.test_medium',
        '.train', '.dev', '.test'
    ]:
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def read_original_splits(
    input_dir: Path
) -> dict[str, Counter[str]]:
    """
    Read original splits from input directory to get original distribution.

    :param input_dir: Directory containing original .jsonl.tsv and .jsonl.gz files
    :return: Dict mapping 'train'/'dev'/'test' to Counter of dataset counts
    """
    original_splits: dict[str, Counter[str]] = {
        'train': Counter(),
        'dev': Counter(),
        'test': Counter()
    }

    # Find both .jsonl.tsv and .jsonl.gz files
    tsv_files = sorted(input_dir.glob('*.jsonl.tsv'))
    gz_files = sorted(input_dir.glob('*.jsonl.gz'))
    files = list(tsv_files) + list(gz_files)

    for filepath in files:
        dataset_name = extract_dataset_name(filepath=filepath)
        if not dataset_name:
            continue

        # Determine original split from filename
        filename = filepath.name
        if '.train' in filename:
            original_split = 'train'
        elif '.dev' in filename:
            original_split = 'dev'
        elif '.test' in filename:
            original_split = 'test'
        else:
            original_split = 'train'  # default

        try:
            # Handle gzipped files
            if filename.endswith('.gz'):
                file_handle = gzip.open(filepath, mode='rt', encoding='utf-8')
            else:
                file_handle = open(filepath, mode='r', encoding='utf-8')

            with file_handle as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split('\t')
                    if len(parts) < 2:
                        continue

                    date_str = parts[0]
                    date_obj = parse_date(date_str=date_str)

                    # Only count articles with dates
                    if date_obj:
                        original_splits[original_split][dataset_name] += 1
        except OSError as e:
            print(f'Error reading {filepath}: {e}', file=sys.stderr)

    return original_splits


def read_new_splits(
    split_dir: Path
) -> dict[str, Counter[str]]:
    """
    Read new splits from output directory to get new distribution.

    :param split_dir: Directory containing new split files (e.g., global_chronological)
    :return: Dict mapping 'train'/'dev'/'test' to Counter of dataset counts
    """
    new_splits: dict[str, Counter[str]] = {
        'train': Counter(),
        'dev': Counter(),
        'test': Counter()
    }

    # Look for split files: train.analysis.jsonl.gz, dev.analysis.jsonl.gz, test.analysis.jsonl.gz
    # or per-dataset files: {dataset}.train.analysis.jsonl.gz, etc.
    split_files = sorted(split_dir.glob('*.analysis.jsonl.gz'))

    for filepath in split_files:
        filename = filepath.name

        # Determine split type from filename
        if '.train' in filename:
            split_type = 'train'
        elif '.dev' in filename:
            split_type = 'dev'
        elif '.test' in filename:
            split_type = 'test'
        elif filename.startswith('train.'):
            split_type = 'train'
        elif filename.startswith('dev.'):
            split_type = 'dev'
        elif filename.startswith('test.'):
            split_type = 'test'
        else:
            continue

        # Extract dataset name (might be None for global splits)
        dataset_name = extract_dataset_name(filepath=filepath)

        try:
            with gzip.open(filepath, mode='rt', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split('\t')
                    if len(parts) < 2:
                        continue

                    date_str = parts[0]
                    date_obj = parse_date(date_str=date_str)

                    # Only count articles with dates
                    if date_obj:
                        # For per-dataset splits, use the dataset from filename
                        # For global splits, extract dataset from the line if possible
                        if dataset_name:
                            new_splits[split_type][dataset_name] += 1
                        else:
                            # Try to extract dataset from article_id or use a default
                            # For now, we'll need to track this differently
                            # This is a limitation - we need the dataset info
                            # Let's assume we can get it from the article_id pattern
                            article_id = parts[1]
                            # Try to infer dataset from article_id (e.g., de_dpa_123 -> de_dpa_iptc)
                            # This is a heuristic and might not work for all cases
                            # For global splits, we might need to store dataset info differently
                            pass
        except OSError as e:
            print(f'Error reading {filepath}: {e}', file=sys.stderr)

    return new_splits


def read_new_splits_with_dataset_info(
    split_dir: Path
) -> dict[str, Counter[str]]:
    """
    Read new splits from output directory, handling both global and per-dataset splits.

    :param split_dir: Directory containing new split files
    :return: Dict mapping 'train'/'dev'/'test' to Counter of dataset counts
    """
    new_splits: dict[str, Counter[str]] = {
        'train': Counter(),
        'dev': Counter(),
        'test': Counter()
    }

    # Look for split files
    split_files = sorted(split_dir.glob('*.analysis.jsonl.gz'))

    for filepath in split_files:
        filename = filepath.name

        # Determine split type from filename
        if '.train' in filename:
            split_type = 'train'
        elif '.dev' in filename:
            split_type = 'dev'
        elif '.test' in filename:
            split_type = 'test'
        elif filename.startswith('train.'):
            split_type = 'train'
        elif filename.startswith('dev.'):
            split_type = 'dev'
        elif filename.startswith('test.'):
            split_type = 'test'
        else:
            continue

        # Extract dataset name
        dataset_name = extract_dataset_name(filepath=filepath)

        # If we have a dataset name (per-dataset split), count articles for that dataset
        # If no dataset name (global split), we need to read the original input to map
        # article_ids to datasets, or we can skip detailed per-dataset counting for global splits
        if dataset_name:
            # Per-dataset split: count all articles as belonging to this dataset
            try:
                with gzip.open(filepath, mode='rt', encoding='utf-8') as f:
                    count = sum(1 for line in f if line.strip())
                    new_splits[split_type][dataset_name] = count
            except OSError as e:
                print(f'Error reading {filepath}: {e}', file=sys.stderr)
        else:
            # Global split: we need to map article_ids back to datasets
            # For now, we'll need to read from original input or store mapping
            # This is a limitation - let's read from original input directory
            pass

    return new_splits


def read_new_splits_enhanced(
    split_dir: Path,
    input_dir: Path
) -> dict[str, Counter[str]]:
    """
    Read new splits and map article_ids to datasets using original input files.

    :param split_dir: Directory containing new split files
    :param input_dir: Directory containing original input files for mapping
    :return: Dict mapping 'train'/'dev'/'test' to Counter of dataset counts
    """
    # First, create a mapping from article_id to dataset from original files
    article_to_dataset: dict[str, str] = {}

    tsv_files = sorted(input_dir.glob('*.jsonl.tsv'))
    gz_files = sorted(input_dir.glob('*.jsonl.gz'))
    files = list(tsv_files) + list(gz_files)

    for filepath in files:
        dataset_name = extract_dataset_name(filepath=filepath)
        if not dataset_name:
            continue

        try:
            filename = filepath.name
            if filename.endswith('.gz'):
                file_handle = gzip.open(filepath, mode='rt', encoding='utf-8')
            else:
                file_handle = open(filepath, mode='r', encoding='utf-8')

            with file_handle as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split('\t')
                    if len(parts) < 2:
                        continue

                    article_id = parts[1]
                    article_to_dataset[article_id] = dataset_name
        except OSError as e:
            print(f'Error reading {filepath}: {e}', file=sys.stderr)

    # Now read new splits and count by dataset
    new_splits: dict[str, Counter[str]] = {
        'train': Counter(),
        'dev': Counter(),
        'test': Counter()
    }

    split_files = sorted(split_dir.glob('*.analysis.jsonl.gz'))

    for filepath in split_files:
        filename = filepath.name

        # Determine split type from filename
        if '.train' in filename:
            split_type = 'train'
        elif '.dev' in filename:
            split_type = 'dev'
        elif '.test' in filename:
            split_type = 'test'
        elif filename.startswith('train.'):
            split_type = 'train'
        elif filename.startswith('dev.'):
            split_type = 'dev'
        elif filename.startswith('test.'):
            split_type = 'test'
        else:
            continue

        # Check if this is a per-dataset split (has dataset name in filename)
        dataset_name = extract_dataset_name(filepath=filepath)

        try:
            with gzip.open(filepath, mode='rt', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split('\t')
                    if len(parts) < 2:
                        continue

                    article_id = parts[1]

                    # If per-dataset split, use dataset from filename
                    # Otherwise, look up in mapping
                    if dataset_name:
                        new_splits[split_type][dataset_name] += 1
                    else:
                        # Global split: look up dataset from article_id
                        mapped_dataset = article_to_dataset.get(article_id)
                        if mapped_dataset:
                            new_splits[split_type][mapped_dataset] += 1
        except OSError as e:
            print(f'Error reading {filepath}: {e}', file=sys.stderr)

    return new_splits


def count_datasets(
    articles: list[tuple]
) -> Counter[str]:
    """
    Count articles per dataset.

    :param articles: List of article tuples (format depends on context)
    :return: Counter with dataset names as keys and counts as values
    """
    counter: Counter[str] = Counter()
    for article in articles:
        if len(article) >= 3:
            dataset = article[2]
            counter[dataset] += 1
    return counter


def create_bar_chart(
    original_counter: Counter[str],
    new_counter: Counter[str],
    title: str,
    ax: plt.Axes,
    dataset_order: list[str]
) -> None:
    """
    Create a bar chart comparing original and new splits.

    :param original_counter: Counter with dataset names and counts for original
    :param new_counter: Counter with dataset names and counts for new split
    :param title: Title for the bar chart
    :param ax: Matplotlib axes to plot on
    :param dataset_order: Ordered list of all datasets
    """
    # Use all datasets to ensure consistent x-axis across all charts
    if not dataset_order:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax.set_title(title, fontsize=12, fontweight='bold')
        return

    # Prepare data for all datasets (use 0 if no data)
    original_counts = [
        original_counter.get(ds, 0) for ds in dataset_order
    ]
    new_counts = [new_counter.get(ds, 0) for ds in dataset_order]

    # Calculate totals for percentage calculation
    original_total = sum(original_counts)
    new_total = sum(new_counts)

    # Convert counts to percentages
    original_percentages = [
        (count / original_total * 100) if original_total > 0 else 0.0
        for count in original_counts
    ]
    new_percentages = [
        (count / new_total * 100) if new_total > 0 else 0.0
        for count in new_counts
    ]

    # Set up bar positions
    x = range(len(dataset_order))
    width = 0.35

    # Create bars with single colors for original and new
    bars1 = ax.bar(
        [i - width/2 for i in x], original_percentages, width,
        label='Original', color='steelblue', edgecolor='black', linewidth=1.5
    )
    bars2 = ax.bar(
        [i + width/2 for i in x], new_percentages, width,
        label='New', color='coral', edgecolor='black', linewidth=1.5
    )

    # Add value labels on bars (only if height > 0)
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom',
                    fontsize=8
                )

    # Customize axes
    ax.set_xlabel('Dataset', fontsize=11)
    ax.set_ylabel('Percentage of Articles', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_order, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)


def create_visualization(
    original_splits: dict[str, Counter[str]],
    new_splits: dict[str, Counter[str]],
    output_path: Path,
    split_type_name: str = 'New'
) -> None:
    """
    Create 3 bar charts comparing original and new splits for train, dev,
    and test.

    :param original_splits: Dict mapping split names to dataset counters
    :param new_splits: Dict mapping split names to dataset counters
    :param output_path: Path where to save the visualization
    :param split_type_name: Name for the new split type (e.g., 'Global' or 'Per-Dataset')
    """
    # Collect all unique datasets and create consistent color mapping
    all_datasets: set[str] = set()
    for counter in list(original_splits.values()) + list(new_splits.values()):
        all_datasets.update(counter.keys())

    # Sort datasets for consistent order
    dataset_order = sorted(all_datasets)

    _, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Create bar charts for each split
    create_bar_chart(
        original_counter=original_splits['train'],
        new_counter=new_splits['train'],
        title=f'Train Split: Original vs {split_type_name} (80% oldest)',
        ax=axes[0],
        dataset_order=dataset_order
    )
    create_bar_chart(
        original_counter=original_splits['dev'],
        new_counter=new_splits['dev'],
        title=f'Dev Split: Original vs {split_type_name} (10% middle)',
        ax=axes[1],
        dataset_order=dataset_order
    )
    create_bar_chart(
        original_counter=original_splits['test'],
        new_counter=new_splits['test'],
        title=f'Test Split: Original vs {split_type_name} (10% most recent)',
        ax=axes[2],
        dataset_order=dataset_order
    )

    plt.suptitle(
        f'Dataset Distribution: Original vs {split_type_name} Splits',
        fontsize=16, fontweight='bold', y=1.02
    )
    plt.tight_layout()

    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'  -> Saved visualization to: {output_path}')
    except OSError as e:
        print(f'  -> Error saving {output_path}: {e}', file=sys.stderr)
    finally:
        plt.close()


def main() -> None:
    """Main entry point for the script."""
    argparser = argparse.ArgumentParser(
        description=(
            'Visualize distributions of train/dev/test splits '
            'comparing original and new splits'
        )
    )
    argparser.add_argument(
        'input',
        nargs='?',
        default='.',
        help=(
            'Input directory containing original .jsonl.tsv and .jsonl.gz files '
            '(default: current directory)'
        )
    )
    argparser.add_argument(
        '-s',
        '--splits',
        type=str,
        default=None,
        help=(
            'Directory containing new split files '
            '(default: looks for global_chronological and per_dataset_chronological '
            'subdirectories in input directory)'
        )
    )
    argparser.add_argument(
        '-o',
        '--output',
        type=str,
        default=None,
        help='Output directory for visualizations (default: same as splits directory)'
    )
    args = argparser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f'Error: {input_dir} is not a directory', file=sys.stderr)
        sys.exit(1)

    # Determine splits directory
    if args.splits:
        splits_base_dir = Path(args.splits)
    else:
        splits_base_dir = input_dir

    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = splits_base_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read original splits
    print(f'Reading original splits from {input_dir}...')
    original_splits = read_original_splits(input_dir=input_dir)

    # Process global chronological splits
    global_split_dir = splits_base_dir / 'global_chronological'
    if global_split_dir.is_dir():
        print(f'\nReading global chronological splits from {global_split_dir}...')
        global_new_splits = read_new_splits_enhanced(
            split_dir=global_split_dir,
            input_dir=input_dir
        )

        print('Creating visualization for global chronological splits...')
        viz_path = output_dir / 'split_distributions_global.png'
        create_visualization(
            original_splits=original_splits,
            new_splits=global_new_splits,
            output_path=viz_path,
            split_type_name='Global'
        )
    else:
        print(f'Warning: {global_split_dir} not found, skipping global splits')

    # Process per-dataset chronological splits
    per_dataset_split_dir = splits_base_dir / 'per_dataset_chronological'
    if per_dataset_split_dir.is_dir():
        print(f'\nReading per-dataset chronological splits from {per_dataset_split_dir}...')
        per_dataset_new_splits = read_new_splits_enhanced(
            split_dir=per_dataset_split_dir,
            input_dir=input_dir
        )

        print('Creating visualization for per-dataset chronological splits...')
        viz_path = output_dir / 'split_distributions_per_dataset.png'
        create_visualization(
            original_splits=original_splits,
            new_splits=per_dataset_new_splits,
            output_path=viz_path,
            split_type_name='Per-Dataset'
        )
    else:
        print(f'Warning: {per_dataset_split_dir} not found, skipping per-dataset splits')

    print('\nDone!')


if __name__ == '__main__':
    main()

