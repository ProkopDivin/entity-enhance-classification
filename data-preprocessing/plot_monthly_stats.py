#!/usr/bin/env python3
"""
Create bar graphs showing monthly article counts from .jsonl.tsv files.
"""
import argparse
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

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


def read_tsv_file(filepath: Path) -> Counter[str]:
    """
    Read .jsonl.tsv file and count articles per month.

    :param filepath: Path to the .jsonl.tsv file
    :return: Counter with month strings (YYYY-MM) as keys and counts as values
    """
    month_counts: Counter[str] = Counter()

    try:
        with open(filepath, mode='r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split('\t')
                if len(parts) < 1:
                    continue

                date_str = parts[0]
                date_obj = parse_date(date_str=date_str)

                if date_obj:
                    # Format as YYYY-MM for month
                    month_key = date_obj.strftime('%Y-%m')
                    month_counts[month_key] += 1

    except OSError as e:
        print(f'Error reading {filepath}: {e}', file=sys.stderr)
        return Counter()

    return month_counts


def create_bar_graph(
    month_counts: Counter[str],
    output_path: Path,
    title: str
) -> None:
    """
    Create and save a bar graph showing monthly article counts.

    :param month_counts: Counter with month strings (YYYY-MM) and counts
    :param output_path: Path where to save the graph
    :param title: Title for the graph
    """
    if not month_counts:
        print(f'No data to plot for {title}', file=sys.stderr)
        return

    # Sort months chronologically
    sorted_months = sorted(month_counts.keys())
    counts = [month_counts[month] for month in sorted_months]

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(sorted_months)), counts, color='steelblue')
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Number of Articles', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')

    # Show only every nth month label to avoid overlap
    num_months = len(sorted_months)
    if num_months > 24:
        step = max(1, num_months // 20)  # Show ~20 labels max
    elif num_months > 12:
        step = 2
    else:
        step = 1

    tick_positions = list(range(0, num_months, step))
    tick_labels = [sorted_months[i] for i in tick_positions]

    plt.xticks(
        tick_positions, tick_labels, rotation=45, ha='right'
    )
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    # Save the figure
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'  -> Saved graph to: {output_path}')
    except OSError as e:
        print(f'  -> Error saving {output_path}: {e}', file=sys.stderr)
    finally:
        plt.close()


def process_file(filepath: Path, output_dir: Optional[Path] = None) -> None:
    """
    Process a single .jsonl.tsv file and create its bar graph.

    :param filepath: Path to the input .jsonl.tsv file
    :param output_dir: Directory where to save the graph
                      (default: same as input)
    """
    print(f'Processing file: {filepath}')

    # Read and count articles per month
    month_counts = read_tsv_file(filepath=filepath)

    if not month_counts:
        print(f'  -> No valid dates found in {filepath}', file=sys.stderr)
        return

    # Determine output path
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filepath.with_suffix('.png').name
    else:
        output_path = filepath.with_suffix('.png')

    # Create title from filename, removing .analysis.jsonl
    title_base = filepath.stem.replace('.analysis.jsonl', '')
    title = f'Monthly Article Count: {title_base}'

    # Create and save the graph
    create_bar_graph(
        month_counts=month_counts,
        output_path=output_path,
        title=title
    )


def extract_dataset_name(filepath: Path) -> Optional[str]:
    """
    Extract dataset name from filepath.

    :param filepath: Path to .jsonl.tsv file
    :return: Dataset name (e.g., 'en_bbc_iptc' from
             'en_bbc_iptc.dev_all.analysis.jsonl.tsv')
    """
    name = filepath.stem.replace('.analysis.jsonl', '')
    # Remove .train_all, .dev_all, .test_all, etc.
    for suffix in [
        '.train_all', '.dev_all', '.test_all', '.train', '.dev', '.test'
    ]:
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def create_combined_line_graph(
    dataset_counts: dict[str, Counter[str]],
    output_path: Path
) -> None:
    """
    Create a combined line graph showing monthly counts for all datasets.

    :param dataset_counts: Dictionary mapping dataset names to month counters
    :param output_path: Path where to save the graph
    """
    if not dataset_counts:
        print('No data to plot for combined graph', file=sys.stderr)
        return

    # Get all unique months across all datasets
    all_months: set[str] = set()
    for counter in dataset_counts.values():
        all_months.update(counter.keys())
    sorted_months = sorted(all_months)

    # Create the plot
    plt.figure(figsize=(14, 8))

    # Generate colors for each dataset
    colors = plt.cm.tab20(np.linspace(0, 1, len(dataset_counts)))

    for (dataset_name, month_counts), color in zip(
        dataset_counts.items(), colors
    ):
        counts = np.array([
            month_counts.get(month, 0) for month in sorted_months
        ], dtype=float)
        # Replace zeros with NaN so the line breaks
        counts[counts == 0] = np.nan
        plt.plot(
            sorted_months, counts, marker='o', label=dataset_name,
            linewidth=2, markersize=4, color=color
        )

    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Number of Articles', fontsize=12)
    plt.title(
        'Monthly Article Count: Combined Datasets (Train+Dev+Test)',
        fontsize=14, fontweight='bold'
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(alpha=0.3, linestyle='--')

    # Show only every nth month label to avoid overlap
    num_months = len(sorted_months)
    if num_months > 24:
        step = max(1, num_months // 20)
    elif num_months > 12:
        step = 2
    else:
        step = 1

    tick_positions = list(range(0, num_months, step))
    tick_labels = [sorted_months[i] for i in tick_positions]

    plt.xticks(tick_positions, tick_labels, rotation=45, ha='right')
    plt.tight_layout()

    # Save the figure
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'  -> Saved combined graph to: {output_path}')
    except OSError as e:
        print(f'  -> Error saving {output_path}: {e}', file=sys.stderr)
    finally:
        plt.close()


def main() -> None:
    """Main entry point for the script."""
    argparser = argparse.ArgumentParser(
        description=(
            'Create bar graphs showing monthly article counts from '
            '.jsonl.tsv files'
        )
    )
    argparser.add_argument(
        'input',
        nargs='?',
        default='.',
        help=(
            'Input file or directory containing .jsonl.tsv files '
            '(default: current directory)'
        )
    )
    argparser.add_argument(
        '-o',
        '--output',
        type=str,
        default=None,
        help='Output directory for graphs (default: same as input)'
    )
    args = argparser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output) if args.output else None

    # Collect files to process
    files_to_process = []

    if input_path.is_file():
        if input_path.suffix == '.tsv' and '.jsonl.tsv' in input_path.name:
            files_to_process.append(input_path)
        else:
            print(
                f'Error: {input_path} is not a .jsonl.tsv file',
                file=sys.stderr
            )
            sys.exit(1)
    elif input_path.is_dir():
        files_to_process = sorted(input_path.glob('*.jsonl.tsv'))
    else:
        print(f'Error: {input_path} does not exist', file=sys.stderr)
        sys.exit(1)

    if not files_to_process:
        print(
            f'No .jsonl.tsv files found in {input_path}',
            file=sys.stderr
        )
        sys.exit(1)

    print(f'Found {len(files_to_process)} file(s) to process\n')

    # Process each file and collect data for combined graph
    dataset_month_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for filepath in files_to_process:
        process_file(filepath=filepath, output_dir=output_dir)

        # Read data for combined graph
        month_counts = read_tsv_file(filepath=filepath)
        if month_counts:
            dataset_name = extract_dataset_name(filepath=filepath)
            if dataset_name:
                # Sum counts for this dataset
                for month, count in month_counts.items():
                    dataset_month_counts[dataset_name][month] += count

    print(f'\nProcessed {len(files_to_process)} file(s)')

    # Create combined line graph if we have data
    if dataset_month_counts:
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            combined_path = output_dir / 'combined_datasets.png'
        else:
            combined_path = input_path / 'combined_datasets.png'
            if input_path.is_file():
                combined_path = input_path.parent / 'combined_datasets.png'

        create_combined_line_graph(
            dataset_counts=dataset_month_counts,
            output_path=combined_path
        )


if __name__ == '__main__':
    main()
