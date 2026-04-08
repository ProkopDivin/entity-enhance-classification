#!/usr/bin/env python3
"""
Analyze .jsonl.gz files and print statistics about metadata.date field.
"""
import argparse
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
from typing import Optional, Sequence, Tuple

# Since this is a CLI script, print is acceptable for output
# For general-purpose tools, logging should be used instead


def extract_dataset_name(filename: str, *, no_group: bool = False) -> str:
    """
    Extract dataset name from filename (e.g. cs_mafra_iptc from
    cs_mafra_iptc.dev_all.analysis.jsonl.gz).

    :param filename: Name of the .jsonl.gz file
    :return: Dataset name
    """
    name = filename
    if name.endswith('.gz'):
        name = name[:-3]
    if name.endswith('.jsonl'):
        name = name[:-6]
    name = name.replace('.analysis', '')
    if no_group:
        return name

    for suffix in (
        '.train_all', '.dev_all', '.test_all',
        '.train_smallpp', '.dev_smallpp', '.test_smallpp',
        '.train_medium', '.dev_medium', '.test_medium',
        '.train', '.dev', '.test',
    ):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """
    Parse ISO date string to datetime, ensuring timezone-aware.

    :param date_str: ISO format date string (e.g., '2021-09-19T00:00:00Z')
    :return: Timezone-aware datetime object or None if parsing fails
    """
    if not date_str:
        return None
    try:
        # Handle ISO format with timezone: '2021-09-19T00:00:00Z'
        if date_str.endswith('Z'):
            date_str = date_str[:-1] + '+00:00'
        else:
            date_str = date_str.replace('Z', '+00:00')

        dt = datetime.fromisoformat(date_str)

        # Make sure datetime is timezone-aware (assume UTC if naive)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        return dt
    except (ValueError, AttributeError):
        return None


def analyze_file(filepath: Path) -> Tuple[
    str, int, int, Optional[str], Optional[str],
    Sequence[Tuple[Optional[datetime], Optional[str]]]
]:
    """
    Analyze a single .jsonl.gz file.

    :param filepath: Path to the .jsonl.gz file to analyze
    :return: Tuple containing (filename, total_count, missing_count,
             min_date, max_date, date_id_list)
             where date_id_list contains tuples of (date, id) in order
    """
    print(f'Analyzing file: {filepath}')
    filename = filepath.name
    total_count = 0
    missing_count = 0
    dates = []
    # All (date, id) tuples in order
    date_id_list: list[Tuple[Optional[datetime], Optional[str]]] = []

    try:
        with gzip.open(filepath, mode='rt', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                total_count += 1
                try:
                    data = json.loads(line)
                    doc_id = data.get('id')
                    metadata = data.get('metadata', {})
                    date_str = metadata.get('date') if metadata else None

                    if not date_str:
                        missing_count += 1
                        date_id_list.append((None, doc_id))
                    else:
                        parsed_date = parse_date(date_str=date_str)
                        if parsed_date:
                            dates.append(parsed_date)
                            date_id_list.append((parsed_date, doc_id))
                        else:
                            missing_count += 1
                            date_id_list.append((None, doc_id))
                except json.JSONDecodeError:
                    missing_count += 1
                    date_id_list.append((None, None))
    except OSError as e:
        print(f'Error processing {filename}: {e}', file=sys.stderr)
        return (filename, 0, 0, None, None, [])

    # Format dates as YYYY-MM-DD for display
    min_date_str = min(dates).strftime('%Y-%m-%d') if dates else None
    max_date_str = max(dates).strftime('%Y-%m-%d') if dates else None

    return (
        filename, total_count, missing_count, min_date_str, max_date_str,
        date_id_list
    )


def write_date_list_file(
    filepath: Path,
    date_id_list: Sequence[Tuple[Optional[datetime], Optional[str]]]
) -> None:
    """
    Write sorted date and id list to a .txt file with two columns.

    :param filepath: Path to the input .jsonl.gz file
    :param date_id_list: Sequence of (date, id) tuples to write
    """
    # Create output filename: replace .jsonl.gz with .txt
    output_file = filepath.with_suffix('.tsv')

    # Sort by date: None values go to the end
    min_dt = datetime.min.replace(tzinfo=timezone.utc)
    sorted_date_ids = sorted(
        date_id_list,
        key=lambda x: (x[0] is None, x[0] or min_dt)
    )

    try:
        with open(output_file, mode='w', encoding='utf-8') as f:
            for date, doc_id in sorted_date_ids:
                if date is None:
                    date_str = 'None'
                else:
                    date_str = date.strftime('%Y-%m-%d')
                id_str = str(doc_id) if doc_id else 'None'
                f.write(f'{date_str}\t{id_str}\n')
        print(f'  -> Wrote date list to: {output_file}')
    except OSError as e:
        print(f'  -> Error writing {output_file}: {e}', file=sys.stderr)


def main() -> None:
    """Main entry point for the script."""
    argparser = argparse.ArgumentParser(
        description=(
            'Analyze .jsonl.gz files and print statistics about '
            'metadata.date field'
        )
    )
    argparser.add_argument(
        'directory',
        nargs='?',
        default='.',
        help=(
            'Directory containing .jsonl.gz files to analyze '
            '(default: current directory)'
        )
    )
    argparser.add_argument(
        '--no-group',
        action='store_true',
        help=(
            'Do not merge train/dev/test variants into one dataset row. '
            'Keep split suffixes (train/dev/test) as separate rows.'
        )
    )
    args = argparser.parse_args()

    directory = Path(args.directory)

    # Find all .jsonl.gz files
    files = sorted(directory.glob('*.jsonl.gz'))

    if not files:
        print(f'No .jsonl.gz files found in {directory}', file=sys.stderr)
        sys.exit(1)

    # Analyze all files
    results = []
    total_items_sum = 0
    total_missing_sum = 0

    for filepath in files:
        result = analyze_file(filepath=filepath)
        filename, total, missing, min_date, max_date, date_id_list = result
        results.append((filename, total, missing, min_date, max_date))
        total_items_sum += total
        total_missing_sum += missing

        # Write date list file
        write_date_list_file(filepath=filepath, date_id_list=date_id_list)

    # Aggregate results by dataset (one row per dataset)
    # dataset -> (total_items, missing, min_date, max_date)
    by_dataset: dict[str, Tuple[int, int, Optional[str], Optional[str]]] = defaultdict(
        lambda: (0, 0, None, None)
    )

    for filename, total, missing, min_date, max_date in results:
        dataset = extract_dataset_name(filename, no_group=args.no_group)
        prev_total, prev_missing, prev_min, prev_max = by_dataset[dataset]
        new_total = prev_total + total
        new_missing = prev_missing + missing

        # Min date: earliest non-None date
        dates_min = [d for d in (prev_min, min_date) if d]
        new_min = min(dates_min) if dates_min else None

        # Max date: latest non-None date
        dates_max = [d for d in (prev_max, max_date) if d]
        new_max = max(dates_max) if dates_max else None

        by_dataset[dataset] = (new_total, new_missing, new_min, new_max)

    # Write markdown table to stats.md file (one row per dataset)
    stats_file = directory / 'stats.md'
    try:
        with open(stats_file, mode='w', encoding='utf-8') as f:
            f.write('## Statistics Summary\n\n')
            header = (
                '| Dataset | Total Items | Missing/None Date | '
                'Min Date | Max Date |\n'
            )
            f.write(header)
            separator = (
                '|--------|-------------|-------------------|'
                '----------|----------|\n'
            )
            f.write(separator)

            for dataset in sorted(by_dataset.keys()):
                total, missing, min_date, max_date = by_dataset[dataset]
                min_date_display = min_date if min_date else 'N/A'
                max_date_display = max_date if max_date else 'N/A'
                f.write(
                    f'| {dataset} | {total:,} | {missing:,} | '
                    f'{min_date_display} | {max_date_display} |\n'
                )

            # Write totals row
            f.write(
                f'| **TOTAL** | **{total_items_sum:,}** | '
                f'**{total_missing_sum:,}** | - | - |\n'
            )
        print(f'\n  -> Wrote statistics to: {stats_file}')
    except OSError as e:
        print(f'  -> Error writing {stats_file}: {e}', file=sys.stderr)


if __name__ == '__main__':
    main()
