#!/usr/bin/env python3
"""
Filter articles with valid dates from .jsonl.gz files.
"""
import argparse
import gzip
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Since this is a CLI script, print is acceptable for output
# For general-purpose tools, logging should be used instead


def parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """
    Parse ISO date string to datetime object.

    :param date_str: Date string in format 'YYYY-MM-DD' or ISO format
    :return: Datetime object or None if date_str is invalid
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
        return dt
    except (ValueError, AttributeError):
        return None


def has_valid_date(data: dict) -> bool:
    """
    Check if article has valid date in metadata.date.

    :param data: JSON object representing an article
    :return: True if metadata.date exists and is valid, False otherwise
    """
    metadata = data.get('metadata', {})
    if not metadata:
        return False

    date_str = metadata.get('date')
    if not date_str:
        return False

    date_obj = parse_date(date_str=date_str)
    return date_obj is not None


def filter_file(input_path: Path, output_path: Path) -> tuple[int, int]:
    """
    Filter articles with valid dates from a .jsonl.gz file.

    :param input_path: Path to input .jsonl.gz file
    :param output_path: Path to output .jsonl.gz file
    :return: Tuple of (total_count, filtered_count)
    """
    total_count = 0
    filtered_count = 0

    try:
        with gzip.open(input_path, mode='rt', encoding='utf-8') as f_in, \
             gzip.open(output_path, mode='wt', encoding='utf-8') as f_out:

            for line in f_in:
                line = line.strip()
                if not line:
                    continue

                total_count += 1
                try:
                    data = json.loads(line)
                    if has_valid_date(data=data):
                        f_out.write(line + '\n')
                        filtered_count += 1
                except json.JSONDecodeError:
                    # Skip invalid JSON lines
                    continue

    except OSError as e:
        print(f'Error processing {input_path}: {e}', file=sys.stderr)
        return (total_count, filtered_count)

    return (total_count, filtered_count)


def main() -> None:
    """Main entry point for the script."""
    argparser = argparse.ArgumentParser(
        description='Filter articles with valid dates from .jsonl.gz files'
    )
    argparser.add_argument(
        'input_dir',
        type=str,
        help='Input directory containing .jsonl.gz files'
    )
    argparser.add_argument(
        '-o',
        '--output',
        type=str,
        required=True,
        help='Output directory for filtered files'
    )
    args = argparser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output)

    if not input_dir.is_dir():
        print(f'Error: {input_dir} is not a directory', file=sys.stderr)
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all .jsonl.gz files
    files = sorted(input_dir.glob('*.jsonl.gz'))

    if not files:
        print(
            f'No .jsonl.gz files found in {input_dir}',
            file=sys.stderr
        )
        sys.exit(1)

    print(f'Found {len(files)} file(s) to process\n')

    total_articles = 0
    total_filtered = 0

    for filepath in files:
        output_path = output_dir / filepath.name
        print(f'Processing: {filepath.name}')

        total, filtered = filter_file(
            input_path=filepath, output_path=output_path
        )

        total_articles += total
        total_filtered += filtered

        print(
            f'  -> Total: {total:,}, Filtered: {filtered:,} '
            f'({filtered/total*100:.1f}%)'
        )
        print(f'  -> Saved to: {output_path}\n')

    print(f'Summary:')
    print(f'  Total articles processed: {total_articles:,}')
    print(f'  Articles with valid dates: {total_filtered:,}')
    print(f'  Percentage: {total_filtered/total_articles*100:.1f}%')


if __name__ == '__main__':
    main()

