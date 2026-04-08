#!/usr/bin/env python3
"""
Analyze dataset distributions across Silver vs. Other corpora files.

Loads all-corpora-*.csv (Group B) and silver-corpora-*.csv (Group A) from
the silver_dataset directory, then reports:
  - average number of categories per document in each group
  - top-20 most frequent categories per group
  - per-corpus summary (by metadata.corpusName) of average category count
"""
import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import pandas as pd

LOG = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / 'silver_dataset'


def _bump_csv_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            break
        except OverflowError:
            limit //= 10


def _extract_corpus_name(meta_str: str) -> str:
    """Extract corpusName from the JSON metadata column."""
    if not meta_str:
        return ''
    try:
        return json.loads(meta_str).get('corpusName', '')
    except (json.JSONDecodeError, TypeError):
        return ''


def load_grouped_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load CSVs from *data_dir* and split into Silver (Group A) and Others (Group B).

    :param data_dir: directory containing the CSV files
    :return: (silver_df, others_df)
    """
    silver_frames: list[pd.DataFrame] = []
    other_frames: list[pd.DataFrame] = []

    for path in sorted(data_dir.glob('*.csv')):
        name = path.name
        is_silver = name.startswith('silver-corpora')
        is_all = name.startswith('all-corpora')
        if not (is_silver or is_all):
            continue

        LOG.info('Loading %s ...', name)
        df = pd.read_csv(path, usecols=['id', 'cats', 'metadata'], dtype=str, keep_default_na=False)
        df['corpus_name'] = df['metadata'].apply(_extract_corpus_name)
        df.drop(columns=['metadata'], inplace=True)
        df['_source'] = name
        if is_silver:
            silver_frames.append(df)
        else:
            other_frames.append(df)

    silver_df = pd.concat(silver_frames, ignore_index=True) if silver_frames else pd.DataFrame(columns=['id', 'cats'])
    others_df = pd.concat(other_frames, ignore_index=True) if other_frames else pd.DataFrame(columns=['id', 'cats'])

    LOG.info('Silver documents: %d   Other documents: %d', len(silver_df), len(others_df))
    return silver_df, others_df


def parse_cats(cats_series: pd.Series) -> pd.Series:
    """
    Split pipe-separated category strings into lists, treating empty/missing values as empty lists.

    :param cats_series: Series of raw category strings
    :return: Series of lists of category IDs
    """
    return cats_series.apply(lambda v: [c.strip() for c in v.split('|') if c.strip()] if isinstance(v, str) and v.strip() else [])


def avg_cats_per_doc(cat_lists: pd.Series) -> float:
    """
    Calculate the average number of categories per document.

    :param cat_lists: Series where each element is a list of category IDs
    :return: mean count
    """
    counts = cat_lists.apply(len)
    return counts.mean() if len(counts) else 0.0


def top_n_categories(cat_lists: pd.Series, *, n: int = 20) -> pd.DataFrame:
    """
    Identify top-N most frequent categories.

    :param cat_lists: Series where each element is a list of category IDs
    :param n: number of top categories to return
    :return: DataFrame with columns [category, count, pct]
    """
    total_docs = len(cat_lists)
    exploded = cat_lists.explode().dropna()
    if exploded.empty:
        return pd.DataFrame(columns=['category', 'count', 'pct'])

    freq = exploded.value_counts().head(n).reset_index()
    freq.columns = ['category', 'count']
    freq['pct'] = (freq['count'] / total_docs * 100).round(2)
    return freq


def corpus_summary(df: pd.DataFrame, cat_lists: pd.Series) -> pd.DataFrame:
    """
    Per-corpus summary: number of documents and average categories per document.

    :param df: DataFrame with a 'corpus_name' column
    :param cat_lists: corresponding parsed category lists
    :return: summary DataFrame with columns [corpus_name, num_docs, avg_cats]
    """
    tmp = df[['corpus_name']].copy()
    tmp['n_cats'] = cat_lists.apply(len)
    summary = tmp.groupby('corpus_name', sort=False).agg(
        num_docs=('n_cats', 'size'),
        avg_cats=('n_cats', 'mean'),
    ).reset_index()
    summary['avg_cats'] = summary['avg_cats'].round(2)
    return summary.sort_values('num_docs', ascending=False)


def print_section(title: str) -> None:
    width = 70
    print(f'\n{"=" * width}')
    print(f' {title}')
    print(f'{"=" * width}')


def report_group(name: str, df: pd.DataFrame, *, top_n: int = 20) -> None:
    """Print a full report for one group."""
    print_section(f'Group: {name}  ({len(df):,} documents)')

    cat_lists = parse_cats(df['cats'])

    avg = avg_cats_per_doc(cat_lists)
    print(f'\n  Average categories per document: {avg:.2f}')

    top = top_n_categories(cat_lists, n=top_n)
    print(f'\n  Top {top_n} categories:')
    print(f'  {"Category":<20} {"Count":>10} {"% of docs":>10}')
    print(f'  {"-" * 20} {"-" * 10} {"-" * 10}')
    for _, row in top.iterrows():
        print(f'  {row["category"]:<20} {row["count"]:>10,} {row["pct"]:>9.2f}%')

    summary = corpus_summary(df, cat_lists)
    print(f'\n  Per-corpus summary ({summary.shape[0]} corpora):')
    print(f'  {"Corpus":<40} {"#docs":>8} {"avg_cats":>10}')
    print(f'  {"-" * 40} {"-" * 8} {"-" * 10}')
    for _, row in summary.iterrows():
        print(f'  {str(row["corpus_name"]):<40} {row["num_docs"]:>8,} {row["avg_cats"]:>10.2f}')

    macro_avg = summary['avg_cats'].mean()
    print(f'\n  Macro-average cats across corpora: {macro_avg:.2f}')


def main() -> None:
    argparser = argparse.ArgumentParser(
        description='Analyze category distributions across Silver vs. Other corpora.',
    )
    argparser.add_argument(
        '-d', '--data-dir', type=str, default=None,
        help=f'Directory with CSV files (default: {DATA_DIR})',
    )
    argparser.add_argument(
        '-n', '--top-n', type=int, default=20,
        help='Number of top categories to show per group (default: 20)',
    )
    argparser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Enable verbose (DEBUG) logging',
    )
    args = argparser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(levelname)s: %(message)s',
    )

    _bump_csv_limit()

    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    silver_df, others_df = load_grouped_data(data_dir)

    report_group('Silver (silver-corpora)', silver_df, top_n=args.top_n)
    report_group('Others (all-corpora-*)', others_df, top_n=args.top_n)

    print_section('Combined overview')
    total = len(silver_df) + len(others_df)
    print(f'  Silver docs:  {len(silver_df):>10,}  ({len(silver_df) / total * 100:.1f}%)')
    print(f'  Other docs:   {len(others_df):>10,}  ({len(others_df) / total * 100:.1f}%)')
    print(f'  Total docs:   {total:>10,}')
    print()


if __name__ == '__main__':
    main()
