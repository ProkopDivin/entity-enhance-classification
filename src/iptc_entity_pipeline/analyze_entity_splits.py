'''
Analyze GKB-linked entities in train and test corpus splits.
'''

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from geneea.core import logutil

from iptc_entity_pipeline.data_loading import (
    DocWithEntities,
    LinkedEntity,
    attach_entities,
    load_and_normalize,
    load_wdid_map,
)

LOG = logutil.getLogger(__package__, __file__)

DEFAULT_WDID_MAPPING_TSV = Path(
    '/home/prokop/Git/entity-enhance-classification/data/gold-chrono-per-dataset/wdId_mapping.tsv'
)
REMOVED_CAT_IDS: tuple[str, ...] = ('20000419',)


@dataclass(frozen=True)
class InputPaths:
    '''Filesystem paths for one train/test entity corpus analysis run.'''

    label: str
    train_csv: Path
    test_csv: Path
    wdid_mapping_tsv: Path


@dataclass(frozen=True)
class EntitySplitStats:
    '''Summary statistics for GKB-linked entities across train/test splits.'''

    article_count_train: int
    article_count_test: int
    unique_entities_train: int
    unique_entities_test: int
    total_entities_train: int
    total_entities_test: int
    avg_occurrence_combined: float
    test_only_unique_count: int
    test_only_avg_occurrence: float
    test_in_train_unique_count: int
    test_in_train_avg_occurrence: float
    avg_test_in_train_entities_per_article: float
    avg_gkb_entities_per_article_train: float
    avg_gkb_entities_per_article_test: float


def has_raw_gkb_id(*, entity: LinkedEntity) -> bool:
    '''
    Return whether the entity record has an explicit GKB identifier.

    :param entity: Linked entity attached to a corpus document.
    :return: ``True`` when ``gkbId`` / ``gkb_id`` is present in the raw payload.
    '''
    raw = entity.raw_entity
    if not raw:
        return False
    gkb_raw = raw.get('gkbId') or raw.get('gkb_id')
    return isinstance(gkb_raw, str) and bool(gkb_raw.strip())


def gkb_ids_in_doc(*, doc: DocWithEntities) -> list[str]:
    '''
    Extract GKB IDs from one document, skipping entities without GKB ID.

    :param doc: Corpus document with attached entities.
    :return: GKB IDs for entities that have an explicit GKB identifier.
    '''
    return [entity.gkb_id for entity in doc.entities if has_raw_gkb_id(entity=entity)]


def count_gkb_occurrences(*, corpus: Iterable[DocWithEntities]) -> Counter[str]:
    '''
    Count GKB entity occurrences across all articles in one corpus.

    :param corpus: Iterable of documents with attached entities.
    :return: Mapping ``gkb_id -> occurrence_count``.
    '''
    counts: Counter[str] = Counter()
    for doc in corpus:
        counts.update(gkb_ids_in_doc(doc=doc))
    return counts


def mean_occurrence(*, gkb_ids: Sequence[str], occurrence_counts: Mapping[str, int]) -> float:
    '''
    Compute mean occurrence for a non-empty sequence of GKB IDs.

    :param gkb_ids: GKB IDs whose global occurrence counts should be averaged.
    :param occurrence_counts: Global occurrence counts from train and test combined.
    :return: Mean occurrence count.
    '''
    if not gkb_ids:
        return 0.0
    total = sum(occurrence_counts.get(gkb_id, 0) for gkb_id in gkb_ids)
    return total / len(gkb_ids)


def compute_entity_split_stats(
    *,
    train_corpus: Iterable[DocWithEntities],
    test_corpus: Iterable[DocWithEntities],
) -> EntitySplitStats:
    '''
    Compute entity statistics for train and test corpora.

    :param train_corpus: Training split documents with attached entities.
    :param test_corpus: Test split documents with attached entities.
    :return: Aggregated entity split statistics.
    '''
    train_docs = list(train_corpus)
    test_docs = list(test_corpus)

    train_counts = count_gkb_occurrences(corpus=train_docs)
    test_counts = count_gkb_occurrences(corpus=test_docs)
    combined_counts = train_counts + test_counts

    train_gkb_ids = set(train_counts)
    test_gkb_ids = set(test_counts)
    test_only_gkb_ids = sorted(test_gkb_ids - train_gkb_ids)
    test_in_train_gkb_ids = sorted(test_gkb_ids & train_gkb_ids)

    train_entities_per_article = [len(gkb_ids_in_doc(doc=doc)) for doc in train_docs]
    test_entities_per_article = [len(gkb_ids_in_doc(doc=doc)) for doc in test_docs]
    test_in_train_per_article = [
        sum(1 for gkb_id in gkb_ids_in_doc(doc=doc) if gkb_id in train_gkb_ids)
        for doc in test_docs
    ]

    unique_combined = len(combined_counts)
    avg_occurrence_combined = (
        sum(combined_counts.values()) / unique_combined if unique_combined else 0.0
    )

    return EntitySplitStats(
        article_count_train=len(train_docs),
        article_count_test=len(test_docs),
        unique_entities_train=len(train_gkb_ids),
        unique_entities_test=len(test_gkb_ids),
        total_entities_train=sum(train_counts.values()),
        total_entities_test=sum(test_counts.values()),
        avg_occurrence_combined=avg_occurrence_combined,
        test_only_unique_count=len(test_only_gkb_ids),
        test_only_avg_occurrence=mean_occurrence(
            gkb_ids=test_only_gkb_ids,
            occurrence_counts=combined_counts,
        ),
        test_in_train_unique_count=len(test_in_train_gkb_ids),
        test_in_train_avg_occurrence=mean_occurrence(
            gkb_ids=test_in_train_gkb_ids,
            occurrence_counts=combined_counts,
        ),
        avg_test_in_train_entities_per_article=(
            sum(test_in_train_per_article) / len(test_docs) if test_docs else 0.0
        ),
        avg_gkb_entities_per_article_train=(
            sum(train_entities_per_article) / len(train_docs) if train_docs else 0.0
        ),
        avg_gkb_entities_per_article_test=(
            sum(test_entities_per_article) / len(test_docs) if test_docs else 0.0
        ),
    )


def load_corpora_with_entities(
    *,
    paths: InputPaths,
) -> tuple[list[DocWithEntities], list[DocWithEntities]]:
    '''
    Load train/test corpora and attach entities using pipeline helpers.

    :param paths: Train/test CSV and wdId mapping paths.
    :return: Tuple ``(train_docs, test_docs)`` with entities attached.
    '''
    LOG.info(
        f'Loading dataset={paths.label}, train_csv={paths.train_csv}, test_csv={paths.test_csv}'
    )
    corpora = load_and_normalize(
        train_csv=str(paths.train_csv),
        test_csv=str(paths.test_csv),
        removed_cat_ids=REMOVED_CAT_IDS,
    )
    wdid_mapping = load_wdid_map(wdid_mapping_tsv=str(paths.wdid_mapping_tsv))
    attach_entities(
        corpus=corpora.train,
        csv_path=str(paths.train_csv),
        wdid_mapping=wdid_mapping,
        min_relevance=0.0,
    )
    attach_entities(
        corpus=corpora.test,
        csv_path=str(paths.test_csv),
        wdid_mapping=wdid_mapping,
        min_relevance=0.0,
    )
    return list(corpora.train), list(corpora.test)


def log_entity_split_stats(*, dataset_name: str, stats: EntitySplitStats) -> None:
    '''
    Log computed entity split statistics.

    :param dataset_name: Human-readable dataset label.
    :param stats: Aggregated statistics to print.
    :return: None
    '''
    LOG.info(f'=== Entity split stats: {dataset_name} ===')
    LOG.info(f'Articles: train={stats.article_count_train}, test={stats.article_count_test}')
    LOG.info(f'Unique GKB entities: train={stats.unique_entities_train}, test={stats.unique_entities_test}')
    LOG.info(
        f'Total GKB entity occurrences (with repetition): '
        f'train={stats.total_entities_train}, test={stats.total_entities_test}'
    )
    LOG.info(f'Average GKB entity occurrence (train+test combined): {stats.avg_occurrence_combined:.4f}')
    LOG.info(
        f'Test-only unique GKB entities: count={stats.test_only_unique_count}, '
        f'avg_occurrence={stats.test_only_avg_occurrence:.4f}'
    )
    LOG.info(
        f'Test GKB entities also in train: unique_count={stats.test_in_train_unique_count}, '
        f'avg_occurrence={stats.test_in_train_avg_occurrence:.4f}'
    )
    LOG.info(
        f'Average test entities also in train per article: '
        f'{stats.avg_test_in_train_entities_per_article:.4f}'
    )
    LOG.info(
        f'Average GKB entities per article: '
        f'train={stats.avg_gkb_entities_per_article_train:.4f}, '
        f'test={stats.avg_gkb_entities_per_article_test:.4f}'
    )


def default_label(*, train_csv: Path) -> str:
    '''
    Derive a short dataset label from the train CSV parent directory.

    :param train_csv: Path to the training CSV file.
    :return: Parent directory name, or the train CSV stem when unavailable.
    '''
    parent_name = train_csv.parent.name
    return parent_name if parent_name and parent_name != '.' else train_csv.stem


def build_input_paths(
    *,
    train_csv: Path,
    test_csv: Path,
    wdid_mapping_tsv: Path,
    label: str | None,
) -> InputPaths:
    '''
    Build validated input path config for one analysis run.

    :param train_csv: Path to training entities CSV.
    :param test_csv: Path to test entities CSV.
    :param wdid_mapping_tsv: Path to GKB-to-Wikidata mapping TSV.
    :param label: Optional dataset label for logging.
    :return: Validated input paths.
    '''
    for path in (train_csv, test_csv, wdid_mapping_tsv):
        if not path.is_file():
            raise FileNotFoundError(f'Input file not found: {path}')
    return InputPaths(
        label=label or default_label(train_csv=train_csv),
        train_csv=train_csv,
        test_csv=test_csv,
        wdid_mapping_tsv=wdid_mapping_tsv,
    )


def main() -> None:
    '''Run entity split analysis from CLI.'''
    argparser = argparse.ArgumentParser(description='Analyze GKB entities in train/test splits.')
    argparser.add_argument(
        '--train-csv',
        required=True,
        type=Path,
        help='Path to training entities CSV.',
    )
    argparser.add_argument(
        '--test-csv',
        required=True,
        type=Path,
        help='Path to test entities CSV.',
    )
    argparser.add_argument(
        '--wdid-mapping-tsv',
        type=Path,
        default=DEFAULT_WDID_MAPPING_TSV,
        help=f'Path to wdId mapping TSV (default: {DEFAULT_WDID_MAPPING_TSV}).',
    )
    argparser.add_argument(
        '--label',
        help='Optional dataset label used in log output (default: train CSV parent directory name).',
    )
    logutil.addLogArguments(argparser)
    args = argparser.parse_args()
    logutil.configureFromArgs(args)

    try:
        paths = build_input_paths(
            train_csv=args.train_csv,
            test_csv=args.test_csv,
            wdid_mapping_tsv=args.wdid_mapping_tsv,
            label=args.label,
        )
        train_docs, test_docs = load_corpora_with_entities(paths=paths)
        stats = compute_entity_split_stats(train_corpus=train_docs, test_corpus=test_docs)
        log_entity_split_stats(dataset_name=paths.label, stats=stats)
    except Exception:
        LOG.exception('Entity split analysis failed')
        raise


if __name__ == '__main__':
    main()
