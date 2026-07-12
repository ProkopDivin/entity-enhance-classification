#!/usr/bin/env python3
"""Analyze near-duplicate articles between train and test CSV splits."""

import argparse
import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from utils.csv_io import ensure_large_csv_fields

TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


@dataclass(frozen=True)
class ArticleStub:
    """Lightweight article representation used for duplicate checks."""

    article_id: str
    token_counts: Counter[str]
    token_total: int


def tokenize(*, text: str) -> list[str]:
    """Extract word-like tokens from text."""
    return TOKEN_PATTERN.findall(text.lower())


def article_stub_from_row(
    *,
    row: dict[str, str],
    id_col: str,
    title_col: str,
    lead_col: str,
    text_col: str,
) -> ArticleStub | None:
    """Convert one CSV row to article stub."""
    article_id = (row.get(id_col) or "").strip()
    if not article_id:
        return None

    title = row.get(title_col) or ""
    lead = row.get(lead_col) or ""
    text = row.get(text_col) or ""
    tokens = tokenize(text=f"{title} {lead} {text}")
    token_counts = Counter(tokens)
    return ArticleStub(
        article_id=article_id,
        token_counts=token_counts,
        token_total=sum(token_counts.values()),
    )


def read_article_stubs(
    *,
    csv_path: Path,
    id_col: str,
    title_col: str,
    lead_col: str,
    text_col: str,
) -> list[ArticleStub]:
    """Load all rows into stubs."""
    stubs: list[ArticleStub] = []
    with csv_path.open("r", encoding="utf-8", newline="") as in_file:
        reader = csv.DictReader(in_file)
        for row in reader:
            stub = article_stub_from_row(
                row=row,
                id_col=id_col,
                title_col=title_col,
                lead_col=lead_col,
                text_col=text_col,
            )
            if stub is not None:
                stubs.append(stub)
    return stubs


def overlap_percent(*, left: ArticleStub, right: ArticleStub) -> float:
    """Compute token-overlap percentage for two articles."""
    common = sum((left.token_counts & right.token_counts).values())
    denominator = max(left.token_total, right.token_total)
    if denominator == 0:
        return 100.0
    return 100.0 * common / denominator


def ensure_csv_limit() -> None:
    """Increase CSV parser field-size limit for large text/entity fields."""
    ensure_large_csv_fields(preferred_limit=10**9)


def write_pairs_csv(*, output_path: Path, rows: list[dict[str, str]]) -> None:
    """Write duplicate pairs to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as out_file:
        fieldnames = [
            "test_id",
            "train_id",
            "word_overlap_percent",
        ]
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_train_groups_csv(*, output_path: Path, groups: list[list[str]]) -> None:
    """Write train duplicate groups to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=["group_size", "train_ids"])
        writer.writeheader()
        for ids in sorted(groups, key=lambda items: (-len(items), items[0])):
            writer.writerow({"group_size": len(ids), "train_ids": "|".join(ids)})


def write_report(
    *,
    output_path: Path,
    train_count: int,
    test_count: int,
    confirmed_count: int,
    threshold: float,
    confirmed_matches: dict[str, list[str]],
    train_groups: list[list[str]],
) -> None:
    """Write concise markdown summary report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    confirmed_pct = 100.0 * confirmed_count / test_count if test_count else 0.0

    lines: list[str] = []
    lines.append("# Train/Test Duplicate Analysis")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- `word_overlap_threshold`: {threshold:.1f}%")
    lines.append("")
    lines.append("## Counts")
    lines.append(f"- Train articles: {train_count}")
    lines.append(f"- Test articles: {test_count}")
    lines.append(
        f"- Test articles with confirmed overlap >= threshold: {confirmed_count} ({confirmed_pct:.2f}%)"
    )
    lines.append(f"- Train duplicate groups (size > 1): {len(train_groups)}")
    lines.append("")
    lines.append("## Confirmed test -> train duplicate IDs")
    if confirmed_matches:
        for test_id in sorted(confirmed_matches):
            train_ids = ", ".join(sorted(confirmed_matches[test_id]))
            lines.append(f"- `{test_id}` -> {train_ids}")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Train duplicate groups")
    if train_groups:
        for ids in sorted(train_groups, key=lambda items: (-len(items), items[0])):
            lines.append(f"- ({len(ids)}) {', '.join(ids)}")
    else:
        lines.append("- None")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze near-duplicate articles with word-overlap threshold only."
    )
    parser.add_argument("--train-csv", required=True, help="Path to train CSV file.")
    parser.add_argument("--test-csv", required=True, help="Path to test CSV file.")
    parser.add_argument(
        "--output-report",
        default="duplicate_report.md",
        help="Output markdown report path.",
    )
    parser.add_argument(
        "--output-pairs-csv",
        default="duplicate_pairs.csv",
        help="Output CSV with confirmed duplicate test/train ID pairs.",
    )
    parser.add_argument(
        "--output-train-groups-csv",
        default="train_duplicate_groups.csv",
        help="Output CSV with duplicate train ID groups.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=90.0,
        help="Minimum token-overlap percentage for confirmed duplicate pair.",
    )
    parser.add_argument("--id-col", default="id", help="CSV column with article ID.")
    parser.add_argument("--title-col", default="title", help="CSV column with title.")
    parser.add_argument("--lead-col", default="lead", help="CSV column with lead.")
    parser.add_argument("--text-col", default="text", help="CSV column with body text.")
    return parser.parse_args()


class UnionFind:
    """Union-find structure for duplicate-group connected components."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, idx: int) -> int:
        """Find canonical root index."""
        while self.parent[idx] != idx:
            self.parent[idx] = self.parent[self.parent[idx]]
            idx = self.parent[idx]
        return idx

    def union(self, left: int, right: int) -> None:
        """Union two sets."""
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            self.parent[root_left] = root_right
            return
        if self.rank[root_left] > self.rank[root_right]:
            self.parent[root_right] = root_left
            return
        self.parent[root_right] = root_left
        self.rank[root_left] += 1


def build_train_duplicate_groups(*, train_stubs: list[ArticleStub], threshold: float) -> list[list[str]]:
    """Build train duplicate groups using pairwise overlap and connected components."""
    uf = UnionFind(size=len(train_stubs))
    for idx, left in enumerate(train_stubs):
        for jdx in range(idx + 1, len(train_stubs)):
            right = train_stubs[jdx]
            similarity = overlap_percent(left=left, right=right)
            if similarity >= threshold:
                uf.union(idx, jdx)

    groups_by_root: dict[int, list[str]] = defaultdict(list)
    for idx, stub in enumerate(train_stubs):
        root = uf.find(idx)
        groups_by_root[root].append(stub.article_id)

    return [sorted(ids) for ids in groups_by_root.values() if len(ids) > 1]


def main() -> None:
    """Run duplicate analysis."""
    args = parse_args()
    ensure_csv_limit()

    train_csv = Path(args.train_csv)
    test_csv = Path(args.test_csv)
    report_path = Path(args.output_report)
    pairs_csv_path = Path(args.output_pairs_csv)
    train_groups_csv_path = Path(args.output_train_groups_csv)

    test_stubs = read_article_stubs(
        csv_path=test_csv,
        id_col=args.id_col,
        title_col=args.title_col,
        lead_col=args.lead_col,
        text_col=args.text_col,
    )
    test_count = len(test_stubs)
    train_stubs = read_article_stubs(
        csv_path=train_csv,
        id_col=args.id_col,
        title_col=args.title_col,
        lead_col=args.lead_col,
        text_col=args.text_col,
    )
    train_count = len(train_stubs)

    confirmed_matches: dict[str, list[str]] = defaultdict(list)
    pair_rows: list[dict[str, str]] = []
    for test_stub in test_stubs:
        for train_stub in train_stubs:
            similarity = overlap_percent(left=test_stub, right=train_stub)
            if similarity < args.threshold:
                continue
            confirmed_matches[test_stub.article_id].append(train_stub.article_id)
            pair_rows.append(
                {
                    "test_id": test_stub.article_id,
                    "train_id": train_stub.article_id,
                    "word_overlap_percent": f"{similarity:.2f}",
                }
            )

    confirmed_count = len(confirmed_matches)
    train_groups = build_train_duplicate_groups(train_stubs=train_stubs, threshold=args.threshold)

    write_pairs_csv(output_path=pairs_csv_path, rows=pair_rows)
    write_train_groups_csv(output_path=train_groups_csv_path, groups=train_groups)
    write_report(
        output_path=report_path,
        train_count=train_count,
        test_count=test_count,
        confirmed_count=confirmed_count,
        threshold=args.threshold,
        confirmed_matches=confirmed_matches,
        train_groups=train_groups,
    )


if __name__ == "__main__":
    main()
