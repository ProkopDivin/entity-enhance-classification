#!/usr/bin/env python3
"""
Add entities column to CSV file by matching article IDs with entities dictionary.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

# Since this is a CLI script, print is acceptable for output


def load_entities_dict(entities_path: Path) -> dict[str, list]:
    """
    Load entities dictionary from TSV file.

    :param entities_path: Path to TSV file with article_id and entities columns
    :return: Dictionary mapping article_id to entities list
    """
    entities_dict: dict[str, list] = {}

    # Increase CSV field size limit to handle large entity JSON strings
    # Default limit is 131072, increase to 10MB (10485760 bytes)
    csv.field_size_limit(10485760)

    try:
        with open(entities_path, mode="r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            if "article_id" not in reader.fieldnames or "entities" not in reader.fieldnames:
                print(
                    f"Error: TSV file must have 'article_id' and 'entities' columns. Found: {reader.fieldnames}",
                    file=sys.stderr,
                )
                sys.exit(1)

            for row_num, row in enumerate(reader, start=2):  # Start at 2 because header is row 1
                article_id = row.get("article_id", "").strip()
                entities_json = row.get("entities", "").strip()

                if not article_id:
                    continue

                try:
                    entities = json.loads(entities_json) if entities_json else []
                    entities_dict[article_id] = entities
                except json.JSONDecodeError as e:
                    print(
                        f"Warning: Invalid JSON in entities column for article_id '{article_id}' "
                        f"at line {row_num}: {e}",
                        file=sys.stderr,
                    )
                    entities_dict[article_id] = []

        print(f"Loaded {len(entities_dict)} article entities from {entities_path}")
    except OSError as e:
        print(f"Error reading {entities_path}: {e}", file=sys.stderr)
        sys.exit(1)

    return entities_dict


def add_entities_to_csv(
    input_csv_path: Path, output_csv_path: Path, entities_dict: dict[str, list], id_column: str
) -> None:
    """
    Add entities column to CSV file.

    :param input_csv_path: Path to input CSV file
    :param output_csv_path: Path to output CSV file
    :param entities_dict: Dictionary mapping article_id to entities list
    :param id_column: Name of the column containing article IDs
    """
    try:
        with open(input_csv_path, mode="r", encoding="utf-8", newline="") as f_in:
            reader = csv.DictReader(f_in)
            fieldnames = list(reader.fieldnames)

            if id_column not in fieldnames:
                print(f"Error: Column '{id_column}' not found in CSV. Available columns: {fieldnames}", file=sys.stderr)
                sys.exit(1)

            # Add entities column if it doesn't exist
            if "entities" not in fieldnames:
                fieldnames.append("entities")
            else:
                print("Warning: 'entities' column already exists. It will be overwritten.", file=sys.stderr)

            rows_processed = 0
            rows_with_entities = 0
            rows_without_entities = 0

            with open(output_csv_path, mode="w", encoding="utf-8", newline="") as f_out:
                writer = csv.DictWriter(f_out, fieldnames=fieldnames)
                writer.writeheader()

                for row in reader:
                    rows_processed += 1
                    article_id = row.get(id_column, "").strip()

                    if article_id in entities_dict:
                        entities = entities_dict[article_id]
                        rows_with_entities += 1
                    else:
                        entities = []
                        rows_without_entities += 1

                    # Serialize entities as JSON string
                    row["entities"] = json.dumps(entities, ensure_ascii=False)
                    writer.writerow(row)

        print(f"Processed {rows_processed} rows")
        print(f"  - {rows_with_entities} rows with matching entities")
        print(f"  - {rows_without_entities} rows without matching entities")
        print(f"Output written to: {output_csv_path}")
    except OSError as e:
        print(f"Error processing CSV file: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Main entry point for the script."""
    argparser = argparse.ArgumentParser(
        description=(
            "Add entities column to CSV file by matching article IDs " "with entities from a TSV dictionary file."
        )
    )
    argparser.add_argument("csv_file", help="Input CSV file path")
    argparser.add_argument("entities_file", help="Path to TSV file with article_id and entities columns")
    argparser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: overwrite input file)",
    )
    argparser.add_argument(
        "--id-column",
        type=str,
        default="id",
        help="Name of the column containing article IDs (default: 'id')",
    )
    args = argparser.parse_args()

    csv_path = Path(args.csv_file)
    entities_path = Path(args.entities_file)

    if not csv_path.is_file():
        print(f"Error: {csv_path} is not a file", file=sys.stderr)
        sys.exit(1)

    if not entities_path.is_file():
        print(f"Error: {entities_path} is not a file", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = csv_path

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading entities from {entities_path}...")
    entities_dict = load_entities_dict(entities_path=entities_path)

    print(f"Adding entities to {csv_path}...")
    add_entities_to_csv(
        input_csv_path=csv_path,
        output_csv_path=output_path,
        entities_dict=entities_dict,
        id_column=args.id_column,
    )

    print("Done!")


if __name__ == "__main__":
    main()
