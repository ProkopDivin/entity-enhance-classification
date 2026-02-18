#!/usr/bin/env python3
"""
Extract article IDs and entities from all analysis.jsonl.gz files in input directory
and save them to a TSV file.
"""
import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

# Since this is a CLI script, print is acceptable for output


def extract_entities_from_files(input_dir: Path) -> dict[str, list]:
    """
    Extract article IDs and entities from all analysis.jsonl.gz files.

    :param input_dir: Directory containing .analysis.jsonl.gz files
    :return: Dictionary mapping article_id to entities list
    """
    entities_dict: dict[str, list] = {}

    # Find all files with 'analysis' in the name and ending with .jsonl.gz
    files = sorted(input_dir.glob("*analysis*.jsonl.gz"))

    if not files:
        print(f"Warning: No files matching *analysis*.jsonl.gz found in {input_dir}", file=sys.stderr)
        return entities_dict

    print(f"Found {len(files)} analysis files to process...")

    for filepath in files:
        print(f"Processing {filepath.name}...")
        try:
            with gzip.open(filepath, mode="rt", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        article = json.loads(line)
                        article_id = article.get("id")
                        entities = article.get("entities", [])

                        if article_id:
                            if article_id in entities_dict:
                                print(
                                    f"Warning: Duplicate article ID '{article_id}' found in {filepath.name} (line {line_num})",
                                    file=sys.stderr,
                                )
                            entities_dict[article_id] = entities
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON in {filepath.name} at line {line_num}: {e}", file=sys.stderr)
                        continue
        except OSError as e:
            print(f"Error reading {filepath}: {e}", file=sys.stderr)

    return entities_dict


def write_entities_tsv(entities_dict: dict[str, list], output_path: Path) -> None:
    """
    Write entities dictionary to TSV file.

    :param entities_dict: Dictionary mapping article_id to entities list
    :param output_path: Path where to save the TSV file
    """
    try:
        with open(output_path, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            # Write header
            writer.writerow(["article_id", "entities"])

            # Write data
            for article_id, entities in sorted(entities_dict.items()):
                # Serialize entities as JSON string
                entities_json = json.dumps(entities, ensure_ascii=False)
                writer.writerow([article_id, entities_json])

        print(f"Wrote {len(entities_dict)} articles to: {output_path}")
    except OSError as e:
        print(f"Error writing {output_path}: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Main entry point for the script."""
    argparser = argparse.ArgumentParser(
        description=(
            "Extract article IDs and entities from all analysis.jsonl.gz files "
            "in input directory and save them to a TSV file."
        )
    )
    argparser.add_argument(
        "input",
        nargs="?",
        default=".",
        help=("Input directory containing .analysis.jsonl.gz files " "(default: current directory)"),
    )
    argparser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output TSV file path (default: entities.tsv in input directory)",
    )
    args = argparser.parse_args()

    input_dir = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_dir / "entities.tsv"

    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Extracting entities from {input_dir}...")
    entities_dict = extract_entities_from_files(input_dir=input_dir)

    if not entities_dict:
        print("No entities found. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(entities_dict)} unique articles with entities")
    write_entities_tsv(entities_dict=entities_dict, output_path=output_path)

    print("Done!")


if __name__ == "__main__":
    main()
