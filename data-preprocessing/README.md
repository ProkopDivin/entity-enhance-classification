# Data Preprocessing Scripts Reference

This README explains all Python and Bash scripts in `data-preprocessing`, what they do, and when to use them.

## How To Get Help For Any Script

- Python scripts:
  - `python3 <script>.py --help`
- Bash scripts:
  - `bash <script>.sh` (most scripts print usage when args are missing)

## Python Scripts

### `analyze_jsonl_stats.py`
- **Purpose:** Analyze `*.jsonl.gz` files and compute date coverage statistics from `metadata.date`.
- **Use when:** You need per-dataset date completeness before splitting/filtering.
- **Outputs:** `stats.md` and one `.tsv` date list per input file.
- **Example:** `python3 analyze_jsonl_stats.py origin-corpora`

### `create_splits.py`
- **Purpose:** Build chronological train/test splits from `*.jsonl.gz`.
- **Use when:** You need either:
  - `global` split (all datasets mixed then split by time), or
  - `per-dataset` split (each dataset split independently by time).
- **Outputs:** `<dataset>.train/test.analysis.jsonl.gz` and `articles_without_dates.txt`.
- **Example:** `python3 create_splits.py origin-corpora-filtred -o chrono-corpora-global --split global`

### `visualize_splits.py`
- **Purpose:** Plot comparisons between original and generated split distributions.
- **Use when:** You already generated `global_chronological` / `per_dataset_chronological` and want visual QA.
- **Outputs:** `split_distributions_global.png`, `split_distributions_per_dataset.png`.
- **Example:** `python3 visualize_splits.py origin-corpora-filtred -s . -o plots`

### `analyze_splits.py`
- **Purpose:** Compare two CSV split directories (dataset mix, categories, entities, article length).
- **Use when:** You want a full side-by-side statistical comparison between two prepared corpora variants.
- **Outputs:** CSV tables and many `.png` charts in output folder.
- **Example:** `python3 analyze_splits.py origin-corpora chrono-corpora-global -o analysis_results`

### `add_entities_to_csv.py`
- **Purpose:** Add/overwrite `entities` column in a CSV by joining article IDs with a TSV mapping.
- **Use when:** You have `article_id -> entities` mapping and need entities appended to model CSV files.
- **Inputs:** CSV + TSV containing `article_id` and `entities`.
- **Example:** `python3 add_entities_to_csv.py all-corpora-train.csv entities.tsv -o entities_all-corpora-train.csv --id-column id`

### `extract_wikidata_mapping.py`
- **Purpose:** Fetch GKB items and extract Wikidata IDs from `source` fields.
- **Use when:** You need `gkb_id -> wikidata_id` mapping for entity normalization.
- **Outputs:** `<prefix>_ids.txt`, `<prefix>_mapping.tsv`.
- **Example:** `python3 extract_wikidata_mapping.py gkb_ids -o wikidata`

### `describe_corpora.py`
- **Purpose:** Generate thesis LaTeX tables with corpus/category/entity statistics.
- **Use when:** You need publication-ready macro tables from prepared corpora.
- **Outputs:** Per-corpus `*_tables.tex`, `all_corpora_summary_basic.tex`, `all_corpora_aggregated.tex`, plus optional `*_missing_entities.txt`.
- **Example:** `python3 describe_corpora.py origin-downsampled-nl -o origin-downsampled-nl/macro-tables`

### `extract_entities_date.py`
- **Purpose:** Extract entities with date information from corpus files.
- **Use when:** You need entity-date associations from the source corpora.
- **Example:** `python3 extract_entities_date.py --help`

### `analyze_article_duplicates.py`
- **Purpose:** Detect and report duplicate articles across corpus files.
- **Use when:** You need to check for data duplication before training.
- **Example:** `python3 analyze_article_duplicates.py --help`

### `analyze_silver_distributions.py`
- **Purpose:** Analyze label distributions in silver-standard (auto-annotated) corpora.
- **Use when:** You need distribution diagnostics on silver data.
- **Example:** `python3 analyze_silver_distributions.py --help`

### `entity_wordclouds.py`
- **Purpose:** Generate word clouds from entity mentions in corpus files.
- **Use when:** You need visual summaries of entity distributions.
- **Example:** `python3 entity_wordclouds.py --help`

### `category_split_analysis.py`
- **Purpose:** Analyze category distributions across train/test splits.
- **Use when:** You need to verify category balance after splitting.
- **Example:** `python3 category_split_analysis.py --help`
