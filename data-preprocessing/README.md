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

### `filter_valid_dates.py`
- **Purpose:** Keep only articles with valid dates in `metadata.date`.
- **Use when:** You want to remove undated/invalid-dated articles before chronological splitting.
- **Outputs:** Filtered `*.jsonl.gz` files in output directory.
- **Example:** `python3 filter_valid_dates.py origin-corpora -o origin-corpora-filtred`

### `create_splits.py`
- **Purpose:** Build chronological train/dev/test splits from `*.jsonl.gz`.
- **Use when:** You need either:
  - `global` split (all datasets mixed then split by time), or
  - `per-dataset` split (each dataset split independently by time).
- **Outputs:** `<dataset>.train/dev/test.analysis.jsonl.gz` and `articles_without_dates.txt`.
- **Example:** `python3 create_splits.py origin-corpora-filtred -o chrono-corpora-global --split-type global`

### `create_new_splits.py`
- **Purpose:** Analyze date-based split proportions and visualize original vs newly computed split distributions.
- **Use when:** You want diagnostics/inspection of split behavior, not production split files.
- **Outputs:** `split_distributions.png`, `articles_without_dates.txt`.
- **Example:** `python3 create_new_splits.py origin-corpora-filtred -o split-debug`

### `visualize_splits.py`
- **Purpose:** Plot comparisons between original and generated split distributions.
- **Use when:** You already generated `global_chronological` / `per_dataset_chronological` and want visual QA.
- **Outputs:** `split_distributions_global.png`, `split_distributions_per_dataset.png`.
- **Example:** `python3 visualize_splits.py origin-corpora-filtred -s . -o plots`

### `plot_monthly_stats.py`
- **Purpose:** Plot monthly article counts from `*.jsonl.tsv`.
- **Use when:** You need timeline plots by dataset and combined timeline chart.
- **Outputs:** One `.png` per `.jsonl.tsv` and `combined_datasets.png`.
- **Example:** `python3 plot_monthly_stats.py chrono-corpora-global`

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

### `read_csv_sample.py`
- **Purpose:** Quick diagnostic script printing first 100-row stats from one CSV.
- **Use when:** You need a fast sanity check of lengths/categories/entities in a sample.
- **Notes:** Path is currently hardcoded inside the script.

### `corpora-iptc-patches/translateIds.py`
- **Purpose:** Translate publisher IDs to standardized IDs in TSV using a mapping CSV.
- **Use when:** Patch or metadata files need ID normalization before downstream processing.
- **Example:** `python3 corpora-iptc-patches/translateIds.py -i in.tsv -m map.csv -o out.tsv`

### `origin-corpora/backup/extract_entities.py` (backup/legacy)
- **Purpose:** Extract `article_id -> entities` from `*analysis*.jsonl.gz` and convert `gkbId` to `wdId`.
- **Use when:** You need to regenerate entity TSV and this backup script version is the intended one.
- **Outputs:** TSV with columns `article_id`, `entities`.
- **Example:** `python3 origin-corpora/backup/extract_entities.py origin-corpora -o article_2_entities.tsv`

## Bash Scripts

### `collectData.sh`
- **Purpose:** Pull/compose IPTC-filtered analysis data from source corpora into one working directory.
- **Use when:** Bootstrapping raw corpora for preprocessing.
- **Notes:** Uses many hardcoded paths under `/data/corpora/...`; expects Geneea tools and patch IDs.
- **Usage:** `./collectData.sh <output_directory> <patches_directory>`

### `applyPatches.sh`
- **Purpose:** Apply IPTC patch TSV files to all `*.analysis.jsonl.gz` in current directory.
- **Use when:** You need corrected category metadata before CSV conversion.
- **Notes:** Original files are backed up into `nonpatched/`.

### `origin-corpora/applyPatches.sh`
- **Purpose:** Same patching workflow as top-level `applyPatches.sh`, scoped in `origin-corpora`.
- **Use when:** Applying patches directly in `origin-corpora`.

### `toCsvAll.sh`
- **Purpose:** Convert all `*.analysis.jsonl.gz` in current directory to catlib CSV using `analysisToCatlibCsv.py`.
- **Use when:** Turning analysis files into training CSV format.

### `chrono-corpora-global/toCsvAll.sh`
- **Purpose:** Same conversion script, local to `chrono-corpora-global`.

### `chrono-corpora-per-dataset/toCsvAll.sh`
- **Purpose:** Same conversion script, local to `chrono-corpora-per-dataset`.

### `origin-corpora/toCsvAll.sh`
- **Purpose:** Same conversion script, local to `origin-corpora`.

### `origin-corpora-filtred/toCsvAll.sh`
- **Purpose:** Same conversion script, local to `origin-corpora-filtred`.

### `add_entities_to_all.sh`
- **Purpose:** Batch-run `add_entities_to_csv.py` over multiple corpora directories.
- **Use when:** You want `entities_all-corpora-*.csv` generated everywhere in one command.
- **Notes:** Requires each target directory to contain `entities.tsv`.

### `corpora-iptc-patches/make_patches.sh`
- **Purpose:** End-to-end patch QA pipeline:
  1) collect patch files,
  2) subset affected docs,
  3) apply patches,
  4) compare before/after,
  5) upload diff to Google Sheets.
- **Use when:** Validating and publishing patch impacts.
- **Usage:** `./make_patches.sh <spreadsheet_name> <docs_dir>`

## Typical End-To-End Usage (Most Common)

```bash
# 1) Gather and patch raw data
./collectData.sh origin-corpora corpora-iptc-patches
./applyPatches.sh

# 2) Keep only valid dates
python3 filter_valid_dates.py origin-corpora -o origin-corpora-filtred

# 3) Create chronological splits
python3 create_splits.py origin-corpora-filtred -o chrono-corpora-global --split-type global
python3 create_splits.py origin-corpora-filtred -o chrono-corpora-per-dataset --split-type per-dataset

# 4) Convert analysis files to CSV
cd chrono-corpora-global && ./toCsvAll.sh
cd ../chrono-corpora-per-dataset && ./toCsvAll.sh

# 5) Add entities to merged CSVs (when entities TSV is ready)
python3 ../add_entities_to_csv.py all-corpora-train.csv entities.tsv -o entities_all-corpora-train.csv --id-column id
```

## Notes

- Many scripts rely on internal Geneea tooling/modules and absolute corpora paths.
- Some scripts are utility/diagnostic scripts (not production pipeline steps).
- For table-generation behavior details, see `DESCRIBE_CORPORA_SPEC.md`.



# Training IPTC classifier on all currently available data

## Create and activate a virtual environment

```
virtualenv --download venv
source venv/bin/activate
pip install -r requirements.txt
```

## Gather IPTC annotated data

```
./collectData.sh
```

## Apply patches
This is temporary until the corpora themselves are patched
```
./applyPatches.sh
```

## Convert analyses to CSV for category training:

```
./toCsvAll.sh
```

## Create merged dataset
```
csvstack *.train*csv | python ~/train/util/csvShuf.py -s geneea > all-corpora-train.csv
echo TRAIN
csvcut -c id all-corpora-train.csv | wc -l

csvstack *.dev*csv | python ~/train/util/csvShuf.py -s geneea > all-corpora-dev.csv
echo DEV
csvcut -c id all-corpora-dev.csv | wc -l

csvstack *.test*csv | python ~/train/util/csvShuf.py -s geneea > all-corpora-test.csv
echo TEST
csvcut -c id all-corpora-test.csv | wc -l
```

## Compute category stats
```
python3 ~/train/topic/training/catStats.py -i all-corpora-train.csv > src-corpus-stats-train.csv 2>corpora-global-stats.txt
echo -e "\n\nDEV\n\n" >> corpora-global-stats.txt
python3 ~/train/topic/training/catStats.py -i all-corpora-dev.csv > src-corpus-stats-dev.csv 2>>corpora-global-stats.txt
echo -e "\n\nTEST\n\n" >> corpora-global-stats.txt
python3 ~/train/topic/training/catStats.py -i all-corpora-test.csv > src-corpus-stats-test.csv 2>>corpora-global-stats.txt
```

## Copy to tau.g
```
scp all-corpora-train.csv all-corpora-dev.csv all-corpora-test.csv tau.g:/home/share/corpora/iptc/all-available-2023-09-19/
```
