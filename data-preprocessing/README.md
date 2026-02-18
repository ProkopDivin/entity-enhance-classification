# Data Preparation Pipeline

This guide documents the complete workflow for preparing IPTC classification training data with entity information.

## Overview

The data preparation pipeline consists of the following steps:

1. **Set up environment and gather data**: Prepare the environment and collect IPTC annotated data
2. **Filter valid dates**: Remove articles with invalid or missing dates from the original corpora
3. **Create chronological splits**: Generate train/dev/test splits based on chronological ordering
4. **Convert to CSV**: Transform analysis files to CSV format for training
5. **Merge datasets**: Combine all corpora splits into unified train/dev/test files
6. **Extract entities**: Extract entity information from analysis files
7. **Add entities to CSV**: Merge entity data into the training CSV files

## Prerequisites

- Python 3.9+
- Virtual environment (recommended)
- Required Python packages (see `requirements.txt`)
- Access to original corpora data

## Step-by-Step Guide

### Step 1: Set Up Environment and Gather Data

Follow the standard IPTC data preparation workflow to set up the environment and collect the original data:

Follow the standard IPTC data preparation workflow:

#### Create and activate virtual environment

```bash
virtualenv --download venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Gather IPTC annotated data

```bash
./collectData.sh
```

#### Apply patches (temporary until corpora are patched)

```bash
./applyPatches.sh
```

### Step 4: Convert Analyses to CSV

Convert the analysis files to CSV format for category training. This processes all `.analysis.jsonl.gz` files in the specified directory.

```bash
./toCsvAll.sh <directory>
```

**What it does:**
- Processes all `*.analysis.jsonl.gz` files in the specified directory
- Converts each file to CSV format using `analysisToCatlibCsv.py`
- Creates CSV files ready for training

**Usage:**
```bash
./toCsvAll.sh <directory>
```

**Example:**
```bash
# Process global chronological splits
./toCsvAll.sh chrono-corpora-global

# Process per-dataset chronological splits
./toCsvAll.sh chrono-corpora-per-dataset
```

### Step 5: Create Merged Datasets

Combine all corpora splits into unified train/dev/test files:

```bash
# Merge train files
csvstack *.train*.csv | python ~/train/util/csvShuf.py -s geneea > all-corpora-train.csv
echo TRAIN
csvcut -c id all-corpora-train.csv | wc -l

# Merge dev files
csvstack *.dev*.csv | python ~/train/util/csvShuf.py -s geneea > all-corpora-dev.csv
echo DEV
csvcut -c id all-corpora-dev.csv | wc -l

# Merge test files
csvstack *.test*.csv | python ~/train/util/csvShuf.py -s geneea > all-corpora-test.csv
echo TEST
csvcut -c id all-corpora-test.csv | wc -l
```

**What it does:**
- Combines all `*.train*.csv` files into `all-corpora-train.csv`
- Combines all `*.dev*.csv` files into `all-corpora-dev.csv`
- Combines all `*.test*.csv` files into `all-corpora-test.csv`
- Shuffles the data using a fixed seed for reproducibility
- Prints the number of articles in each split

### Step 6: Compute Category Statistics

Generate statistics about category distribution in the datasets:

```bash
python3 ~/train/topic/training/catStats.py -i all-corpora-train.csv > src-corpus-stats-train.csv 2>corpora-global-stats.txt
echo -e "\n\nDEV\n\n" >> corpora-global-stats.txt
python3 ~/train/topic/training/catStats.py -i all-corpora-dev.csv > src-corpus-stats-dev.csv 2>>corpora-global-stats.txt
echo -e "\n\nTEST\n\n" >> corpora-global-stats.txt
python3 ~/train/topic/training/catStats.py -i all-corpora-test.csv > src-corpus-stats-test.csv 2>>corpora-global-stats.txt
```

**What it does:**
- Analyzes category distribution in each split
- Generates statistics files for train, dev, and test sets
- Creates a combined statistics report in `corpora-global-stats.txt`

### Step 7: Extract Entities

Extract entity information from all analysis files and save to a TSV file:

```bash
python3 extract_entities.py <input_dir> -o entities.tsv
```

**What it does:**
- Reads all `*analysis*.jsonl.gz` files from the input directory
- Extracts article IDs and their associated entities
- Saves the mapping to a TSV file with columns: `article_id` and `entities` (JSON format)

**Usage:**
```bash
python3 extract_entities.py <input_dir> [-o <output_file>]
```

**Options:**
- `input_dir`: Directory containing `.analysis.jsonl.gz` files (default: current directory)
- `-o, --output`: Output TSV file path (default: `entities.tsv` in input directory)

**Example:**
```bash
# Extract entities from global chronological splits
python3 extract_entities.py chrono-corpora-global -o entities-global.tsv

# Extract entities from per-dataset chronological splits
python3 extract_entities.py chrono-corpora-per-dataset -o entities-per-dataset.tsv
```

### Step 8: Add Entities to CSV Files

Add entity information to the merged CSV files:

```bash
python3 add_entities_to_csv.py all-corpora-train.csv entities.tsv -o all-corpora-train-with-entities.csv --id-column id
python3 add_entities_to_csv.py all-corpora-dev.csv entities.tsv -o all-corpora-dev-with-entities.csv --id-column id
python3 add_entities_to_csv.py all-corpora-test.csv entities.tsv -o all-corpora-test-with-entities.csv --id-column id
```

**What it does:**
- Loads entity mappings from the TSV file
- Matches article IDs in the CSV with entities from the TSV
- Adds an `entities` column to the CSV with JSON-formatted entity data
- Articles without matching entities get an empty array `[]`

**Usage:**
```bash
python3 add_entities_to_csv.py <csv_file> <entities_file> [-o <output_file>] [--id-column <column_name>]
```

**Options:**
- `csv_file`: Input CSV file path
- `entities_file`: Path to TSV file with article_id and entities columns
- `-o, --output`: Output CSV file path (default: overwrites input file)
- `--id-column`: Name of the column containing article IDs (default: `id`)

**Example:**
```bash
# Add entities to train set
python3 add_entities_to_csv.py all-corpora-train.csv entities-global.tsv \
    -o all-corpora-train-with-entities.csv --id-column id
```

## Complete Workflow Example

Here's a complete example workflow for preparing data with global chronological splits:

```bash
# Step 1: Set up environment and gather data
virtualenv --download venv
source venv/bin/activate
pip install -r requirements.txt
./collectData.sh
./applyPatches.sh

# Step 2: Filter valid dates
python3 filter_valid_dates.py origin_corpora origin-corpora-filtred

# Step 3: Create global chronological splits
python3 create_splits.py origin-corpora-filtred -o chrono-corpora-global --split-type global

# Step 4: Convert to CSV
./toCsvAll.sh chrono-corpora-global

# Step 5: Merge datasets
csvstack *.train*.csv | python ~/train/util/csvShuf.py -s geneea > all-corpora-train.csv
csvstack *.dev*.csv | python ~/train/util/csvShuf.py -s geneea > all-corpora-dev.csv
csvstack *.test*.csv | python ~/train/util/csvShuf.py -s geneea > all-corpora-test.csv

# Step 6: Compute statistics
python3 ~/train/topic/training/catStats.py -i all-corpora-train.csv > src-corpus-stats-train.csv 2>corpora-global-stats.txt
python3 ~/train/topic/training/catStats.py -i all-corpora-dev.csv > src-corpus-stats-dev.csv 2>>corpora-global-stats.txt
python3 ~/train/topic/training/catStats.py -i all-corpora-test.csv > src-corpus-stats-test.csv 2>>corpora-global-stats.txt

# Step 7: Extract entities
python3 extract_entities.py chrono-corpora-global -o entities-global.tsv

# Step 8: Add entities to CSV files
python3 add_entities_to_csv.py all-corpora-train.csv entities-global.tsv \
    -o all-corpora-train-with-entities.csv --id-column id
python3 add_entities_to_csv.py all-corpora-dev.csv entities-global.tsv \
    -o all-corpora-dev-with-entities.csv --id-column id
python3 add_entities_to_csv.py all-corpora-test.csv entities-global.tsv \
    -o all-corpora-test-with-entities.csv --id-column id
```

## Output Files

After completing the pipeline, you will have:

- **Merged CSV files**: `all-corpora-train.csv`, `all-corpora-dev.csv`, `all-corpora-test.csv`
- **CSV files with entities**: `all-corpora-train-with-entities.csv`, `all-corpora-dev-with-entities.csv`, `all-corpora-test-with-entities.csv`
- **Statistics files**: `src-corpus-stats-train.csv`, `src-corpus-stats-dev.csv`, `src-corpus-stats-test.csv`, `corpora-global-stats.txt`
- **Entities mapping**: `entities.tsv` (or custom name)

## Notes

- The `entities` column in the final CSV files contains JSON arrays of entities for each article
- Articles without matching entities will have an empty array `[]` in the entities column
- The chronological splits ensure temporal ordering: oldest articles in train, newest in test
- Global splits may result in some datasets having empty files in certain splits (this is expected)
- Per-dataset splits ensure each dataset has articles in all splits, but may not maintain global temporal ordering
