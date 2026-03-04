# EntityEnhanceClassification

Repository for diploma-thesis experiments on improving cross-language IPTC multi-label classification with
entity-enhanced inputs.

## IPTC Entity-Enhanced ClearML Pipeline (v1)

The implemented pipeline trains and evaluates an IPTC classifier using:

- article text (`title`, `lead`, `text`)
- article embedding (`<article_id>.npy`, computed on demand if missing)
- linked Wikidata entities (`data/article_2_entities.tsv`)
- entity embeddings from `.npy` files (`{wdid}_en_{chunk}.npy`)

v1 settings:

- entity pooling: sum (after per-entity chunk averaging)
- article/entity combination: concatenation
- model training and evaluation logic: copied from the original IPTC ClearML pipeline internals
- article embedding backend default: origin-compatible service mode (`SvcTextVectorizer` via `DocVectorizer`)

## Project Structure

- `src/iptc_entity_pipeline/config.py` - dataclass configuration.
- `src/iptc_entity_pipeline/data_loading.py` - corpora loading + category normalization + wdId extraction.
- `src/iptc_entity_pipeline/article_embeddings.py` - article embedding cache and fallback compute.
- `src/iptc_entity_pipeline/entity_embeddings.py` - `EntityEmbeddingStore` with chunk averaging and cache.
- `src/iptc_entity_pipeline/pooling.py` - pooling strategy interface + v1 sum pooling.
- `src/iptc_entity_pipeline/feature_builder.py` - creates concatenated article+entity vectors.
- `src/iptc_entity_pipeline/dataset_builder.py` - converts vectors to `EmbeddingDataset`.
- `src/iptc_entity_pipeline/legacy_reuse.py` - reused original model/train/eval functions.
- `src/iptc_entity_pipeline/clearml_pipeline.py` - ClearML components and orchestration.
  - `load_data`
  - `prepare_article_embeddings`
  - `prepare_entity_embeddings`
  - `link_embeddings_and_build_datasets`
- `src/iptc_entity_pipeline/run_pipeline.py` - CLI entrypoint.
- `notes.md` - architecture and implementation decisions.

## Data Layout

Default paths are hardcoded (origin-style) to:

- `/home/prokop/Git/entity-enhance-classification/data/origin-corpora/all-corpora-train.csv`
- `/home/prokop/Git/entity-enhance-classification/data/origin-corpora/all-corpora-dev.csv`
- `/home/prokop/Git/entity-enhance-classification/data/origin-corpora/all-corpora-test.csv`
- `/home/prokop/Git/entity-enhance-classification/data/article_2_entities.tsv`
- `/home/prokop/Git/entity-enhance-classification/data/entity_embeddings/WikidataProject/`
- `/home/prokop/Git/entity-enhance-classification/data/article_embeddings/` (created automatically)

Article embedding computation defaults:

- backend: `origin_service`
- model id: `paraphrase-multilingual-MiniLM-L12-v2-300-0.3`
- service URL: `http://tau.g:5533`
- dimension: `384`

Entity embeddings must be `.npy` files named as:

`{wdid}_en_{chunk}.npy`

Examples:

- `Q1000033_en_1.npy`
- `Q1000033_en_2.npy`

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Notes:

- This pipeline reuses `geneea.*` model/evaluation data structures. Install internal `geneea` packages in your
  environment before running training.
- ClearML agent/execution queue configuration is expected in your environment.

## Usage

Run locally from repository root:

```bash
python3 -m iptc_entity_pipeline.run_pipeline --local
```

Useful arguments:

- `--task-name`: ClearML task name (default: `iptc-entity-enhanced-v1`)

## What Is Reused vs New

Reused (copied with behavior preserved from original pipeline):

- `createClassificationModel`
- `trainClassificationModel`
- `evaluateModel`

New components added in this repository:

- `EntityEmbeddingStore`
- article embedding provider with origin-compatible service mode and local sentence-transformers fallback
- entity pooling strategy interface (`SumEntityPooling` in v1)
- feature builder for concatenated article+entity vectors
- ClearML orchestration for the entity-enhanced input path


Run new worker

```sh
    env CLEARML_AGENT_SKIP_PIP_VENV_INSTALL=/home/prokop/Git/entity-enhance-classification/venv/bin/python  clearml-agent daemon --queue iptc_entity_tasks --detached
```