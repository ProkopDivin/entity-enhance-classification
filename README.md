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
- `src/iptc_entity_pipeline/pipeline.py` - ClearML components and orchestration.
  - `load_data`
  - `prepare_article_embeddings`
  - `link_embeddings_and_build_datasets`
  - `train_classification_model`
  - `evaluate_classification_model`
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
- `--config` / `-c`: config variant to run:
  - `debug`
  - `wpentities`
  - `wpentities_weighted_mean`
  - `wpentities_relevance_weighted_sum`
  - `wpentities_mention_weighted_sum`
  - `article_only`
  - `wpentities_en_nl`
  - `wpentities_nl`
  - `wpentities_all_langs`
  - `wpentities_rel_th_5`
  - `best_wpentities`
  - `best_article_only`

## Wikipedia2Vec Embedding Prefetch

Use the downloader to precompute entity embeddings as `{QID}_{lang}_1.npy` + `{QID}_{lang}_1.json`.

Single language (backward compatible):

```bash
python3 -m iptc_entity_pipeline.download_wikipedia2vec_embeddings \
  --lang en
```

Multiple languages in one run (titles fetched for all requested languages per batch):

```bash
python3 -m iptc_entity_pipeline.download_wikipedia2vec_embeddings \
  --lang en de cs \
  --model-url-map en=http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_win10_500d.pkl.bz2 \
  --model-url-map de=http://wikipedia2vec.s3.amazonaws.com/models/de/2018-04-20/dewiki_20180420_win10_500d.pkl.bz2 \
  --model-url-map cs=http://wikipedia2vec.s3.amazonaws.com/models/cs/2018-04-20/cswiki_20180420_win10_500d.pkl.bz2 \
  --model-name-map en=enwiki_20180420_win10_500d \
  --model-name-map de=dewiki_20180420_win10_500d \
  --model-name-map cs=cswiki_20180420_win10_500d \
  --dump-date-map en=2018-04-20 \
  --dump-date-map de=2018-04-20 \
  --dump-date-map cs=2018-04-20
```

Notes:

- For multi-language runs, provide `--model-url-map`, `--model-name-map`, and `--dump-date-map` for each requested
  language.
- Cache file `_qid_to_title.tsv` stores language-aware rows
  (`QID<TAB>lang<TAB>status<TAB>title`) where `status` is `ok` or `no_sitelink`.
- If a `(QID, lang)` pair is absent from cache, it means that language was not fetched for that QID in previous runs.
- Storage format and naming remain unchanged and compatible with `EntityEmbeddingStore`.

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


Run new worker(agent)

```sh
    env CLEARML_AGENT_SKIP_PIP_VENV_INSTALL=/home/prokop/Git/entity-enhance-classification/venv/bin/python  clearml-agent daemon --queue iptc_entity_tasks --detached
```
```sh
    env CLEARML_AGENT_SKIP_PIP_VENV_INSTALL=/home/prokop/Git/entity-enhance-classification/venv/bin/python  clearml-agent daemon --queue iptc_entity_pipeline --detached
```
### 3.A To kill agent use:
   ```sh
   clearml-agent daemon --stop
   ```


python3 -m iptc_entity_pipeline.run_pipeline --local --config article_only