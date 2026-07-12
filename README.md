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
  - `entity_only`
  - `wpentities_en_nl`
  - `wpentities_nl`
  - `wpentities_all_langs`
  - `wpentities_rel_th_5`
  - `best_wpentities`
  - `best_article_only`
  - `wikipedia2vec_entities_all_langs`
  - `wikidata_description_entities`

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



python3 -m iptc_entity_pipeline.run_pipeline --local --config entity_only

### Entity embedding preparation

Every entity-enhanced config reads pre-computed entity vectors from a directory of
`.npy` files (one per entity/language/chunk) plus JSON metadata sidecars. The loader
(`EntityEmbeddingStore`) expects the filename convention:

```
{QID}_{lang}_{chunk}.npy      # e.g. Q42_en_1.npy
{QID}_{lang}_{chunk}.json     # sidecar with model/metadata
```

Which directory a run reads is set per config via `PathsCnf.entity_embeddings_dir`
(see `src/iptc_entity_pipeline/config.py`). The table below maps each entity source to
the directory the configs expect and how to produce it.

| Entity source | Config default dir (`data/entity_embeddings/...`) | How to produce |
|---------------|---------------------------------------------------|----------------|
| WikiProject (Wikidata graph text) | `WikidataProject` | External repo (below) |
| Wikipedia2Vec | `wikipedia2vec` | `entity_embeddings.wikipedia2vec` (below) |
| Wikidata description | `WikidataDescription_jina`, `WikidataDescription` | `entity_embeddings` (`--input-source wikidata`) |
| Wikipedia intro | `cuted-article-embeddings` | `entity_embeddings` (`--input-source files`) |
| Whole Wikipedia article | `selected-article-embeddings` | `entity_embeddings` (`--input-source files`) |
| Shared MiniLM space (PMM) | `entity_embeddings_pmm` | `entity_embeddings` (`--embed-backend svc`) |

The QID list consumed by all generators is `data/gold-chrono-per-dataset/wdId_ids.txt`
(one Wikidata QID per line).

#### Prerequisites

Install the preprocessing dependencies (includes `wikipedia2vec`, which is **not** part
of the main pipeline install):

```bash
pip install -r data-preprocessing/requirements.txt
```

All generators are run as modules with the preprocessing sources on `PYTHONPATH`:

```bash
export PYTHONPATH=data-preprocessing/src:src
```

#### 1. WikiProject entities

These are produced by a separate project. Follow its instructions and place the output
under `data/entity_embeddings/WikidataProject/`:
[Wikidata Embedding Project](https://github.com/ProkopDivin/WikidataTextEmbedding).

#### 2. Wikipedia2Vec entities

This generator downloads the pretrained Wikipedia2Vec model automatically on first run
(`enwiki_20180420_win10_500d.pkl.bz2`, ~2 GB, from `wikipedia2vec.s3.amazonaws.com`) into
`data/wikipedia2vec_models/`, decompresses it once, then writes one vector per entity:

```bash
PYTHONPATH=data-preprocessing/src:src python -m entity_embeddings.wikipedia2vec \
  --lang en \
  --out-dir data/entity_embeddings/wikipedia2vec
```

Requires internet access and several GB of free disk. Use `--skip-download` if the model
is already present, and `--titles-only` to resolve Wikidata titles without computing
vectors. For multiple languages, provide per-language model URLs/names/dump dates via
`--model-url-map`, `--model-name-map`, and `--dump-date-map` (see
`docs/wikipedia2vec_entity_embeddings.md`).

#### 3. Text-based entities (Jina / embedding service)

`entity_embeddings` embeds entity *text* — either Wikidata descriptions fetched over
SPARQL, or local text files (Wikipedia intros / full articles). Backend defaults to a
local Jina model; use `--embed-backend svc` for the Geneea embedding service (needed for
the shared-MiniLM `entity_embeddings_pmm` variant).

Wikidata descriptions (Jina v3):

```bash
PYTHONPATH=data-preprocessing/src:src python -m entity_embeddings \
  --input-source wikidata \
  --ids data/gold-chrono-per-dataset/wdId_ids.txt \
  --out-dir data/entity_embeddings/WikidataDescription_jina \
  --variant jina-v3 --task passage --langs en,cs,de
```

Wikipedia intros from local cut-text files:

```bash
PYTHONPATH=data-preprocessing/src:src python -m entity_embeddings \
  --input-source files \
  --text-dir data/cuted-articles \
  --out-dir data/entity_embeddings/cuted-article-embeddings \
  --variant jina-v3 --task passage --langs en --skip-existing
```

Run `python -m entity_embeddings --help` for the full list of backends, variants, and
tasks. All generators emit the same `{QID}_{lang}_{chunk}.npy` + `.json` layout consumed
by the pipeline.