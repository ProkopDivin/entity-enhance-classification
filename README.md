# EntityEnhanceClassification

Repository for diploma-thesis experiments on improving cross-language IPTC multi-label classification with
entity-enhanced inputs.

## Quick start (debug run)

Use the `debug` config for a fast end-to-end smoke test on a tiny dataset. It runs the full
pipeline locally (load data → embeddings → 5-fold CV → train → test eval) with reduced
training time.

### What `debug` uses

Defined in `src/iptc_entity_pipeline/config.py` (`DebugCnf`):

- **Train/test corpora:** `data/debug/all-corpora-*.sample_4plus1.csv` (small sample)
- **Entity mapping:** `data/wd-id_mapping_debug.tsv`
- **Entity embeddings:** `data/entity_embeddings/debug/`
- **Article embeddings:** `data/article_embeddings_debug`
- **Training:** 5 epochs, MLP with relevance-weighted entity pooling
- **CV / HPO:** 5 folds, small grid over `dropouts2` (3 trials)

### Prerequisites

- Python 3.10+
- ClearML installed and credentials configured only for non-local pipeline mode

### Run

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --find-links=wheels -e .

python3 -m iptc_entity_pipeline.run_pipeline --local --config debug
```

`--local` runs even when `clearml` is not installed.
Non-local mode still requires ClearML:

```bash
python3 -m iptc_entity_pipeline.run_pipeline --config debug
```


### Output

- Progress is logged to the terminal (6 pipeline stages).
- Metrics are reported to ClearML when running with ClearML available.
- Saved model and evaluation artifacts are written under
  `results/saved_models/debug_<timestamp>/` (model weights, thresholds, test tables).


## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --find-links=wheels -e .
```

The `--find-links=wheels` flag tells pip to look in the `wheels/` directory for the
bundled `geneea` dependency (a pure Python wheel shipped with this repo).

Notes:

- Install `clearml` and configure credentials only when you run non-local queue mode.
- ClearML agent/execution queue configuration is expected in your environment.

## ClearML Setup

ClearML is **optional**. The pipeline runs fully offline with the `--local` flag -- no
ClearML server, credentials, or agent required. Use `--local` for development, debugging,
and standalone experiments.

When you want experiment tracking, metric logging, and remote execution via queues,
set up ClearML:

1. **Create a free account** at [app.clear.ml](https://app.clear.ml) (or use a
   self-hosted server).

2. **Install the SDK** (already included in this project's dependencies):

   ```bash
   pip install clearml
   ```

3. **Generate credentials** -- go to *Settings > Workspace > Create new credentials*
   in the ClearML web UI
   ([docs](https://clear.ml/docs/latest/docs/getting_started/ds/ds_first_steps/#connect-clearml-sdk-to-the-server)).

4. **Configure locally** -- run `clearml-init` and paste the credentials when prompted:

   ```bash
   clearml-init
   ```

   This writes `~/clearml.conf` with your API key, secret, and server URL.

5. **Run with ClearML** (omit `--local`):

   ```bash
   python3 -m iptc_entity_pipeline.run_pipeline --config debug
   ```

   Metrics, parameters, and artifacts are now logged to your ClearML project.

For remote queue execution with a ClearML agent, see the
[ClearML Agent docs](https://clear.ml/docs/latest/docs/clearml_agent/).

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