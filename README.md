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
cd entity-enhance-classification/
python3 -m venv .venv
source .venv/bin/activate 
pip install --upgrade pip
pip install --find-links=wheels -e .

python3 -m iptc_entity_pipeline.run_pipeline --local --config debug
```
- for Windows: `.\.venv\Scripts\activate`

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
cd entity-enhance-classification/
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --find-links=wheels -e .
```

The `--find-links=wheels` flag tells pip to look in the `wheels/` directory for the
bundled `geneea` dependency (a pure Python wheel shipped with this repo).

## ClearML Setup (optional)

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

# Making your own input

Due to legal constraints the original corpora cannot be distributed. This section
describes how to prepare your own input from raw text so that the pipeline can
consume it. You need three things:

1. **Corpus CSVs** (train + test) — article text with IPTC labels and entity annotations
2. **Wikidata ID mapping TSV** — maps internal entity IDs to Wikidata QIDs - maping entities to their embeddings
3. **Article embeddings** — one `.npy` vector per article (pre-computed)
4. Entity embeddings — `.npy` vector per entity embedding source (pre-computed)



## 1. Corpus CSV format

The pipeline reads standard CSV files via `Corpus.fromCsv`. Required columns:


| Column     | Description                              |
| ---------- | ---------------------------------------- |
| `id`       | Unique article identifier                |
| `title`    | Article title (optional, can be empty)   |
| `lead`     | Article lead/summary (optional)          |
| `text`     | Full article body                        |
| `cats`     | IPTC category IDs, pipe-delimited        |
| `entities` | JSON array of entity objects (see below) |


**Minimal example** (`my_corpus_train.csv`):

```csv
id,title,lead,text,cats,entities
art_001,Floods hit region,,Heavy floods struck the Danube region today...,06000000|06007000,"[{""gkbId"":""e_123"",""type"":""location"",""relevance"":0.9,""mentions"":[""Danube""]}]"
art_002,New CEO appointed,,Acme Corp announced a new CEO...,04000000,"[{""gkbId"":""e_456"",""type"":""person"",""relevance"":0.8,""mentions"":[""John Smith""]},{""gkbId"":""e_789"",""type"":""organization"",""relevance"":0.7,""mentions"":[""Acme Corp""]}]"
```

Each entity object in the JSON array supports:


| Field       | Required | Description                                                                 |
| ----------- | -------- | --------------------------------------------------------------------------- |
| `gkbId`     | yes*     | Internal entity ID (resolved via the mapping TSV)                           |
| `wdId`      | yes*     | Wikidata QID directly (alternative to `gkbId` + mapping)                    |
| `type`      | no       | One of: `person`, `organization`, `location`, `event`, `product`, `general` |
| `relevance` | no       | Float 0–1 indicating entity importance in the article                       |
| `mentions`  | no       | List of surface forms; length = mention count                               |


*Provide either `gkbId` (with a mapping TSV) or `wdId` / `wdids` inline.

## 2. Wikidata ID mapping TSV

Tab-separated file mapping your internal entity IDs to Wikidata QIDs.


| Column         | Description                     |
| -------------- | ------------------------------- |
| `gkb_id`       | Your internal entity identifier |
| `wikidata_ids` | Space-separated Wikidata QIDs   |


**Example** (`wdId_mapping.tsv`):

```tsv
gkb_id	wikidata_ids
e_123	Q1653
e_456	Q42
e_789	Q312
```



## 3. Article embeddings

The pipeline expects one pre-computed embedding per article stored as
`<article_id>.npy` in a flat directory. Our default model was
`paraphrase-multilingual-MiniLM-L12-v2` (384-dim).

We provide sample of computed articles embeddings in `data/article_embeddings_debug`

```python
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
vec = model.encode('Heavy floods struck the Danube region today...')
np.save('data/article_embeddings/art_001.npy', vec)
```



## 4. Pointing the config at your data

Override `PathsCnf` in a custom config or pass paths via environment:

```python
from iptc_entity_pipeline.config import BaseCnf, PathsCnf
from dataclasses import field, replace

@dataclass(frozen=True)
class MyDataCnf(BaseCnf):
    paths: PathsCnf = field(default_factory=lambda: replace(
        PathsCnf(),
        train_csv='data/my_corpus_train.csv',
        test_csv='data/my_corpus_test.csv',
        wdid_mapping_tsv='data/my_wdId_mapping.tsv',
        article_embeddings_dir='data/my_article_embeddings',
    ))
```

Then run:

```bash
python3 -m iptc_entity_pipeline.run_pipeline --local --config my_data
```



## Entity embedding preparation

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


| Entity source                                | Config default dir (`data/entity_embeddings/...`) | How to produce                                  |
| -------------------------------------------- | ------------------------------------------------- | ----------------------------------------------- |
| WikiProject (Wikidata graph/text projection) | `WikidataProject`                                 | External repo (below)                           |
| Wikipedia2Vec                                | `wikipedia2vec`                                   | `entity_embeddings.wikipedia2vec` (below)       |
| Text-based entity embeddings                 | `WikidataDescription_jina`, `WikidataDescription` | `entity_embeddings` (`--input-source wikidata`) |
| Wikipedia intro text                         | `cuted-article-embeddings`                        | `entity_embeddings` (`--input-source files`)    |
| Whole Wikipedia article text                 | `selected-article-embeddings`                     | `entity_embeddings` (`--input-source files`)    |


The QID list consumed by all generators is `data/gold-chrono-per-dataset/wdId_ids.txt`
(one Wikidata QID per line).

### Important constraint (`tau.g` / `psi.g` unavailable)

If your environment cannot access internal services such as `tau.g` or `psi.g`, use only
the local/offline-compatible path in this repository:

1. Build `gkb_id -> wikidata_ids` locally from your corpus entities:
  - `data-preprocessing/src/extract_wikidata_mapping.py`
  - Run without `--url` to use deterministic `G... -> Q...` conversion.
2. Generate text representations from public or local sources:
  - Wikidata descriptions (`--input-source wikidata`)
  - Local text files (`--input-source files`) for intros/full pages
3. Embed the generated text with the built-in embedding module (`entity_embeddings`).

This fully substitutes private infrastructure and still produces vectors in the format
required by the training pipeline.

#### Prerequisites

Install the preprocessing package (includes `wikipedia2vec`, which is **not** part
of the main pipeline install):

```bash
pip install -e data-preprocessing
```

Install the `entity_embeddings` package in editable mode (once):

```bash
pip install -e data-preprocessing
```



### 1) Entities from WikiProject

These are produced by a separate project, which we fork and adapted for our purpose. Follow its instructions and place the output
under `data/entity_embeddings/WikidataProject/`:
[Wikidata Embedding Project](https://github.com/ProkopDivin/WikidataTextEmbedding).

Warning: This project requires at least 120 GB of free disk space, because it downloads part of Wikidata dump.

### 2) Entities from Wikipedia2Vec

This generator downloads the pretrained Wikipedia2Vec model automatically on first run
(`enwiki_20180420_win10_500d.pkl.bz2`, ~20 GB, from `wikipedia2vec.s3.amazonaws.com`) into
`data/wikipedia2vec_models/`, decompresses it once, then writes one vector per entity:

Warning: even with a small input, the first run can take several minutes because the model must be downloaded.

```bash
python -m entity_embeddings.wikipedia2vec \
  --ids-path data/gold-chrono-per-dataset/wdId_ids_sample.txt \
  --lang en \
  --out-dir data/entity_embeddings/wikipedia2vec_sample
```

Requires internet access and several GB of free disk. Use `--skip-download` if the model  
is already present, and `--titles-only` to resolve Wikidata titles without computing  
vectors. For multiple languages, provide per-language model URLs/names/dump dates via  
`--model-url-map`, `--model-name-map`, and `--dump-date-map` (run
`python -m entity_embeddings.wikipedia2vec --help` for details).

Warning: `entity_embeddings.wikipedia2vec` may repeatedly log `Wikidata maxlag=...` and pause for 30s when Wikidata replicas are overloaded. This is a temporary upstream API throttling condition (not invalid QIDs). The command is retrying automatically and may take significantly longer, or require rerunning later when Wikidata load is lower.

### 3) Entity embeddings from text representation

`entity_embeddings` embeds entity *text* — either Wikidata descriptions fetched over SPARQL, or local text files (Wikipedia intros / full articles). Backend defaults to a local Jina model;

#### 3.1 Get text representation from Wikidata descriptions

```bash
python -m entity_embeddings \
  --input-source wikidata \
  --ids data/gold-chrono-per-dataset/wdId_ids_sample.txt \
  --out-dir data/entity_embeddings/WikidataDescription_jina_sample \
  --variant jina-v3 --task passage --langs en,cs,de
```



#### 3.2 Get text representation from local files (intros/full text)

```bash
python -m entity_embeddings \
  --input-source files \
  --text-dir data/cuted-articles_sample \
  --out-dir data/entity_embeddings/cuted-article-embeddings_sample \
  --variant jina-v3 --task passage --langs en --skip-existing
```

Run `python -m entity_embeddings --help` for the full list of backends, variants, and tasks. All generators emit the same `{QID}_{lang}_{chunk}.npy` + `.json` layout consumed by the pipeline.

##### Getting Wikipedia text representation for entities

To generate Wikipedia article text representations for entities, we created this
[project](https://github.com/ProkopDivin/extractWikiArticles). 

Warning: This project requires at least 250 GB of free disk space to avoid running out during decompression. It processes part of the Wikipedia dump; using the public API would be too slow and impractical, for our number of entities.