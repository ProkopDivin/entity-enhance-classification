'''Default paths and settings for entity embedding preparation.'''

from entity_embeddings.jina_embed import JinaModelVariant, TaskKind

DEFAULT_IDS_PATH = 'data/gold-chrono-per-dataset/wdId_ids.txt'
DEFAULT_OUT_DIR = 'data/entity_embeddings/WikidataProject'
DEFAULT_LANGS = ('en', 'cs', 'nl', 'fr', 'de', 'es')
DEFAULT_SPARQL_URL = 'https://query.wikidata.org/sparql'
DEFAULT_SPARQL_ACCEPT = 'application/sparql-results+json'
DEFAULT_SPARQL_USER_AGENT = (
    'entity-enhance-classification/1.0 '
    '(https://github.com/your-username/entity-enhance-classification)'
)
DEFAULT_EMBED_SVC_URL = ''
DEFAULT_SVC_EMBED_DIM = 384
DEFAULT_JINA_EMBED_DIM = 1024
DEFAULT_SPARQL_BATCH_SIZE = 100
DEFAULT_EMBED_BACKEND = 'jina'
DEFAULT_JINA_VARIANT = JinaModelVariant.JINA_V3.value
DEFAULT_JINA_TASK: TaskKind = 'passage'
DEFAULT_TEXT_DIR = 'data/cuted-articles'
DEFAULT_INPUT_SOURCE = 'wikidata'
DEFAULT_EMBED_BATCH_SIZE = 32
