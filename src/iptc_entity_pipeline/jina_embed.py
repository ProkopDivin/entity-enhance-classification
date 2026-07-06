'''
Standalone Jina embedding utilities with lazy model setup.

Variants:
- jina-v3 (default): retrieval.passage / retrieval.query via transformers
- jina-v3-classification: classification task on v3
- jina-v5: retrieval query/document prompts (same roles as jina-v3)
- jina-v5-classification: classification embeddings on v5
'''

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Literal, Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

TaskKind = Literal['passage', 'query', 'classification']


class JinaModelVariant(str, Enum):
    JINA_V3 = 'jina-v3'
    JINA_V3_CLASSIFICATION = 'jina-v3-classification'
    JINA_V5 = 'jina-v5'
    JINA_V5_CLASSIFICATION = 'jina-v5-classification'


class BackendKind(str, Enum):
    TRANSFORMERS = 'transformers'
    SENTENCE_TRANSFORMERS = 'sentence_transformers'


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    backend: BackendKind
    default_task: str
    query_task: str | None
    passage_prompt: str | None
    query_prompt: str | None
    embedding_dim: int


MODEL_CONFIGS: dict[JinaModelVariant, ModelConfig] = {
    JinaModelVariant.JINA_V3: ModelConfig(
        model_id='jinaai/jina-embeddings-v3',
        backend=BackendKind.TRANSFORMERS,
        default_task='retrieval.passage',
        query_task='retrieval.query',
        passage_prompt=None,
        query_prompt=None,
        embedding_dim=1024,
    ),
    JinaModelVariant.JINA_V3_CLASSIFICATION: ModelConfig(
        model_id='jinaai/jina-embeddings-v3',
        backend=BackendKind.TRANSFORMERS,
        default_task='classification',
        query_task=None,
        passage_prompt=None,
        query_prompt=None,
        embedding_dim=1024,
    ),
    JinaModelVariant.JINA_V5: ModelConfig(
        model_id='jinaai/jina-embeddings-v5-text-small-retrieval',
        backend=BackendKind.SENTENCE_TRANSFORMERS,
        default_task='retrieval',
        query_task='retrieval',
        passage_prompt='document',
        query_prompt='query',
        embedding_dim=1024,
    ),
    JinaModelVariant.JINA_V5_CLASSIFICATION: ModelConfig(
        model_id='jinaai/jina-embeddings-v5-text-small-classification',
        backend=BackendKind.SENTENCE_TRANSFORMERS,
        default_task='classification',
        query_task=None,
        passage_prompt=None,
        query_prompt=None,
        embedding_dim=1024,
    ),
}


@dataclass
class _LoadedModel:
    model: AutoModel | SentenceTransformer
    tokenizer: AutoTokenizer | None
    device: str
    config: ModelConfig


def _resolve_variant(variant: str | JinaModelVariant) -> JinaModelVariant:
    if isinstance(variant, JinaModelVariant):
        return variant
    try:
        return JinaModelVariant(variant)
    except ValueError as exc:
        supported = ', '.join(v.value for v in JinaModelVariant)
        raise ValueError(f'Unknown variant {variant!r}. Supported: {supported}') from exc


def _resolve_task(*, config: ModelConfig, task_kind: TaskKind) -> str:
    if task_kind == 'query':
        if config.query_task is None:
            raise ValueError(
                f'Variant {config.model_id} does not support query embeddings. '
                'Use passage or classification, or switch to jina-v3 / jina-v5.'
            )
        return config.query_task
    if task_kind == 'classification':
        return 'classification'
    return config.default_task


def _resolve_prompt_name(*, config: ModelConfig, task_kind: TaskKind) -> str | None:
    if task_kind == 'query':
        if config.query_prompt is None:
            raise ValueError(
                f'Variant {config.model_id} does not support query embeddings. '
                'Use passage or classification, or switch to jina-v3 / jina-v5.'
            )
        return config.query_prompt
    if task_kind == 'classification':
        return None
    return config.passage_prompt


def _load_model(*, config: ModelConfig, device: str) -> tuple[AutoModel | SentenceTransformer, AutoTokenizer | None]:
    if config.backend == BackendKind.TRANSFORMERS:
        model = AutoModel.from_pretrained(config.model_id, trust_remote_code=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=True)
        model.eval()
        return model, tokenizer

    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError(
            'sentence-transformers is required for jina-v5 variants. '
            'Install it with: pip install sentence-transformers'
        )

    model_kwargs: dict = {}
    if device == 'cuda':
        model_kwargs['dtype'] = torch.bfloat16

    model = SentenceTransformer(
        config.model_id,
        trust_remote_code=True,
        device=device,
        model_kwargs=model_kwargs,
    )
    return model, None


def _encode_transformers(
    *,
    model: AutoModel,
    texts: Sequence[str],
    task: str,
    embedding_dim: int,
) -> list[list[float]]:
    with torch.no_grad():
        embeddings = model.encode(list(texts), task=task, truncate_dim=embedding_dim)
    if isinstance(embeddings, np.ndarray):
        return embeddings.tolist()
    return embeddings


def _encode_sentence_transformers(
    *,
    model: SentenceTransformer,
    texts: Sequence[str],
    prompt_name: str | None,
    embedding_dim: int,
) -> list[list[float]]:
    encode_kwargs: dict = {'truncate_dim': embedding_dim}
    if prompt_name is not None:
        encode_kwargs['prompt_name'] = prompt_name

    embeddings = model.encode(list(texts), **encode_kwargs)
    if isinstance(embeddings, np.ndarray):
        return embeddings.tolist()
    return embeddings


@lru_cache(maxsize=None)
def setup_model(variant: str | JinaModelVariant = JinaModelVariant.JINA_V3) -> _LoadedModel:
    '''
    Load and cache the Jina embedding model for the given variant.

    Downloads from Hugging Face on first use; later calls reuse the HF cache.

    :param variant: Model variant key (default: jina-v3)
    :return: Loaded model bundle
    '''
    resolved = _resolve_variant(variant)
    config = MODEL_CONFIGS[resolved]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer = _load_model(config=config, device=device)
    return _LoadedModel(model=model, tokenizer=tokenizer, device=device, config=config)


def embed_texts(
    texts: Sequence[str],
    *,
    variant: str | JinaModelVariant = JinaModelVariant.JINA_V3,
    task_kind: TaskKind = 'passage',
    embedding_dim: int | None = None,
) -> list[list[float]]:
    '''
    Compute embeddings for one or more texts.

    :param texts: Input strings to embed
    :param variant: Model variant (default: jina-v3)
    :param task_kind: passage (documents), query (search), or classification
    :param embedding_dim: Output dimension; defaults to variant config
    :return: List of embedding vectors
    '''
    if not texts:
        return []

    loaded = setup_model(variant)
    dim = embedding_dim or loaded.config.embedding_dim

    if loaded.config.backend == BackendKind.TRANSFORMERS:
        task = _resolve_task(config=loaded.config, task_kind=task_kind)
        return _encode_transformers(
            model=loaded.model,
            texts=texts,
            task=task,
            embedding_dim=dim,
        )

    prompt_name = _resolve_prompt_name(config=loaded.config, task_kind=task_kind)
    return _encode_sentence_transformers(
        model=loaded.model,
        texts=texts,
        prompt_name=prompt_name,
        embedding_dim=dim,
    )


def embed_query(
    text: str,
    *,
    variant: str | JinaModelVariant = JinaModelVariant.JINA_V3,
    embedding_dim: int | None = None,
) -> list[float]:
    '''
    Embed a single search query (jina-v3 and jina-v5 only).

    :param text: Query text
    :param variant: Model variant (default: jina-v3)
    :param embedding_dim: Output dimension; defaults to variant config
    :return: Embedding vector
    '''
    return embed_texts(
        [text],
        variant=variant,
        task_kind='query',
        embedding_dim=embedding_dim,
    )[0]


def clear_model_cache() -> None:
    '''Clear the in-memory model cache (useful when switching variants in one process).'''
    setup_model.cache_clear()


class JinaTextVectorizer:
    '''
    Adapter exposing ``toMatrix`` for pipelines that batch-embed text via Jina models.

    Wikidata entity descriptions should use ``task_kind='passage'`` (retrieval.passage).
    Use ``task_kind='query'`` when embedding search queries against those passages.
    '''

    def __init__(
        self,
        *,
        variant: str | JinaModelVariant = JinaModelVariant.JINA_V3,
        task_kind: TaskKind = 'passage',
        embedding_dim: int | None = None,
    ) -> None:
        self._variant = variant
        self._task_kind = task_kind
        self._embedding_dim = embedding_dim

    @property
    def model_name(self) -> str:
        '''
        Stable model identifier for embedding metadata sidecars.

        :return: ``{variant}:{task_kind}`` string
        '''
        resolved = _resolve_variant(self._variant)
        return f'{resolved.value}:{self._task_kind}'

    def toMatrix(self, texts: Sequence[str]) -> np.ndarray:
        '''
        Embed texts and return a dense matrix compatible with ``SvcTextVectorizer``.

        :param texts: Input strings
        :return: Embedding matrix with shape ``(len(texts), embedding_dim)``
        '''
        vectors = embed_texts(
            texts,
            variant=self._variant,
            task_kind=self._task_kind,
            embedding_dim=self._embedding_dim,
        )
        return np.asarray(vectors, dtype=np.float64)
