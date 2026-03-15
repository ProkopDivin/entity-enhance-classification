"""Configuration dataclasses for the IPTC entity-enhanced pipeline."""

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

DATA_ROOT = '/home/prokop/Git/entity-enhance-classification/data'


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem paths for data and artifacts."""

    train_csv: str = f'{DATA_ROOT}/origin-corpora/all-corpora-train.csv'
    dev_csv: str = f'{DATA_ROOT}/origin-corpora/all-corpora-dev.csv'
    test_csv: str = f'{DATA_ROOT}/origin-corpora/all-corpora-test.csv'
    article_entities_tsv: str = f'{DATA_ROOT}/article_2_entities.tsv'
    article_embeddings_dir: str = f'{DATA_ROOT}/article_embeddings'
    entity_embeddings_dir: str = f'{DATA_ROOT}/entity_embeddings/WikidataProject'
    removed_cat_ids: list[str] = field(default_factory=lambda: ['20000419'])


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding loading and fallback-computation parameters."""

    article_embedding_backend: str = 'origin_service'
    article_model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2-300-0.3'
    article_embedding_dim: int = 384
    embed_svc_url: str = 'http://tau.g:5533'
    entity_lang: str = 'en'
    use_entity_embeddings: bool = True
    combine_method: str = 'concat'
    entity_pooling: str = 'sum'


@dataclass(frozen=True)
class ModelConfig:
    """Model architecture parameters."""

    hidden_dim: int = 1024
    dropouts1: float = 0
    dropouts2: float = 0.3


@dataclass(frozen=True)
class TrainingConfig:
    """Training loop parameters."""

    epochs: int = 25
    batch_size: int = 64
    optimizer_name: str = 'adam'
    learning_rate: float = 0.00037
    lr_scheduler_name: str = 'stepLR'
    step_size: int = 1
    gamma: float = 1
    loss_name: str = 'bceWithLogitsLoss'


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation behavior and threshold settings."""

    threshold_predict: float = -0.3
    threshold_eval: float = 0.5
    per_corpus: bool = True
    per_class: bool = True
    averaging_type: str = 'datapoint'


@dataclass(frozen=True)
class LoggingConfig:
    """Logging behavior for ClearML components."""

    print_logs: bool = True


@dataclass(frozen=True)
class BaseConfig:
    """Top-level pipeline config grouped by concern."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    objective_corpora: str = 'All-datapoint'

    def to_clearml_mapping(self) -> dict[str, Any]:
        """Convert dataclasses to serializable mapping."""
        return asdict(self)


def resolve_paths(config: BaseConfig, root_dir: str | Path) -> BaseConfig:
    """Return a config with absolute paths resolved from ``root_dir``."""
    root_path = Path(root_dir)
    paths = config.paths
    resolved_paths = PathsConfig(
        train_csv=str(root_path / paths.train_csv),
        dev_csv=str(root_path / paths.dev_csv),
        test_csv=str(root_path / paths.test_csv),
        article_entities_tsv=str(root_path / paths.article_entities_tsv),
        article_embeddings_dir=str(root_path / paths.article_embeddings_dir),
        entity_embeddings_dir=str(root_path / paths.entity_embeddings_dir),
        removed_cat_ids=paths.removed_cat_ids,
    )
    return BaseConfig(
        paths=resolved_paths,
        embeddings=config.embeddings,
        model=config.model,
        training=config.training,
        evaluation=config.evaluation,
        logging=config.logging,
        objective_corpora=config.objective_corpora,
    )


def to_article_only_config(config: BaseConfig) -> BaseConfig:
    """
    Return a config variant that uses only article embeddings.

    :param config: Base config to adapt.
    :return: Config with entity embeddings disabled.
    """
    return replace(
        config,
        embeddings=replace(
            config.embeddings,
            use_entity_embeddings=False,
        ),
    )


def get_config(config_name: str) -> BaseConfig:
    """
    Return config variant by name.

    Supported names:
    - ``base``: entity-enhanced default setup.
    - ``article_only``: article embeddings only (entity embeddings disabled).

    :param config_name: Config variant name.
    :return: Selected config object.
    :raises ValueError: If ``config_name`` is unknown.
    """
    name = config_name.strip().lower()
    if name == 'base':
        return BaseConfig()
    if name == 'article_only':
        return to_article_only_config(config=BaseConfig())
    raise ValueError(f'Unsupported config_name: {config_name}')


def list_config_names() -> tuple[str, ...]:
    """
    Return names of supported config variants.

    :return: Tuple of supported config names.
    """
    return ('base', 'article_only')


# Backward compatibility alias for older imports.
PipelineConfig = BaseConfig

