"""Configuration dataclasses for the IPTC entity-enhanced pipeline."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem paths for data and artifacts."""

    train_csv: str = 'data/origin-corpora/all-corpora-train.csv'
    dev_csv: str = 'data/origin-corpora/all-corpora-dev.csv'
    test_csv: str = 'data/origin-corpora/all-corpora-test.csv'
    article_entities_tsv: str = 'data/article_2_entities.tsv'
    article_embeddings_dir: str = 'data/article_embeddings'
    entity_embeddings_dir: str = 'data/entity_embeddings/WikidataProject'
    removed_cat_ids: list[str] = field(default_factory=lambda: ['20000419'])


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding loading and fallback-computation parameters."""

    article_embedding_backend: str = 'origin_service'
    article_model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2-300-0.3'
    article_embedding_dim: int = 384
    embed_svc_url: str = 'http://tau.g:5533'
    entity_lang: str = 'en'
    combine_method: str = 'concat'
    entity_pooling: str = 'sum'


@dataclass(frozen=True)
class ModelConfig:
    """Model architecture parameters."""

    hidden_dim: int = 1024
    dropouts1: float = 0.2
    dropouts2: float = 0.2


@dataclass(frozen=True)
class TrainingConfig:
    """Training loop parameters."""

    epochs: int = 10
    batch_size: int = 64
    optimizer_name: str = 'adam'
    learning_rate: float = 1e-3
    lr_scheduler_name: str = 'stepLR'
    step_size: int = 1
    gamma: float = 0.95
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
class PipelineConfig:
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


def resolve_paths(config: PipelineConfig, root_dir: str | Path) -> PipelineConfig:
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
    return PipelineConfig(
        paths=resolved_paths,
        embeddings=config.embeddings,
        model=config.model,
        training=config.training,
        evaluation=config.evaluation,
        logging=config.logging,
        objective_corpora=config.objective_corpora,
    )

