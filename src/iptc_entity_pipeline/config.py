"""Configuration dataclasses for the IPTC entity-enhanced pipeline."""

from dataclasses import asdict, dataclass, field, fields, replace
from itertools import product
from pathlib import Path
from typing import Any, Mapping

DATA_ROOT = '/home/prokop/Git/entity-enhance-classification/data'


def config_from_dict(cls, d: Mapping[str, Any]):
    """Reconstruct a frozen dataclass from a dict, ignoring unknown keys."""
    valid_keys = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in valid_keys})


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem paths for data and artifacts."""

    train_csv: str = f'{DATA_ROOT}/gold-chrono-per-dataset/all-corpora-train-entities.csv'
    test_csv: str = f'{DATA_ROOT}/gold-chrono-per-dataset/all-corpora-test-entities.csv'
    wdid_mapping_tsv: str = f'{DATA_ROOT}/gold-chrono-per-dataset/wdId_mapping.tsv'
    article_embeddings_dir: str = f'{DATA_ROOT}/article_embeddings'
    entity_embeddings_dir: str = f'{DATA_ROOT}/entity_embeddings/WikidataProject'
    downsampling_order_cache_json: str = f'{DATA_ROOT}/downsampling_order_cache.json'
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
    """Scalar model architecture parameters for a single training run."""

    hidden_dim: int = 1024
    dropouts1: float = 0.0
    dropouts2: float = 0.3


@dataclass(frozen=True)
class TrainingConfig:
    """Scalar training loop parameters for a single training run."""

    epochs: int = 100
    batch_size: int = 100
    optimizer_name: str = 'adam'
    learning_rate: float = 0.00037
    lr_scheduler_name: str = 'stepLR'
    step_size: int = 1
    gamma: float = 1
    loss_name: str = 'bceWithLogitsLoss'
    # 0 = disabled. When > 0, monitors dev loss, stops after this many epochs
    # without improvement, and restores the best weights.
    early_stopping_patience: int = 6
    early_stopping_min_delta: float = 0.000000001


@dataclass(frozen=True)
class HyperparamSpace:
    """Grid-search space for tunable hyperparameters.

    Each list field defines candidate values to try.  Use
    :meth:`iter_combinations` to expand the full Cartesian product.
    """

    hidden_dims: list[int] = field(default_factory=lambda: [1024])
    dropouts1: list[float] = field(default_factory=lambda: [0.0])
    dropouts2: list[float] = field(default_factory=lambda: [0.3])
    batch_sizes: list[int] = field(default_factory=lambda: [100])
    learning_rates: list[float] = field(default_factory=lambda: [0.00037])

    def iter_combinations(
        self, base_training: TrainingConfig,
    ) -> list[tuple[ModelConfig, TrainingConfig]]:
        """Expand grid into all ``(ModelConfig, TrainingConfig)`` combinations.

        :param base_training: Base training config whose non-grid fields are preserved.
        :return: List of ``(ModelConfig, TrainingConfig)`` tuples.
        """
        return [
            (
                ModelConfig(hidden_dim=hd, dropouts1=d1, dropouts2=d2),
                replace(base_training, batch_size=bs, learning_rate=lr),
            )
            for hd, d1, d2, bs, lr in product(
                self.hidden_dims, self.dropouts1, self.dropouts2,
                self.batch_sizes, self.learning_rates,
            )
        ]


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation behavior and threshold settings."""

    threshold_predict: float = -9999
    threshold_eval: float = 0.5
    per_corpus: bool = True
    per_class: bool = True
    averaging_type: str = 'datapoint'
    base_probabilities_csv: str = ''


@dataclass(frozen=True)
class CvConfig:
    """Cross-validation setup."""

    folds: int = 5
    random_seed: int = 43


@dataclass(frozen=True)
class BaseConfig:
    """Top-level pipeline config grouped by concern."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    cv: CvConfig = field(default_factory=CvConfig)
    hyperparam_space: HyperparamSpace = field(default_factory=HyperparamSpace)
    objective_corpora: str = 'All-datapoint'
    downsample_corpora: dict[str, float] = field(default_factory=dict)
    print_logs: bool = True
    debug: bool = True

    def to_clearml_mapping(self) -> dict[str, Any]:
        """Convert dataclasses to serializable mapping."""
        return asdict(self)


@dataclass(frozen=True)
class WpEntitiesConfig(BaseConfig):
    """Default entity-enhanced configuration."""


@dataclass(frozen=True)
class ArticleOnlyConfig(BaseConfig):
    """Article-only configuration without entity embeddings."""

    embeddings: EmbeddingConfig = field(
        default_factory=lambda: EmbeddingConfig(
            article_embedding_backend='origin_service',
            article_model_name='paraphrase-multilingual-MiniLM-L12-v2-300-0.3',
            article_embedding_dim=384,
            embed_svc_url='http://tau.g:5533',
            entity_lang='en',
            use_entity_embeddings=False,
            combine_method='concat',
            entity_pooling='sum',
        )
    )


@dataclass(frozen=True)
class DebugConfig(BaseConfig):
    """Debug configuration for quick local runs."""

    paths: PathsConfig = field(
        default_factory=lambda: PathsConfig(
            train_csv=f'{DATA_ROOT}/debug/all-corpora-train-entities.csv',
            test_csv=f'{DATA_ROOT}/debug/all-corpora-test-entities.csv',
            wdid_mapping_tsv=f'{DATA_ROOT}/debug/wdId_mapping.tsv',
            article_embeddings_dir=f'{DATA_ROOT}/article_embeddings',
            entity_embeddings_dir=f'{DATA_ROOT}/entity_embeddings/WikidataProject',
            downsampling_order_cache_json=f'{DATA_ROOT}/downsampling_order_cache.json',
            removed_cat_ids=['20000419'],
        )
    )
    model: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            hidden_dim=1024,
            dropouts1=0.1,
            dropouts2=0.3,
        )
    )
    training: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(
            epochs=5,
            batch_size=100,
            optimizer_name='adam',
            learning_rate=0.00037,
            lr_scheduler_name='stepLR',
            step_size=1,
            gamma=1,
            loss_name='bceWithLogitsLoss',
            early_stopping_patience=6,
            early_stopping_min_delta=0.000000001,
        )
    )
    hyperparam_space: HyperparamSpace = field(
        default_factory=lambda: HyperparamSpace(
            hidden_dims=[1024],
            dropouts1=[0.1],
            dropouts2=[0.3],
            batch_sizes=[100],
            learning_rates=[0.00037],
        )
    )
    cv: CvConfig = field(default_factory=lambda: CvConfig(folds=2, random_seed=43))
    debug: bool = False


def resolve_paths(config: BaseConfig, root_dir: str | Path) -> BaseConfig:
    """Return a config with absolute paths resolved from ``root_dir``."""
    root_path = Path(root_dir)
    paths = config.paths
    resolved_paths = PathsConfig(
        train_csv=str(root_path / paths.train_csv),
        test_csv=str(root_path / paths.test_csv),
        wdid_mapping_tsv=str(root_path / paths.wdid_mapping_tsv),
        article_embeddings_dir=str(root_path / paths.article_embeddings_dir),
        entity_embeddings_dir=str(root_path / paths.entity_embeddings_dir),
        downsampling_order_cache_json=str(root_path / paths.downsampling_order_cache_json),
        removed_cat_ids=paths.removed_cat_ids,
    )
    return replace(config, paths=resolved_paths)


def _config_map() -> dict[str, BaseConfig]:
    """Return supported config instances."""
    return {
        'debug': DebugConfig(),
        'wpentities': WpEntitiesConfig(),
        'article_only': ArticleOnlyConfig(),
    }


def get_config(config_name: str) -> BaseConfig:
    """
    Return config variant by name.

    Supported names:
    - ``debug``: minimal config loading from ``data/debug`` for fast local testing.
    - ``wpentities``: entity-enhanced default setup (gold-chrono-per-dataset).
    - ``article_only``: article embeddings only (entity embeddings disabled).

    :param config_name: Config variant name.
    :return: Selected config object.
    :raises ValueError: If ``config_name`` is unknown.
    """
    name = config_name.strip().lower()
    config_map = _config_map()
    if name not in config_map:
        raise ValueError(f'Unsupported config_name: {config_name}')
    return config_map[name]


def list_config_names() -> tuple[str, ...]:
    """
    Return names of supported config variants.

    :return: Tuple of supported config names.
    """
    return (
        'debug',
        'wpentities',
        'article_only',
    )


# Backward compatibility alias for older imports.
PipelineConfig = BaseConfig
