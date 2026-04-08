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
    """Model architecture parameters."""

    hidden_dim: list[int] = field(default_factory=lambda: [1024])
    dropouts1: list[float] = field(default_factory=lambda: [0.0])
    dropouts2: list[float] = field(default_factory=lambda: [0.3])


@dataclass(frozen=True)
class TrainingConfig:
    """Training loop parameters."""

    epochs: int = 100
    batch_size: list[int] = field(default_factory=lambda: [64])
    optimizer_name: str = 'adam'
    learning_rate: list[float] = field(default_factory=lambda: [0.00037])
    lr_scheduler_name: str = 'stepLR'
    step_size: int = 1
    gamma: float = 1
    loss_name: str = 'bceWithLogitsLoss'
    # 0 = disabled. When > 0, monitors dev loss, stops after this many epochs
    # without improvement, and restores the best weights.
    early_stopping_patience: int = 6
    early_stopping_min_delta: float = 0.000001


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation behavior and threshold settings."""

    threshold_predict: float = -0.3
    threshold_eval: float = 0.5
    per_corpus: bool = True
    per_class: bool = True
    averaging_type: str = 'datapoint'


@dataclass(frozen=True)
class CvConfig:
    """Cross-validation setup."""

    folds: int = 5
    random_seed: int = 43


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
    cv: CvConfig = field(default_factory=CvConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    objective_corpora: str = 'All-datapoint'
    downsample_corpora: dict[str, float] = field(default_factory=dict)

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
        downsampling_order_cache_json=str(root_path / paths.downsampling_order_cache_json),
        removed_cat_ids=paths.removed_cat_ids,
    )
    return BaseConfig(
        paths=resolved_paths,
        embeddings=config.embeddings,
        model=config.model,
        training=config.training,
        evaluation=config.evaluation,
        cv=config.cv,
        logging=config.logging,
        objective_corpora=config.objective_corpora,
        downsample_corpora=config.downsample_corpora,
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




def _with_corpora_dir(config: BaseConfig, corpora_dir_name: str) -> BaseConfig:
    """
    Return a config variant with train/dev/test CSVs from selected corpora directory.

    :param config: Base config to adapt.
    :param corpora_dir_name: Subdirectory under ``DATA_ROOT`` containing corpora CSV files.
    :return: Config with updated corpora CSV paths.
    """
    return replace(
        config,
        paths=replace(
            config.paths,
            train_csv=f'{DATA_ROOT}/{corpora_dir_name}/all-corpora-train.csv',
            dev_csv=f'{DATA_ROOT}/{corpora_dir_name}/all-corpora-dev.csv',
            test_csv=f'{DATA_ROOT}/{corpora_dir_name}/all-corpora-test.csv',
        ),
    )


def to_article_config_filtred_dataset(config: BaseConfig, corpora_dir: str) -> BaseConfig:
    config = replace(
        config,
        embeddings=replace(
            config.embeddings,
            use_entity_embeddings=False,
        )
    )
    return _with_corpora_dir(config=config, corpora_dir_name=corpora_dir)


def to_downsample_nl_noordhollandsdagblad_033(config: BaseConfig) -> BaseConfig:
    """
    Return config variant with train/dev downsampling for ``nl_noordhollandsdagblad``.

    :param config: Base config to adapt.
    :return: Config with deterministic per-corpus downsampling enabled.
    """
    return replace(
        config,
        downsample_corpora={'nl_noordhollandsdagblad': 0.33},
    )


def get_config(config_name: str) -> BaseConfig:
    """
    Return config variant by name.

    Supported names:
    - ``base``: entity-enhanced default setup.
    - ``article_only``: article embeddings only (entity embeddings disabled).
    - ``entities_origin_filtred``: entity-enhanced with ``origin-corpora-filtred`` inputs.
    - ``entities_chrono_global``: entity-enhanced with ``chrono-corpora-global`` inputs.
    - ``entities_chrono_per_dataset``: entity-enhanced with ``chrono-corpora-per-dataset`` inputs.
    - ``downsample_nl_noordhollandsdagblad_033``: downsample ``nl_noordhollandsdagblad`` train/dev to 33%.

    :param config_name: Config variant name.
    :return: Selected config object.
    :raises ValueError: If ``config_name`` is unknown.
    """
    name = config_name.strip().lower()
    if name == 'base':
        return BaseConfig()
    if name == 'article_only':
        return to_article_only_config(config=BaseConfig())
    if name == 'entities_origin_filtred':
        return _with_corpora_dir(config=BaseConfig(), corpora_dir_name='origin-corpora-filtred')
    if name == 'entities_chrono_global':
        return _with_corpora_dir(config=BaseConfig(), corpora_dir_name='chrono-corpora-global')
    if name == 'entities_chrono_per_dataset':
        return _with_corpora_dir(config=BaseConfig(), corpora_dir_name='chrono-corpora-per-dataset')
    if name == 'article_only_filtred':
        return to_article_config_filtred_dataset(config=BaseConfig(), corpora_dir='origin-corpora-filtred')
    if name == 'downsample_nl_noordhollandsdagblad_033':
        return to_downsample_nl_noordhollandsdagblad_033(config=BaseConfig())
    raise ValueError(f'Unsupported config_name: {config_name}')


def list_config_names() -> tuple[str, ...]:
    """
    Return names of supported config variants.

    :return: Tuple of supported config names.
    """
    return (
        'base',
        'article_only',
        'entities_origin_filtred',
        'entities_chrono_global',
        'entities_chrono_per_dataset',
        'article_only_filtred',
        'downsample_nl_noordhollandsdagblad_033',
    )


# Backward compatibility alias for older imports.
PipelineConfig = BaseConfig

