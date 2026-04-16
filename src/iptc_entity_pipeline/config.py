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
    entity_langs: tuple[str, ...] = ()
    entity_relevance_threshold: float = 0.0
    use_entity_embeddings: bool = True
    combine_method: str = 'concat'
    entity_pooling: str = 'weighted_mean'


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
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.00001


@dataclass(frozen=True)
class HyperparamSpace:
    """Grid-search space for tunable hyperparameters.

    Each list field defines candidate values to try.  Use
    :meth:`iter_combinations` to expand the full Cartesian product.
    """

    hidden_dims: tuple[int, ...] = (100, 384, 1024, 2048, 4096, 8192)
    dropouts1: tuple[float, ...] = (0.0,)
    dropouts2: tuple[float, ...] = (0.0, 0.1, 0.3, 0.5)
    batch_sizes: tuple[int, ...] = (100,)
    learning_rates: tuple[float, ...] = (0.00037,)

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
    debug: bool = False
    

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
        default_factory=lambda: replace(EmbeddingConfig(), use_entity_embeddings=False)
    )



@dataclass(frozen=True)
class DebugConfig(BaseConfig):
    """Debug configuration for quick local runs."""

    paths: PathsConfig = field(
        default_factory=lambda: PathsConfig(
            train_csv=f'{DATA_ROOT}/debug/all-corpora-train-entities.csv',
            test_csv=f'{DATA_ROOT}/debug/all-corpora-test-entities.csv',
            wdid_mapping_tsv=f'{DATA_ROOT}/debug/wdId_mapping.tsv',
        )
    )
    model: ModelConfig = field(default_factory=lambda: replace(ModelConfig(), dropouts1=0.1))
    training: TrainingConfig = field(default_factory=lambda: replace(TrainingConfig(), epochs=5))
    hyperparam_space: HyperparamSpace = field(
        default_factory=lambda: replace(
            HyperparamSpace(),
            hidden_dims=(1024, ),
            dropouts1=(0.1,),
            dropouts2=(0.3,),
            learning_rates=(0.00037,),
        )
    )
    cv: CvConfig = field(default_factory=lambda: replace(CvConfig(), folds=2))
    debug: bool = False



@dataclass(frozen=True)
class WPEntitiesEnDeConfig(BaseConfig):
    """Entity-enhanced configuration with English and German entity embeddings."""
    embeddings: EmbeddingConfig = field(default_factory=lambda: replace(EmbeddingConfig(), entity_langs=('en', 'de')))


@dataclass(frozen=True)
class WPEntitiesEnEsConfig(BaseConfig):
    """Entity-enhanced configuration with English and Spanish entity embeddings."""
    embeddings: EmbeddingConfig = field(default_factory=lambda: replace(EmbeddingConfig(), entity_langs=('en', 'es')))


@dataclass(frozen=True)
class WPEntitiesEnNlConfig(BaseConfig):
    """Entity-enhanced configuration with English and Dutch entity embeddings."""
    embeddings: EmbeddingConfig = field(default_factory=lambda: replace(EmbeddingConfig(), entity_langs=('en', 'nl')))


@dataclass(frozen=True)
class WPEntitiesEnFrConfig(BaseConfig):
    """Entity-enhanced configuration with English and French entity embeddings."""
    embeddings: EmbeddingConfig = field(default_factory=lambda: replace(EmbeddingConfig(), entity_langs=('en', 'fr')))


@dataclass(frozen=True)
class WPEntitiesEnCsConfig(BaseConfig):
    """Entity-enhanced configuration with English and Czech entity embeddings."""
    embeddings: EmbeddingConfig = field(default_factory=lambda: replace(EmbeddingConfig(), entity_langs=('en', 'cs')))


@dataclass(frozen=True)
class WPEntitiesAllLangsConfig(BaseConfig):
    """Entity-enhanced configuration with all supported entity embedding languages."""
    embeddings: EmbeddingConfig = field(
        default_factory=lambda: replace(EmbeddingConfig(), entity_langs=('en', 'de', 'es', 'nl', 'fr', 'cs'))
    )


@dataclass(frozen=True)
class WPEntitiesRelTH(BaseConfig):
    """Entity-enhanced configuration with English, German and Czech entity embeddings."""
    embeddings: EmbeddingConfig = field(default_factory=lambda: replace(EmbeddingConfig(), entity_langs=('en', 'de', 'cs')))
    relevance_threshold: float = field(default_factory=lambda: 0.0)
    debug: bool = field(default_factory=lambda: True)
    hyperparam_space: HyperparamSpace = field(
        default_factory=lambda: replace(
            HyperparamSpace(),
            hidden_dims=(1024, ),
            dropouts1=(0.1,),
            dropouts2=(0.3,),
            learning_rates=(0.00037,),
        )
    )   
    
@dataclass(frozen=True)
class WPEntitiesRelTH5(BaseConfig):
    """Entity-enhanced configuration with English, German and Czech entity embeddings."""
    embeddings: EmbeddingConfig = field(default_factory=lambda: replace(EmbeddingConfig(), entity_langs=('en', 'de', 'cs')))
    relevance_threshold: float = field(default_factory=lambda: 5.0)
    debug: bool = field(default_factory=lambda: True)
    hyperparam_space: HyperparamSpace = field(
        default_factory=lambda: replace(
            HyperparamSpace(),
            hidden_dims=(1024, ),
            dropouts1=(0.1,),
            dropouts2=(0.3,),
            learning_rates=(0.00037,),
        )
    )   
    
@dataclass(frozen=True)
class WPEntitiesRelTH10(BaseConfig):
    """Entity-enhanced configuration with English, German and Czech entity embeddings."""
    embeddings: EmbeddingConfig = field(default_factory=lambda: replace(EmbeddingConfig(), entity_langs=('en', 'de', 'cs')))
    relevance_threshold: float = field(default_factory=lambda: 10.0)
    debug: bool = field(default_factory=lambda: True)
    hyperparam_space: HyperparamSpace = field(
        default_factory=lambda: replace(
            HyperparamSpace(),
            hidden_dims=(1024, ),
            dropouts1=(0.1,),
            dropouts2=(0.3,),
            learning_rates=(0.00037,),
        )
    )   
    
@dataclass(frozen=True)
class WPEntitiesRelTH20(BaseConfig):
    """Entity-enhanced configuration with English, German and Czech entity embeddings."""
    embeddings: EmbeddingConfig = field(default_factory=lambda: replace(EmbeddingConfig(), entity_langs=('en', 'de', 'cs')))
    relevance_threshold: float = field(default_factory=lambda: 20.0)
    debug: bool = field(default_factory=lambda: True)
    hyperparam_space: HyperparamSpace = field(
        default_factory=lambda: replace(
            HyperparamSpace(),
            hidden_dims=(1024, ),
            dropouts1=(0.1,),
            dropouts2=(0.3,),
            learning_rates=(0.00037,),
        )
    )   
    
@dataclass(frozen=True)
class WPEntitiesRelTH30(BaseConfig):
    """Entity-enhanced configuration with English, German and Czech entity embeddings."""
    embeddings: EmbeddingConfig = field(default_factory=lambda: replace(EmbeddingConfig(), entity_langs=('en', 'de', 'cs')))
    relevance_threshold: float = field(default_factory=lambda: 30.0)
    debug: bool = field(default_factory=lambda: True)
    hyperparam_space: HyperparamSpace = field(
        default_factory=lambda: replace(
            HyperparamSpace(),
            hidden_dims=(1024, ),
            dropouts1=(0.1,),
            dropouts2=(0.3,),
            learning_rates=(0.00037,),
        )
    )   
    
    
@dataclass(frozen=True)
class WPEntitiesRelTH40(BaseConfig):
    """Entity-enhanced configuration with English, German and Czech entity embeddings."""
    embeddings: EmbeddingConfig = field(default_factory=lambda: replace(EmbeddingConfig(), entity_langs=('en', 'de', 'cs')))
    relevance_threshold: float = field(default_factory=lambda: 40.0)
    debug: bool = field(default_factory=lambda: True)
    hyperparam_space: HyperparamSpace = field(
        default_factory=lambda: replace(
            HyperparamSpace(),
            hidden_dims=(1024, ),
            dropouts1=(0.1,),
            dropouts2=(0.3,),
            learning_rates=(0.00037,),
        )
    )   
    
    
    
@dataclass(frozen=True)
class WPEntitiesRelTH50(BaseConfig):
    """Entity-enhanced configuration with English, German and Czech entity embeddings."""
    embeddings: EmbeddingConfig = field(default_factory=lambda: replace(EmbeddingConfig(), entity_langs=('en', 'de', 'cs')))
    relevance_threshold: float = field(default_factory=lambda: 50.0)
    debug: bool = field(default_factory=lambda: True)
    hyperparam_space: HyperparamSpace = field(
        default_factory=lambda: replace(
            HyperparamSpace(),
            hidden_dims=(1024, ),
            dropouts1=(0.1,),
            dropouts2=(0.3,),
            learning_rates=(0.00037,),
        )
    )   
    
    
    
@dataclass(frozen=True)
class WPEntitiesRelTH60(BaseConfig):
    """Entity-enhanced configuration with English, German and Czech entity embeddings."""
    embeddings: EmbeddingConfig = field(default_factory=lambda: replace(EmbeddingConfig(), entity_langs=('en', 'de', 'cs')))
    relevance_threshold: float = field(default_factory=lambda: 60.0)
    debug: bool = field(default_factory=lambda: True)
    hyperparam_space: HyperparamSpace = field(
        default_factory=lambda: replace(
            HyperparamSpace(),
            hidden_dims=(1024, ),
            dropouts1=(0.1,),
            dropouts2=(0.3,),
            learning_rates=(0.00037,),
        )
    )   
    
    
    
@dataclass(frozen=True)
class WPEntitiesRelTH70(BaseConfig):
    """Entity-enhanced configuration with English, German and Czech entity embeddings."""
    embeddings: EmbeddingConfig = field(default_factory=lambda: replace(EmbeddingConfig(), entity_langs=('en', 'de', 'cs')))
    relevance_threshold: float = field(default_factory=lambda: 70.0)
    debug: bool = field(default_factory=lambda: True)
    hyperparam_space: HyperparamSpace = field(
        default_factory=lambda: replace(
            HyperparamSpace(),
            hidden_dims=(1024, ),
            dropouts1=(0.1,),
            dropouts2=(0.3,),
            learning_rates=(0.00037,),
        )
    )   
    
    
@dataclass(frozen=True)
class WPEntitiesRelTH80(BaseConfig):
    """Entity-enhanced configuration with English, German and Czech entity embeddings."""
    embeddings: EmbeddingConfig = field(default_factory=lambda: replace(EmbeddingConfig(), entity_langs=('en', 'de', 'cs')))
    relevance_threshold: float = field(default_factory=lambda: 80.0)
    debug: bool = field(default_factory=lambda: True)
    hyperparam_space: HyperparamSpace = field(
        default_factory=lambda: replace(
            HyperparamSpace(),
            hidden_dims=(1024, ),
            dropouts1=(0.1,),
            dropouts2=(0.3,),
            learning_rates=(0.00037,),
        )
    )   
    
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
        'wpentities_en_de': WPEntitiesEnDeConfig(),
        'wpentities_en_es': WPEntitiesEnEsConfig(),
        'wpentities_en_nl': WPEntitiesEnNlConfig(),
        'wpentities_en_fr': WPEntitiesEnFrConfig(),
        'wpentities_en_cs': WPEntitiesEnCsConfig(),
        'wpentities_all_langs': WPEntitiesAllLangsConfig(),
        'wpentities_rel_th': WPEntitiesRelTH(),
        'wpentities_rel_th_5': WPEntitiesRelTH5(),
        'wpentities_rel_th_10': WPEntitiesRelTH10(),
        'wpentities_rel_th_20': WPEntitiesRelTH20(),
        'wpentities_rel_th_30': WPEntitiesRelTH30(),
        'wpentities_rel_th_40': WPEntitiesRelTH40(),
        'wpentities_rel_th_50': WPEntitiesRelTH50(),
        'wpentities_rel_th_60': WPEntitiesRelTH60(),
        'wpentities_rel_th_70': WPEntitiesRelTH70(),
        'wpentities_rel_th_80': WPEntitiesRelTH80(),
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
    return _config_map().keys()


# Backward compatibility alias for older imports.
PipelineConfig = BaseConfig
