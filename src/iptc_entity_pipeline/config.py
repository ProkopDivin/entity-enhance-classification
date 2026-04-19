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
class PathsCnf:
    """Filesystem paths for data and artifacts."""

    train_csv: str = f'{DATA_ROOT}/gold-chrono-per-dataset/all-corpora-train-entities.csv'
    test_csv: str = f'{DATA_ROOT}/gold-chrono-per-dataset/all-corpora-test-entities.csv'
    wdid_mapping_tsv: str = f'{DATA_ROOT}/gold-chrono-per-dataset/wdId_mapping.tsv'
    article_embeddings_dir: str = f'{DATA_ROOT}/article_embeddings'
    entity_embeddings_dir: str = f'{DATA_ROOT}/entity_embeddings/WikidataProject'
    downsampling_order_cache_json: str = f'{DATA_ROOT}/downsampling_order_cache.json'
    removed_cat_ids: list[str] = field(default_factory=lambda: ['20000419'])


@dataclass(frozen=True)
class EmbeddingCnf:
    """Embedding loading and fallback-computation parameters."""

    article_model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2-300-0.3'
    article_embedding_dim: int = 384
    embed_svc_url: str = 'http://tau.g:5533'
    entity_lang: str = 'en'
    entity_langs: tuple[str, ...] = ()
    entity_relevance_threshold: float = 0.0
    use_entity_relevance_weights: bool = False
    use_entity_embeddings: bool = True
    combine_method: str = 'concat'
    entity_pooling: str = 'sum'


@dataclass(frozen=True)
class ModelCnf:
    """Scalar model architecture parameters for a single training run."""

    hidden_dim: int = 1024
    dropouts1: float = 0.0
    dropouts2: float = 0.3


@dataclass(frozen=True)
class TrainingCnf:
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
    early_stopping_min_delta: float = 0.000000001


@dataclass(frozen=True)
class HyperparamSpace:
    """Grid-search space for tunable hyperparameters.

    Each list field defines candidate values to try.  Use
    :meth:`iter_combinations` to expand the full Cartesian product.
    """

    hidden_dims: tuple[int, ...] = (1024,)
    dropouts1: tuple[float, ...] = (0.0,)
    dropouts2: tuple[float, ...] = (0.3,)
    batch_sizes: tuple[int, ...] = (100,)
    learning_rates: tuple[float, ...] = (0.00037,)

    def iter_combinations(
        self, base_training: TrainingCnf,
    ) -> list[tuple[ModelCnf, TrainingCnf]]:
        """Expand grid into all ``(ModelCnf, TrainingCnf)`` combinations.

        :param base_training: Base training config whose non-grid fields are preserved.
        :return: List of ``(ModelCnf, TrainingCnf)`` tuples.
        """
        return [
            (
                ModelCnf(hidden_dim=hd, dropouts1=d1, dropouts2=d2),
                replace(base_training, batch_size=bs, learning_rate=lr),
            )
            for hd, d1, d2, bs, lr in product(
                self.hidden_dims, self.dropouts1, self.dropouts2,
                self.batch_sizes, self.learning_rates,
            )
        ]


@dataclass(frozen=True)
class EvaluationCnf:
    """Evaluation behavior and threshold settings."""

    threshold_predict: float = -9999
    threshold_eval: float = 0.5
    per_corpus: bool = True
    per_class: bool = True
    averaging_type: str = 'datapoint'
    base_run_dir: str = ''


@dataclass(frozen=True)
class CvCnf:
    """Cross-validation setup."""

    folds: int = 5
    random_seed: int = 43


@dataclass(frozen=True)
class BaseCnf:
    """Top-level pipeline config grouped by concern."""

    paths: PathsCnf = field(default_factory=PathsCnf)
    emb: EmbeddingCnf = field(default_factory=EmbeddingCnf)
    model: ModelCnf = field(default_factory=ModelCnf)
    train: TrainingCnf = field(default_factory=TrainingCnf)
    eval: EvaluationCnf = field(default_factory=EvaluationCnf)
    cv: CvCnf = field(default_factory=CvCnf)
    hparam: HyperparamSpace = field(default_factory=HyperparamSpace)
    objective_corpora: str = 'All-datapoint'
    downsample_corpora: dict[str, float] = field(default_factory=dict)
    print_logs: bool = True
    upload_artifacts: bool = True
    debug: bool = True
    

    def to_clearml_mapping(self) -> dict[str, Any]:
        """Convert dataclasses to serializable mapping."""
        return asdict(self)

@dataclass(frozen=True)
class BaseCnfWithHPO(BaseCnf):
    """Base configuration with hyperparameter space."""
    debug: bool = field(default_factory=lambda: False)
    hparam: HyperparamSpace = field(
        default_factory=lambda: replace(
            HyperparamSpace(),
            hidden_dims=(384, 1024, 2048, 4096, 8192,),
            dropouts1=(0.0,),
            dropouts2=(0.0, 0.3, 0.5,),
            learning_rates=(0.00037,),
        )
    ) 

@dataclass(frozen=True)
class WpEntitiesCnf(BaseCnfWithHPO):
    """Default entity-enhanced configuration."""

@dataclass(frozen=True)
class WPEntitiesMeanCnf(BaseCnf):
    """Entity-enhanced configuration with mean pooling."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_pooling='mean')
    )


@dataclass(frozen=True)
class WPEntitiesWeightedMeanCnf(BaseCnfWithHPO):
    """Entity-enhanced config with relevance-weighted mean pooling enabled."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            use_entity_relevance_weights=True,
            entity_pooling='weighted_mean',
        )
    )


@dataclass(frozen=True)
class ArticleOnlyCnf(BaseCnfWithHPO):
    """Article-only configuration without entity embeddings."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), use_entity_embeddings=False)
    )



@dataclass(frozen=True)
class DebugCnf(BaseCnf):
    """Debug configuration for quick local runs."""

    paths: PathsCnf = field(
        default_factory=lambda: PathsCnf(
            train_csv=f'{DATA_ROOT}/debug/all-corpora-train-entities.csv',
            test_csv=f'{DATA_ROOT}/debug/all-corpora-test-entities.csv',
            wdid_mapping_tsv=f'{DATA_ROOT}/debug/wdId_mapping.tsv',
        )
    )
    model: ModelCnf = field(default_factory=lambda: replace(ModelCnf(), dropouts1=0.1))
    train: TrainingCnf = field(default_factory=lambda: replace(TrainingCnf(), epochs=5))
    hparam: HyperparamSpace = field(
        default_factory=lambda: replace(
            HyperparamSpace(),
            hidden_dims=(1024, ),
            dropouts1=(0.0,),
            dropouts2=(0.0, 0.3, 0.5, ),
            learning_rates=(0.00037,),
        )
    )
    cv: CvCnf = field(default_factory=lambda: replace(CvCnf(), folds=2))
    debug: bool = False



@dataclass(frozen=True)
class WPEntitiesEnDeCnf(BaseCnf):
    """Entity-enhanced configuration with English and German entity embeddings."""
    emb: EmbeddingCnf = field(default_factory=lambda: replace(EmbeddingCnf(), entity_langs=('en', 'de')))


@dataclass(frozen=True)
class WPEntitiesEnEsCnf(BaseCnf):
    """Entity-enhanced configuration with English and Spanish entity embeddings."""
    emb: EmbeddingCnf = field(default_factory=lambda: replace(EmbeddingCnf(), entity_langs=('en', 'es')))


@dataclass(frozen=True)
class WPEntitiesEnNlCnf(BaseCnfWithHPO):
    """Entity-enhanced configuration with English and Dutch entity embeddings."""
    emb: EmbeddingCnf = field(default_factory=lambda: replace(EmbeddingCnf(), entity_langs=('en', 'nl')))


@dataclass(frozen=True)
class WPEntitiesEnFrCnf(BaseCnf):
    """Entity-enhanced configuration with English and French entity embeddings."""
    emb: EmbeddingCnf = field(default_factory=lambda: replace(EmbeddingCnf(), entity_langs=('en', 'fr')))


@dataclass(frozen=True)
class WPEntitiesEnCsCnf(BaseCnf):
    """Entity-enhanced configuration with English and Czech entity embeddings."""
    emb: EmbeddingCnf = field(default_factory=lambda: replace(EmbeddingCnf(), entity_langs=('en', 'cs')))


@dataclass(frozen=True)
class WPEntitiesAllLangsCnf(BaseCnfWithHPO):
    """Entity-enhanced configuration with all supported entity embedding languages."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_langs=('en', 'de', 'es', 'nl', 'fr', 'cs'))
    )


@dataclass(frozen=True)
class WPEntitiesRelTH(BaseCnf):
    """Entity-enhanced configuration with English, German and Czech entity embeddings."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            entity_langs=('en',),
            entity_relevance_threshold=0.0,
        )
    )
    debug: bool = field(default_factory=lambda: True)
    hparam: HyperparamSpace = field(
        default_factory=lambda: replace(
            HyperparamSpace(),
            hidden_dims=(1024, ),
            dropouts1=(0.1,),
            dropouts2=(0.3,),
            learning_rates=(0.00037,),
        )
    )   
    
@dataclass(frozen=True)
class WPEntitiesRelTH1(BaseCnf):
    """Entity-enhanced configuration with English, German and Czech entity embeddings."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            entity_langs=('en',),
            entity_relevance_threshold=1.0,
        )
    )
    debug: bool = field(default_factory=lambda: True)
    hparam: HyperparamSpace = field(
        default_factory=lambda: replace(
            HyperparamSpace(),
            hidden_dims=(1024, ),
            dropouts1=(0.1,),
            dropouts2=(0.3,),
            learning_rates=(0.00037,),
        )
    )   
    
@dataclass(frozen=True)
class WPEntitiesRelTH3(BaseCnf):
    """Entity-enhanced configuration with English, German and Czech entity embeddings."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            entity_langs=('en',),
            entity_relevance_threshold=3.0,
        )
    )
    debug: bool = field(default_factory=lambda: True)
    hparam: HyperparamSpace = field(
        default_factory=lambda: replace(
            HyperparamSpace(),  
            hidden_dims=(1024, ),
            dropouts1=(0.1,),
            dropouts2=(0.3,),
            learning_rates=(0.00037,),
        )
    )   
    
@dataclass(frozen=True)
class WPEntitiesRelTH5(BaseCnfWithHPO):
    """Entity-enhanced configuration with English, German and Czech entity embeddings."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            entity_langs=('en',),
            entity_relevance_threshold=5.0,
        )
    )
    debug: bool = field(default_factory=lambda: True)
    hparam: HyperparamSpace = field(
        default_factory=lambda: replace(
            HyperparamSpace(),
            hidden_dims=(1024, ),
            dropouts1=(0.1,),
            dropouts2=(0.3,),
            learning_rates=(0.00037,),
        )
    )   
    
@dataclass(frozen=True)
class WPEntitiesRelTH7(BaseCnf):
    """Entity-enhanced configuration with English, German and Czech entity embeddings."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            entity_langs=('en',),
            entity_relevance_threshold=7.0,
        )
    )
    debug: bool = field(default_factory=lambda: True)
    hparam: HyperparamSpace = field(
        default_factory=lambda: replace(
            HyperparamSpace(),
            hidden_dims=(1024, ),
            dropouts1=(0.1,),
            dropouts2=(0.3,),
            learning_rates=(0.00037,),
        )
    )   
    
    
@dataclass(frozen=True)
class WPEntitiesRelTH9(BaseCnf):
    """Entity-enhanced configuration with English, German and Czech entity embeddings."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            entity_langs=('en',),
            entity_relevance_threshold=9.0,
        )
    )
    debug: bool = field(default_factory=lambda: True)
    hparam: HyperparamSpace = field(
        default_factory=lambda: replace(
            HyperparamSpace(),
            hidden_dims=(1024, ),
            dropouts1=(0.1,),
            dropouts2=(0.3,),
            learning_rates=(0.00037,),
        )
    )   
    
    
    
@dataclass(frozen=True)
class WPEntitiesRelTH11(BaseCnf):
    """Entity-enhanced configuration with English, German and Czech entity embeddings."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            entity_langs=('en',),
            entity_relevance_threshold=11.0,
        )
    )
    debug: bool = field(default_factory=lambda: True)
    hparam: HyperparamSpace = field(
        default_factory=lambda: replace(
            HyperparamSpace(),
            hidden_dims=(1024, ),
            dropouts1=(0.1,),
            dropouts2=(0.3,),
            learning_rates=(0.00037,),
        )
    )   
    
    
    
@dataclass(frozen=True)
class WPEntitiesRelTH13(BaseCnf):
    """Entity-enhanced configuration with English, German and Czech entity embeddings."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            entity_langs=('en',),
            entity_relevance_threshold=13.0,
        )
    )
    debug: bool = field(default_factory=lambda: True)
    hparam: HyperparamSpace = field(
        default_factory=lambda: replace(
            HyperparamSpace(),
            hidden_dims=(1024, ),
            dropouts1=(0.1,),
            dropouts2=(0.3,),
            learning_rates=(0.00037,),
        )
    )   
    

@dataclass(frozen=True)
class WPEntitiesNlCnf(BaseCnfWithHPO):
    """Entity-enhanced configuration with English and Dutch entity embeddings."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            entity_langs=('nl',),
        )
    )
    
@dataclass(frozen=True)
class WPEntitiesRelTH15(BaseCnf):
    """Entity-enhanced configuration with English, German and Czech entity embeddings."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            entity_langs=('en',),
            entity_relevance_threshold=15.0,
        )
    )
    debug: bool = field(default_factory=lambda: True)
    hparam: HyperparamSpace = field(
        default_factory=lambda: replace(
            HyperparamSpace(),
            hidden_dims=(1024, ),
            dropouts1=(0.1,),
            dropouts2=(0.3,),
            learning_rates=(0.00037,),
        )
    )   
    
    
@dataclass(frozen=True)
class WPEntitiesRelTH17(BaseCnf):
    """Entity-enhanced configuration with English, German and Czech entity embeddings."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            entity_langs=('en',),
            entity_relevance_threshold=17.0,
        )
    )
    debug: bool = field(default_factory=lambda: True)
    hparam: HyperparamSpace = field(
        default_factory=lambda: replace(
            HyperparamSpace(),
            hidden_dims=(1024, ),
            dropouts1=(0.1,),
            dropouts2=(0.3,),
            learning_rates=(0.00037,),
        )
    )   
    
@dataclass(frozen=True)
class WPEntitiesRelTH999(BaseCnf):
    """Entity-enhanced configuration with English, German and Czech entity embeddings."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            entity_langs=('en',),
            entity_relevance_threshold=999.0,
        )
    )
    debug: bool = field(default_factory=lambda: True)
    hparam: HyperparamSpace = field(
        default_factory=lambda: replace(
            HyperparamSpace(),
            hidden_dims=(1024, ),
            dropouts1=(0.1,),
            dropouts2=(0.3,),
            learning_rates=(0.00037,),
        )
    ) 
    

@dataclass(frozen=True)
class WPEntitiesMentionWeightedSumCnf(BaseCnfWithHPO):
    """Entity-enhanced configuration with mention-weighted sum pooling enabled."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            entity_pooling='weighted_sum',
        )
    )


@dataclass(frozen=True)
class WPEntitiesRelevanceWeightedSumCnf(BaseCnfWithHPO):
    """Entity-enhanced configuration with relevance-weighted sum pooling enabled."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            entity_pooling='weighted_sum_relevance',
        )
    )


def resolve_paths(config: BaseCnf, root_dir: str | Path) -> BaseCnf:
    """Return a config with absolute paths resolved from ``root_dir``."""
    root_path = Path(root_dir)
    paths = config.paths
    resolved_paths = PathsCnf(
        train_csv=str(root_path / paths.train_csv),
        test_csv=str(root_path / paths.test_csv),
        wdid_mapping_tsv=str(root_path / paths.wdid_mapping_tsv),
        article_embeddings_dir=str(root_path / paths.article_embeddings_dir),
        entity_embeddings_dir=str(root_path / paths.entity_embeddings_dir),
        downsampling_order_cache_json=str(root_path / paths.downsampling_order_cache_json),
        removed_cat_ids=paths.removed_cat_ids,
    )
    return replace(config, paths=resolved_paths)


@dataclass(frozen=True)
class BestWpEntitiesCnf(BaseCnfWithHPO):
    hparam: HyperparamSpace = field(
        default_factory=lambda: replace(
            HyperparamSpace(),
            hidden_dims=(1024, ),
            dropouts1=(0.0,),
            dropouts2=(0.0, ),
            learning_rates=(0.00037,),
        )
    )
    
@dataclass(frozen=True)
class BestArticleOnlyCnf(BaseCnfWithHPO):
    hparam: HyperparamSpace = field(
        default_factory=lambda: replace(
            HyperparamSpace(),
            hidden_dims=(2048, ),
            dropouts1=(0.0,),
            dropouts2=(0.3, ),
            learning_rates=(0.00037,),
        )
    )
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), use_entity_embeddings=False)
    )
########################################################
# Best configurations for different relevance thresholds
########################################################


@dataclass(frozen=True)
class BestWpentitiesAllLangsCnf(BaseCnfWithHPO):
    hparam: HyperparamSpace = field(
        default_factory=lambda: replace(
            HyperparamSpace(),
            hidden_dims=(384, 1024),
            dropouts1=(0.0,),
            dropouts2=(0.3, ),
            learning_rates=(0.00037,),
        )
    )
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_langs=('en', 'de', 'es', 'nl', 'fr', 'cs'))
    )


@dataclass(frozen=True)
class BestWpentitiesNlCnf(BaseCnfWithHPO):
    hparam: HyperparamSpace = field(
        default_factory=lambda: replace(
            HyperparamSpace(),
            hidden_dims=(384, 1024),
            dropouts1=(0.0,),
            dropouts2=(0.3, ),
            learning_rates=(0.00037,),
        )
    )
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_langs=('nl',))
    )

@dataclass(frozen=True)
class BestWPEntitiesENNLCnf(BaseCnfWithHPO):
    hparam: HyperparamSpace = field(
        default_factory=lambda: replace(
            HyperparamSpace(),
            hidden_dims=(1024, 8192),
            dropouts1=(0.0,),
            dropouts2=(0.3, ),
            learning_rates=(0.00037,),
        )
    )
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_langs=('en', 'nl'))
    )

def _config_map() -> dict[str, BaseCnf]:
    """Return supported config instances."""
    return {
        'debug': DebugCnf(),
        'article_only': ArticleOnlyCnf(),
        'wpentities': WpEntitiesCnf(),
        'wpentities_weighted_mean': WPEntitiesWeightedMeanCnf(),
        'wpentities_relevance_weighted_sum': WPEntitiesRelevanceWeightedSumCnf(),
        'wpentities_mention_weighted_sum': WPEntitiesMentionWeightedSumCnf(),
        'wpentities_en_nl': WPEntitiesEnNlCnf(),
        'wpentities_nl': WPEntitiesNlCnf(),
        'wpentities_all_langs': WPEntitiesAllLangsCnf(),
        'wpentities_rel_th_5': WPEntitiesRelTH5(),
        'best_wpentities': BestWpEntitiesCnf(),
        'best_article_only': BestArticleOnlyCnf(),
        'best_wpentities_all_langs': BestWpentitiesAllLangsCnf(),
        'best_wpentities_nl': BestWpentitiesNlCnf(),
        'best_wpentities_en_nl': BestWPEntitiesENNLCnf(),

    }



def get_config(config_name: str) -> BaseCnf:
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
PipelineCnf = BaseCnf
