"""Configuration dataclasses for the IPTC entity-enhanced pipeline."""

from dataclasses import Field, asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal, Mapping

DATA_ROOT = '/home/prokop/Git/entity-enhance-classification/data'
ALL_ENTITY_LANGS = ('en', 'de', 'es', 'nl', 'fr', 'cs')


def conf_from_dict(cls, d: Mapping[str, Any]):
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
    use_article_embeddings: bool = True
    use_entity_embeddings: bool = True
    combine_method: str = 'concat'
    entity_pooling: Literal[
        'sum',
        'mean',
        'weighted_mean',
        'weighted_sum',
        'weighted_sum_relevance',
        'no_pooling',
    ] = 'sum'


@dataclass(frozen=True)
class ModelCnf:
    """Scalar model architecture parameters for a single training run."""

    hidden_dim: int = 1024
    dropouts1: float = 0.0
    dropouts2: float = 0.3
    nn_type: str = 'mlp'
    entity_dim: int = 0
    attention_hidden_dim: int = 128
    attention_dropout: float = 0.0
    bias_from_prior: bool = False


@dataclass(frozen=True)
class TrainingCnf:
    """Scalar training loop parameters for a single training run."""
    
    learning_rate: float = 0.00037
    epochs: int = 100
    batch_size: int = 64
    optimizer_name: str = 'adam'
   
    lr_scheduler_name: str = 'stepLR'
    step_size: int = 1
    gamma: float = 1 # for now 
    loss_name: str = 'bceWithLogitsLoss'
    # 0 = disabled. When > 0, monitors dev loss, stops after this many epochs
    # without improvement, and restores the best weights.
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.000000001 # because of small classes the improvement can be very small
    # because the validation set is different than test set - not splited chronologically 
    # we do not want to overfit it - not expecting all the articles be the same
    early_stopping_metric: Literal['loss', 'f1'] = 'loss'
    # If False, skip the extra forward pass over the full training set each epoch (dev
    # validation unchanged). Saves wall time and avoids holding large per-batch logits
    # during that pass; per-epoch train curves in ClearML stay empty.
    train_validation: bool = True


@dataclass(frozen=True)
class HyperparamSpace:
    """Grid-search space for tunable hyperparameters.

    Each field defines candidate values for the Optuna sampler.
    """

    hidden_dims: tuple[int, ...] = (1024,)
    dropouts1: tuple[float, ...] = (0.0,)
    dropouts2: tuple[float, ...] = (0.3,)
    attention_hidden_dims: tuple[int, ...] = (128,)
    attention_dropouts: tuple[float, ...] = (0.0,)
    batch_sizes: tuple[int, ...] = (100,)
    learning_rates: tuple[float, ...] = (0.00037,)


@dataclass(frozen=True)
class EvaluationCnf:
    """Evaluation behavior and threshold settings."""

    threshold_predict: float = -9999
    threshold_eval: float = 0.5
    per_corpus: bool = True
    per_class: bool = True
    averaging_type: str = 'micro' # strategy for treshold tuning and metric to use for corpora level evaluation
    base_run_dir: str = ''


@dataclass(frozen=True)
class CvCnf:
    """Cross-validation setup. The pipeline-wide ``BaseCnf.random_seed``
    drives the fold splitter; this block intentionally has no seed field."""

    folds: int = 5


@dataclass(frozen=True)
class OptunaCnf:
    """Optuna optimization behavior for CV hyperparameter search.

    The pipeline-wide ``BaseCnf.random_seed`` drives the sampler seed;
    this block intentionally has no seed field.
    """

    sampler: Literal['grid', 'tpe', 'random'] = 'grid'
    direction: str = 'maximize'
    n_trials: int = 0
    pruner: str = 'none'
    startup_trials: int = 5
    warmup_steps: int = 2


@dataclass(frozen=True)
class ThresholdTuningCnf:
    """Per-class decision-threshold tuning on dev folds.

    When ``enabled``, after each CV fold of the best Optuna trial the dev-set
    raw scores are scanned over the ``thresholds`` grid and the threshold that
    maximizes F-beta is selected per class. Per-fold per-class thresholds are
    aggregated across folds (``mean`` by default; ``median`` and ``mode`` also
    supported) and the resulting map is reused
    by the final-model evaluation as ``customThresholds``.
    """

    enabled: bool = False
    thresholds: tuple[float, ...] = field(
        # používám jen pro toto
        default_factory=lambda: tuple(round(0.05 * i, 2) for i in range(5, 14))
    )
    f_beta: float = 1.0
    aggregation: Literal['mean', 'median', 'mode'] = 'mean' # remove other not usefull 
    min_folds_for_tuning: int = 3
    # Controls which CV metric selects hyperparameters in Optuna.
    # `F1_micro` uses the objective corpora row F1 (typically `All_micro`).
    # `F1_macro_relevant` uses macro F1 averaged over relevant classes.
    selection_metric: Literal['F1_micro', 'F1_macro_relevant'] = 'F1_micro'


@dataclass(frozen=True)
class BaseCnf:
    """Top-level pipeline config grouped by concern."""

    paths: PathsCnf = field(default_factory=PathsCnf)
    emb: EmbeddingCnf = field(default_factory=EmbeddingCnf)
    model: ModelCnf = field(default_factory=ModelCnf)
    train: TrainingCnf = field(default_factory=TrainingCnf)
    eval: EvaluationCnf = field(default_factory=EvaluationCnf)
    cv: CvCnf = field(default_factory=CvCnf)
    optuna: OptunaCnf = field(default_factory=OptunaCnf)
    hparam: HyperparamSpace = field(default_factory=HyperparamSpace)
    tuning: ThresholdTuningCnf = field(default_factory=ThresholdTuningCnf)
    objective_row: str = 'All_micro'
    downsample_corpora: dict[str, float] = field(default_factory=dict)
    # Single random seed that drives every randomness source in the pipeline:
    # global RNGs (python random / numpy / torch CPU+CUDA, cudnn deterministic),
    # the CV fold splitter, the Optuna sampler, and per-fold model init /
    # DataLoader shuffling (re-seeded with a fold-derived offset).
    random_seed: int = 43
    print_logs: bool = True
    upload_artifacts: bool = False
    debug: bool = True


    def to_clearml_mapping(self) -> dict[str, Any]:
        """Convert dataclasses to serializable mapping."""
        return asdict(self)


@dataclass(frozen=True)
class AssemblyMemberCnf:
    """One member of an assembly ensemble.

    :param config: Full pipeline config instance for this member. Its
        ``paths``, ``emb``, ``model``, ``train``, ``hparam``, ``cv``, and
        ``optuna`` blocks drive that member's data prep and per-member
        ``run_cv``. The member's ``tuning`` block is honored only outside
        assembly mode; in assembly mode tuning is force-disabled because
        ``thresholds_path`` is the source of truth.
    :param thresholds_path: Per-class threshold JSON ``{cat_id: float}``.
        Used both as the assembly's per-fold ``eval_thresholds`` (so each
        member's CV per-class F1 is measured at production thresholds)
        and as the source for the final stitched per-class thresholds.
        Missing classes fall back to ``EvaluationCnf.threshold_eval``.
    :param label: Short identifier used in tables and artifact filenames.
    """

    config: BaseCnf = field(default_factory=BaseCnf)
    thresholds_path: str = ''
    label: str = ''


@dataclass(frozen=True)
class AssemblyCnf:
    """Dual-model ensemble configuration.

    Lives only on configs that opt in to assembly (e.g. by adding an
    ``assembly`` field to a ``BaseCnf`` subclass). In assembly mode each
    member is trained through the regular ``run_cv`` step (with that
    member's loaded thresholds applied during per-fold evaluation), then
    a per-class winner is picked from each member's CV per-class F1.

    :param members: Tuple of two member specs. Index 0 is the primary
        member; ties on average F1 resolve to the primary.
    :param mapping_artifact_name: ClearML artifact name for the
        ``class_to_model`` mapping JSON.
    :param sign_test: When True, per-class selection switches from "pick
        member with highest mean CV F1" to a sign-test rule: the primary
        member is kept unless the non-primary member strictly beats it in
        at least ``folds - 1`` of the CV folds (e.g. 4 out of 5). Ties in
        a fold count as a primary win.
    """

    enabled: bool = True
    members: tuple[AssemblyMemberCnf, ...] = ()
    mapping_artifact_name: str = 'assembly_class_to_model'
    sign_test: bool = False


@dataclass(frozen=True)
class PreBaseCnfWithHPO(BaseCnf):
    """Base configuration with hyperparameter space."""
    debug: bool = field(default_factory=lambda: False)
    hparam: HyperparamSpace = field(
        default_factory=lambda: replace(
            HyperparamSpace(),
            hidden_dims=(1024, 2048, 4096, 8192,),
            dropouts1=(0.0, 0.1,),
            dropouts2=(0.1, 0.3, 0.5,),
            learning_rates=(0.00037,),
        )
    ) 
    optuna: OptunaCnf = field(
        default_factory=lambda: replace(OptunaCnf(), sampler='grid')
    )
    
    
@dataclass(frozen=True)
class BaseCnfWithHPO(PreBaseCnfWithHPO):
    """Base configuration with hyperparameter space."""
    tuning: ThresholdTuningCnf = field(
        default_factory=lambda: replace(ThresholdTuningCnf(), enabled=True)
    )
    
    train: TrainingCnf = field(
        default_factory=lambda: replace(TrainingCnf(), train_validation=True)
    )


@dataclass(frozen=True)
class BaseCnfWithHPO2(BaseCnf):
    """Base configuration with hyperparameter space."""
    debug: bool = field(default_factory=lambda: False)
    hparam: HyperparamSpace = field(
        default_factory=lambda: replace(
            HyperparamSpace(),
            hidden_dims=(100, 384, 1024, 2048, 4096, 8192, 16384,),
            dropouts1=(0.0, 0.1,),
            dropouts2=(0.0, 0.15, 0.3, 0.5,),
            learning_rates=(0.00037,),
        )
    ) 
    

@dataclass(frozen=True)
class BaseCnfWithHPO3(BaseCnf):
    """Base configuration with hyperparameter space."""
    debug: bool = field(default_factory=lambda: False)
    hparam: HyperparamSpace = field(
        default_factory=lambda: replace(
            HyperparamSpace(),
            hidden_dims=(1024, ),
            dropouts1=(0.0,),
            dropouts2=(0.3,),
            learning_rates=(0.000005, 0.00001, 0.00005, 0.0001,0.00037, 0.001),
        )
    ) 
    optuna: OptunaCnf = field(
        default_factory=lambda: replace(OptunaCnf(), sampler='grid')
    )

@dataclass(frozen=True)
class WpEntitiesCnf(BaseCnfWithHPO2):
    """Default entity-enhanced configuration."""
    

@dataclass(frozen=True)
class WpEntitiesCnf2(BaseCnfWithHPO2):
    """Default entity-enhanced configuration."""
    random_seed: int = 294613

@dataclass(frozen=True)
class WpEntitiesCnf3(BaseCnfWithHPO2):
    """Default entity-enhanced configuration."""
    random_seed: int = 999751

@dataclass(frozen=True)
class WpEntitiesCnf4(BaseCnfWithHPO2):
    """Default entity-enhanced configuration."""
    random_seed: int = 212654

@dataclass(frozen=True)
class WpEntitiesCnf5(BaseCnfWithHPO2):
    """Default entity-enhanced configuration."""
    random_seed: int = 984621
    
    
@dataclass(frozen=True)
class WPEntitiesMeanCnf(BaseCnf):
    """Entity-enhanced configuration with mean pooling."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_pooling='mean')
    )


@dataclass(frozen=True)
class WPEntitiesWeightedMeanCnf(PreBaseCnfWithHPO):
    """Entity-enhanced config with relevance-weighted mean pooling enabled."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            use_entity_relevance_weights=True,
            entity_pooling='weighted_mean',
        )
    )
    tuning: ThresholdTuningCnf = field(
        default_factory=lambda: replace(ThresholdTuningCnf(), enabled=True)
    )


@dataclass(frozen=True)
class ArticleOnlyCnf(PreBaseCnfWithHPO):
    """Article-only configuration without entity embeddings."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), use_entity_embeddings=False)
    )
    


@dataclass(frozen=True)
class EntityOnlyCnf(BaseCnfWithHPO):
    """Entity-only configuration without article embeddings."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), use_article_embeddings=False)
    )
    
    
@dataclass(frozen=True)
class W2VEntityOnlyCnf(BaseCnfWithHPO):
    """Entity-only configuration without article embeddings."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), use_article_embeddings=False)
    )
    paths: PathsCnf = field(
        default_factory=lambda: replace(
            PathsCnf(),
            entity_embeddings_dir=f'{DATA_ROOT}/entity_embeddings/wikipedia2vec_old',
        )
    )

@dataclass(frozen=True)
class NoEmbeddingsCnf(BaseCnf):
    """No embeddings configuration."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), use_article_embeddings=False, use_entity_embeddings=False)
    )

@dataclass(frozen=True)
class ArticleOnlyCnf2(ArticleOnlyCnf):
    """Article-only configuration without entity embeddings."""
    random_seed: int = 294613


@dataclass(frozen=True)
class ArticleOnlyCnf3(ArticleOnlyCnf):
    """Article-only configuration without entity embeddings."""
    random_seed: int = 999751


@dataclass(frozen=True)
class ArticleOnlyCnf4(ArticleOnlyCnf):
    """Article-only configuration without entity embeddings."""
    random_seed: int = 212654


@dataclass(frozen=True)
class ArticleOnlyCnf5(ArticleOnlyCnf):
    """Article-only configuration without entity embeddings."""
    random_seed: int = 984621


@dataclass(frozen=True)
class DebugCnf(BaseCnf):
    """Debug configuration for quick local runs."""
    model: ModelCnf = field(
        default_factory=lambda: replace(ModelCnf(), nn_type='mlp_gelu')
    )
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
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            use_entity_relevance_weights=True,
            entity_pooling='weighted_mean',
        )
    )

    random_seed: int = 2
    tuning: ThresholdTuningCnf = field(
        default_factory=lambda: replace(ThresholdTuningCnf(), enabled=True)
    )
    debug: bool = True



@dataclass(frozen=True)
class WPEntitiesEnDeCnf(BaseCnf):
    """Entity-enhanced configuration with English and German entity embeddings."""
    emb: EmbeddingCnf = field(default_factory=lambda: replace(EmbeddingCnf(), entity_langs=('en', 'de')))


@dataclass(frozen=True)
class WPEntitiesEnEsCnf(BaseCnf):
    """Entity-enhanced configuration with English and Spanish entity embeddings."""
    emb: EmbeddingCnf = field(default_factory=lambda: replace(EmbeddingCnf(), entity_langs=('en', 'es')))


@dataclass(frozen=True)
class WPEntitiesEnNlCnf(PreBaseCnfWithHPO):
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
class WPEntitiesAllLangsCnf(PreBaseCnfWithHPO):
    """Entity-enhanced configuration with all supported entity embedding languages."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_langs=ALL_ENTITY_LANGS)
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
class WPEntitiesRelTH5(PreBaseCnfWithHPO):
    tuning: ThresholdTuningCnf = field(
        default_factory=lambda: replace(ThresholdTuningCnf(), enabled=True)
    )
    """Entity-enhanced configuration with English, German and Czech entity embeddings."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            entity_langs=('en',),
            entity_relevance_threshold=5.0,
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
class WPEntitiesNlCnf(PreBaseCnfWithHPO):
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
class WPEntitiesMentionWeightedSumCnf(PreBaseCnfWithHPO):
    """Entity-enhanced configuration with mention-weighted sum pooling enabled."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            entity_pooling='weighted_sum',
        )
    )
    tuning: ThresholdTuningCnf = field(
        default_factory=lambda: replace(ThresholdTuningCnf(), enabled=True)
    )


@dataclass(frozen=True)
class WPEntitiesRelevanceWeightedSumCnf(PreBaseCnfWithHPO):
    """Entity-enhanced configuration with relevance-weighted sum pooling enabled."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            entity_pooling='weighted_sum_relevance',
        )
    )
    tuning: ThresholdTuningCnf = field(
        default_factory=lambda: replace(ThresholdTuningCnf(), enabled=True)
    )




def resolve_paths(config: BaseCnf, root_dir: str | Path) -> BaseCnf:
    """Return a config with absolute paths resolved from ``root_dir``.

    Recursively rebases an attached ``assembly`` block (when present) so each
    member's nested config also gets its paths resolved.
    """
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
    assembly = getattr(config, 'assembly', None)
    if assembly is None:
        return replace(config, paths=resolved_paths)
    resolved_assembly = _resolve_assembly(assembly=assembly, root_path=root_path)
    return replace(config, paths=resolved_paths, assembly=resolved_assembly)


def _resolve_assembly(*, assembly: AssemblyCnf, root_path: Path) -> AssemblyCnf:
    """Return an ``AssemblyCnf`` with all member paths rebased on ``root_path``.

    For each member: rebases ``thresholds_path`` and recursively resolves
    the embedded ``config``'s ``paths`` block. Absolute paths pass through
    unchanged because ``Path / abs`` returns the absolute path.
    """
    resolved_members: list[AssemblyMemberCnf] = []
    for member in assembly.members:
        resolved_thr = (
            str(root_path / member.thresholds_path)
            if member.thresholds_path else member.thresholds_path
        )
        resolved_config = resolve_paths(config=member.config, root_dir=root_path)
        resolved_members.append(replace(
            member,
            thresholds_path=resolved_thr,
            config=resolved_config,
        ))
    return replace(assembly, members=tuple(resolved_members))


@dataclass(frozen=True)
class BestWpEntitiesCnf(PreBaseCnfWithHPO):
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
class BestWpEntitiesF1Cnf(BestWpEntitiesCnf):
    train: TrainingCnf = field(default_factory=lambda: replace(TrainingCnf(), early_stopping_metric='f1'))
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
class WpEntitiesTunedCnf(PreBaseCnfWithHPO):
    """Best entity-enhanced config with per-class threshold tuning enabled.

    The dev folds are scanned over a 17-point sigmoid grid (0.10..0.90 by 0.05)
    and per-class thresholds are aggregated by mean across folds, then reused
    when evaluating the final model on test.
    """

    tuning: ThresholdTuningCnf = field(
        default_factory=lambda: replace(ThresholdTuningCnf(), enabled=True)
    )


@dataclass(frozen=True)
class WpEntitiesJV3ClsTunedCnf(WpEntitiesTunedCnf):
    """Best entity-enhanced config with per-class threshold tuning enabled.

    The dev folds are scanned over a 17-point sigmoid grid (0.10..0.90 by 0.05)
    and per-class thresholds are aggregated by mean across folds, then reused
    when evaluating the final model on test.
    """

    tuning: ThresholdTuningCnf = field(
        default_factory=lambda: replace(ThresholdTuningCnf(), enabled=True)
    )
    paths: PathsCnf = field(
        default_factory=lambda: replace(
            PathsCnf(),
            entity_embeddings_dir=f'{DATA_ROOT}/entity_embeddings/entity_embeddings_en_jina_v3_cls',
        )
    )
    
    
@dataclass(frozen=True)
class WpEntitiesJV5ClsTunedCnf(WpEntitiesTunedCnf):
    """Best entity-enhanced config with per-class threshold tuning enabled.

    The dev folds are scanned over a 17-point sigmoid grid (0.10..0.90 by 0.05)
    and per-class thresholds are aggregated by mean across folds, then reused
    when evaluating the final model on test.
    """

    tuning: ThresholdTuningCnf = field(
        default_factory=lambda: replace(ThresholdTuningCnf(), enabled=True)
    )
    paths: PathsCnf = field(
        default_factory=lambda: replace(
            PathsCnf(),
            entity_embeddings_dir=f'{DATA_ROOT}/entity_embeddings/entity_embeddings_en_jina_v5_cls'
        )
    )
    

@dataclass(frozen=True)
class WpEntitiesPmmTunedCnf(WpEntitiesTunedCnf):
    """Best entity-enhanced config with per-class threshold tuning enabled.

    The dev folds are scanned over a 17-point sigmoid grid (0.10..0.90 by 0.05)
    and per-class thresholds are aggregated by mean across folds, then reused
    when evaluating the final model on test.
    """

    tuning: ThresholdTuningCnf = field(
        default_factory=lambda: replace(ThresholdTuningCnf(), enabled=True)
    )
    paths: PathsCnf = field(
        default_factory=lambda: replace(
            PathsCnf(),
            entity_embeddings_dir=f'{DATA_ROOT}/entity_embeddings/entity_embeddings_pmm',
        )
    )
    
    

@dataclass(frozen=True)
class BestWpEntitiesTunedCnf2(BestWpEntitiesCnf):
    """Best entity-enhanced config with per-class threshold tuning enabled.

    The dev folds are scanned over a 17-point sigmoid grid (0.10..0.90 by 0.05)
    and per-class thresholds are aggregated by mean across folds, then reused
    when evaluating the final model on test.
    """

    tuning: ThresholdTuningCnf = field(
        default_factory=lambda: replace(ThresholdTuningCnf(), enabled=True)
    )
    random_seed: int = 53351
    
@dataclass(frozen=True)
class BestWpEntitiesTunedCnf3(BestWpEntitiesCnf):
    """Best entity-enhanced config with per-class threshold tuning enabled.

    The dev folds are scanned over a 17-point sigmoid grid (0.10..0.90 by 0.05)
    and per-class thresholds are aggregated by mean across folds, then reused
    when evaluating the final model on test.
    """

    tuning: ThresholdTuningCnf = field(
        default_factory=lambda: replace(ThresholdTuningCnf(), enabled=True)
    )
    random_seed: int = 163485
    
    
@dataclass(frozen=True)
class BestWpEntitiesTunedCnf4(BestWpEntitiesCnf):
    """Best entity-enhanced config with per-class threshold tuning enabled.

    The dev folds are scanned over a 17-point sigmoid grid (0.10..0.90 by 0.05)
    and per-class thresholds are aggregated by mean across folds, then reused
    when evaluating the final model on test.
    """

    tuning: ThresholdTuningCnf = field(
        default_factory=lambda: replace(ThresholdTuningCnf(), enabled=True)
    )
    random_seed: int = 61144
    
    
@dataclass(frozen=True)
class BestWpEntitiesTunedCnf5(BestWpEntitiesCnf):
    """Best entity-enhanced config with per-class threshold tuning enabled.

    The dev folds are scanned over a 17-point sigmoid grid (0.10..0.90 by 0.05)
    and per-class thresholds are aggregated by mean across folds, then reused
    when evaluating the final model on test.
    """

    tuning: ThresholdTuningCnf = field(
        default_factory=lambda: replace(ThresholdTuningCnf(), enabled=True)
    )
    random_seed: int = 8689129
  
    
@dataclass(frozen=True)
class BestWpEntitiesAttentionCnf(WpEntitiesTunedCnf):
    """Best entity-enhanced config with explicit attention over entities."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            entity_pooling='no_pooling',
        )
    )
    model: ModelCnf = field(
        default_factory=lambda: replace(
            ModelCnf(),
            nn_type='entity_attention_mlp',
            attention_hidden_dim=512,
        )
    )


@dataclass(frozen=True)
class WPEntitiesAttentionHPOCnf(WpEntitiesTunedCnf):
    """Entity-enhanced configuration with explicit attention over entities."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            entity_pooling='no_pooling',
        )
    )
    model: ModelCnf = field(
        default_factory=lambda: replace(
            ModelCnf(),
            nn_type='entity_attention_mlp',
            attention_hidden_dim=128,
        )
    )
    hparam: HyperparamSpace = field(
        default_factory=lambda: replace(
            HyperparamSpace(),
            hidden_dims=(4096,),
            dropouts1=(0.0,),
            dropouts2=(0.5,),
            attention_hidden_dims=(64, 128, 256, 512),
            attention_dropouts=(0.0,0.3,),

        )
    )
    
@dataclass(frozen=True)
class WPEntitiesPmmAttentionHPOCnf(WpEntitiesPmmTunedCnf):
    """Entity-enhanced configuration with explicit attention over entities."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            entity_pooling='no_pooling',
        )
    )
    model: ModelCnf = field(
        default_factory=lambda: replace(
            ModelCnf(),
            nn_type='entity_attention_mlp',
            attention_hidden_dim=128,
        )
    )
    hparam: HyperparamSpace = field(
        default_factory=lambda: replace(
            HyperparamSpace(),
            hidden_dims=(1024,),
            dropouts1=(0.0,),
            dropouts2=(0.5,),
            attention_hidden_dims=(64, 128, 256, 512),
            attention_dropouts=(0.0,0.3,),

        )
    )
    
@dataclass(frozen=True)
class WikipediaArticleEntitiesCnf(BaseCnfWithHPO):
    """Entity-enhanced configuration using Wikidata description embeddings."""

    paths: PathsCnf = field(
        default_factory=lambda: replace(
            PathsCnf(),
            entity_embeddings_dir=f'{DATA_ROOT}/entity_embeddings/selected-article-embeddings',
        )
    )

@dataclass(frozen=True)	
class WArticleAtentionCnf(WikipediaArticleEntitiesCnf):
    """Entity-enhanced configuration with explicit attention over entities."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            entity_pooling='no_pooling',
        )
    )
    model: ModelCnf = field(
        default_factory=lambda: replace(
            ModelCnf(),
            nn_type='entity_attention_mlp',
            attention_hidden_dim=128,
        )
    )
    hparam: HyperparamSpace = field(
        default_factory=lambda: replace(
            HyperparamSpace(),
            hidden_dims=(1024,),
            dropouts1=(0.0,),
            dropouts2=(0.3,),
            attention_hidden_dims=(64, 128, 256, 512),
            attention_dropouts=(0.0,0.3,),

        )
    )

@dataclass(frozen=True)
class Wikipedia2VecEntitiesCnf(BaseCnfWithHPO):
    paths: PathsCnf = field(
        default_factory=lambda: replace(
            PathsCnf(),
            entity_embeddings_dir=f'{DATA_ROOT}/entity_embeddings/wikipedia2vec_old',
        )
    )
    

@dataclass(frozen=True)	
class W2VecAttentionHPOCnf(Wikipedia2VecEntitiesCnf):
    """Entity-enhanced configuration with explicit attention over entities."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            entity_pooling='no_pooling',
        )
    )
    model: ModelCnf = field(
        default_factory=lambda: replace(
            ModelCnf(),
            nn_type='entity_attention_mlp',
            attention_hidden_dim=128,
        )
    )
    hparam: HyperparamSpace = field(
        default_factory=lambda: replace(
            HyperparamSpace(),
            hidden_dims=(1024,),
            dropouts1=(0.1,),
            dropouts2=(0.3,),
            attention_hidden_dims=(64, 128, 256, 512),
            attention_dropouts=(0.0,0.3,),

        )
    )  
    
@dataclass(frozen=True)
class WPEntitiesAttentionHPOCnf2(WpEntitiesTunedCnf):
    """Entity-enhanced configuration with explicit attention over entities."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            entity_pooling='no_pooling',
        )
    )
    model: ModelCnf = field(
        default_factory=lambda: replace(
            ModelCnf(),
            nn_type='entity_attention_mlp',
            attention_hidden_dim=128,
        )
    )
    hparam: HyperparamSpace = field(
        default_factory=lambda: replace(
            HyperparamSpace(),
            hidden_dims=(1024,),
            dropouts1=(0.0,),
            dropouts2=(0.0,),
            attention_hidden_dims=(512,),
            attention_dropouts=(0.0, 0.1, 0.3, 0.5),

        )
    )

@dataclass(frozen=True)
class BestArticleOnlyCnf(PreBaseCnfWithHPO):
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


@dataclass(frozen=True)
class ArticleOnlyGeluCnf(ArticleOnlyCnf):
    """Article-only config using an MLP with GELU activation."""
    model: ModelCnf = field(
        default_factory=lambda: replace(ModelCnf(), nn_type='mlp_gelu')
    )


@dataclass(frozen=True)
class ArticleOnlySkipCnf(ArticleOnlyCnf):
    """Article-only config using an MLP with skip connection (concat input + hidden)."""
    model: ModelCnf = field(
        default_factory=lambda: replace(ModelCnf(), nn_type='skip_mlp')
    )


@dataclass(frozen=True)
class ArticleOnlyLeakyCnf(ArticleOnlyCnf):
    """Article-only config using an MLP with Leaky ReLU activation."""
    model: ModelCnf = field(
        default_factory=lambda: replace(ModelCnf(), nn_type='leaky_mlp')
    )


@dataclass(frozen=True)
class ArticleOnlyPriorCnf(ArticleOnlyCnf):
    """Article-only GELU with per-class output bias initialized from train priors."""
    model: ModelCnf = field(
        default_factory=lambda: replace(ModelCnf(), bias_from_prior=True)
    )


########################################################
# Best configurations for different relevance thresholds
########################################################


@dataclass(frozen=True)
class BestWpentitiesAllLangsCnf(PreBaseCnfWithHPO):
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
        default_factory=lambda: replace(EmbeddingCnf(), entity_langs=ALL_ENTITY_LANGS)
    )


@dataclass(frozen=True)
class BestWpentitiesNlCnf(PreBaseCnfWithHPO):
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
class BestWPEntitiesENNLCnf(PreBaseCnfWithHPO):
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
    
    



@dataclass(frozen=True)
class Wikipedia2VecEntitiesAllLangsCnf(PreBaseCnfWithHPO):
    """Wikipedia2Vec entity embedding config with all supported languages."""
    paths: PathsCnf = field(
        default_factory=lambda: replace(
            PathsCnf(),
            entity_embeddings_dir=f'{DATA_ROOT}/entity_embeddings/wikipedia2vec_old',
        )
    )
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_langs=ALL_ENTITY_LANGS)
    )


@dataclass(frozen=True)
class BestWikipedia2VecEntitiesCnf(PreBaseCnfWithHPO):
    paths: PathsCnf = field(
        default_factory=lambda: replace(
            PathsCnf(),
            entity_embeddings_dir=f'{DATA_ROOT}/entity_embeddings/wikipedia2vec_old',
        )
    )
    hparam: HyperparamSpace = field(default_factory=lambda: replace(
            HyperparamSpace(),
            hidden_dims=(1024,),
            dropouts1=(0.0,),
            dropouts2=(0.0, ),
            learning_rates=(0.00037,),
        )
    )


@dataclass(frozen=True)
class ArticleOnlyTunedCnf(ArticleOnlyCnf):
    tuning: ThresholdTuningCnf = field(
        default_factory=lambda: replace(ThresholdTuningCnf(), enabled=True)
    )
    
@dataclass(frozen=True)
class ArticleOnlyTunedDiffThresholdsCnf(ArticleOnlyCnf):
    tuning: ThresholdTuningCnf = field(
        default_factory=lambda: replace(ThresholdTuningCnf(), enabled=True,
                                        thresholds=([round(0.05 * i, 2) for i in range(2, 14)]))
    )   
    hparam: HyperparamSpace = field(
        default_factory=lambda: replace(
            HyperparamSpace(),
            hidden_dims=(1024,),
            dropouts1=(0.0,),
            dropouts2=(0.3, ),
            learning_rates=(0.00037,),
        )
    )

@dataclass(frozen=True)
class BestArticleOnlyTunedCnf2(BestArticleOnlyCnf):
    tuning: ThresholdTuningCnf = field(
        default_factory=lambda: replace(ThresholdTuningCnf(), enabled=True)
    )
    random_seed: int = 53351

@dataclass(frozen=True)  
class BestArticleOnlyTunedCnf3(BestArticleOnlyCnf):
    tuning: ThresholdTuningCnf = field(
        default_factory=lambda: replace(ThresholdTuningCnf(), enabled=True)
    )
    random_seed: int = 163485
    
@dataclass(frozen=True)  
class BestArticleOnlyTunedCnf4(BestArticleOnlyCnf):
    tuning: ThresholdTuningCnf = field(
        default_factory=lambda: replace(ThresholdTuningCnf(), enabled=True)
    )
    random_seed: int = 61144
    
@dataclass(frozen=True)  
class BestArticleOnlyTunedCnf5(BestArticleOnlyCnf):
    tuning: ThresholdTuningCnf = field(
        default_factory=lambda: replace(ThresholdTuningCnf(), enabled=True)
    )
    random_seed: int = 8689129


@dataclass(frozen=True)
class WpEntitiesGeluCnf(PreBaseCnfWithHPO):
    """Entity-enhanced config using an MLP with GELU activation."""
    model: ModelCnf = field(
        default_factory=lambda: replace(ModelCnf(), nn_type='mlp_gelu')
    )


@dataclass(frozen=True)
class WpEntitiesSkipCnf(PreBaseCnfWithHPO):
    """Entity-enhanced config using an MLP with skip connection (concat input + hidden)."""
    model: ModelCnf = field(
        default_factory=lambda: replace(ModelCnf(), nn_type='skip_mlp')
    )


@dataclass(frozen=True)
class WpEntitiesLeakyCnf(PreBaseCnfWithHPO):
    """Entity-enhanced config using an MLP with Leaky ReLU activation."""
    model: ModelCnf = field(
        default_factory=lambda: replace(ModelCnf(), nn_type='leaky_mlp')
    )


@dataclass(frozen=True)
class WpEntitiesPriorCnf(PreBaseCnfWithHPO):
    """Entity-enhanced GELU with per-class output bias initialized from train priors."""
    model: ModelCnf = field(
        default_factory=lambda: replace(ModelCnf(), bias_from_prior=True)
    )


@dataclass(frozen=True)
class WikidataDescriptionEntitiesCnf(BaseCnfWithHPO):
    """Entity-enhanced configuration using Wikidata description embeddings."""

    paths: PathsCnf = field(
        default_factory=lambda: replace(
            PathsCnf(),
            entity_embeddings_dir=f'{DATA_ROOT}/entity_embeddings/WikidataDescription',
        )
    )


@dataclass(frozen=True)
class WikipediaIntroEntitiesCnf(BaseCnfWithHPO):
    """Entity-enhanced configuration using Wikidata description embeddings."""

    paths: PathsCnf = field(
        default_factory=lambda: replace(
            PathsCnf(),
            entity_embeddings_dir=f'{DATA_ROOT}/entity_embeddings/cuted-article-embeddings',
        )
    )
    


@dataclass(frozen=True)
class Wikipedia2VecEntityOnlyCnf(Wikipedia2VecEntitiesCnf):
    """Wikipedia2Vec entity embeddings without article embeddings."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), use_article_embeddings=False)
    )


@dataclass(frozen=True)
class WikidataDescriptionEntityOnlyCnf(WikidataDescriptionEntitiesCnf):
    """Wikidata description entity embeddings without article embeddings."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), use_article_embeddings=False)
    )


@dataclass(frozen=True)
class WikipediaIntroEntityOnlyCnf(WikipediaIntroEntitiesCnf):
    """Cuted-article entity embeddings without article embeddings."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), use_article_embeddings=False)
    )


@dataclass(frozen=True)
class WikipediaArticleEntityOnlyCnf(WikipediaArticleEntitiesCnf):
    """Selected-article entity embeddings without article embeddings."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), use_article_embeddings=False)
    )


@dataclass(frozen=True)
class Assembly1Cnf(BaseCnf):
    """Dual-model assembly demo using two tuned single-model configs.

    Each member directly carries a full pipeline config instance. The
    ``thresholds_path`` files come from prior single-model runs in
    ``results/saved_models`` and drive both per-fold evaluation and the
    final stitched per-class thresholds.

    ``debug`` is forced to ``False`` here so each member's per-member
    ``run_cv`` runs the full k-fold loop. ``BaseCnf.debug`` defaults to
    ``True`` for fast local single-model iteration; that default would
    silently collapse the assembly to a single fold.
    """
    debug: bool = field(default_factory=lambda: False)
    assembly: AssemblyCnf = field(
        default_factory=lambda: AssemblyCnf(
            enabled=True,
            members=(
                AssemblyMemberCnf(
                    config=ArticleOnlyTunedCnf(),
                    thresholds_path='/home/prokop/Git/entity-enhance-classification/results/saved_models/best_article_only_tuned_20260507_102132/thresholds.json',
                    label='article_only_tuned',
                ),
                AssemblyMemberCnf(
                    config=WpEntitiesTunedCnf(),
                    thresholds_path='/home/prokop/Git/entity-enhance-classification/results/saved_models/best_wpentities_tuned_20260507_102033/thresholds.json',
                    label='wpentities_tuned',
                ),
            ),
        )
    )

@dataclass(frozen=True)
class Assembly2Cnf(BaseCnf):
    """Dual-model assembly with sign-test per-class selection.

    Same member layout as :class:`Assembly1Cnf` but the primary member is
    kept by default and only swapped to the non-primary on a per-class
    basis when that non-primary beats the primary in ``folds - 1`` or
    more CV folds.
    """
    debug: bool = field(default_factory=lambda: False)
    assembly: AssemblyCnf = field(
        default_factory=lambda: AssemblyCnf(
            enabled=True,
            sign_test=True,
            members=(
                AssemblyMemberCnf(
                    config=WpEntitiesTunedCnf(),
                    thresholds_path='/home/prokop/Git/entity-enhance-classification/results/saved_models/best_wpentities_tuned_20260507_102033/thresholds.json',
                    label='wpentities_tuned',
                ),
                AssemblyMemberCnf(
                    config=ArticleOnlyTunedCnf(),
                    thresholds_path='/home/prokop/Git/entity-enhance-classification/results/saved_models/best_article_only_tuned_20260507_102132/thresholds.json',
                    label='article_only_tuned',
                ),
            ),
        )
    )

@dataclass(frozen=True)
class Assembly3Cnf(BaseCnf):
    debug: bool = field(default_factory=lambda: False)
    assembly: AssemblyCnf = field(
        default_factory=lambda: AssemblyCnf(
            enabled=True,
            sign_test=True,
            members=(
                AssemblyMemberCnf(
                    config=BestWpEntitiesAttentionCnf(),
                    thresholds_path='/home/prokop/Git/entity-enhance-classification/results/saved_models/wpentities_attention_20260514_014702/custom_thresholds.json',
                    label='wpentities_tuned+attention',
                ),
                AssemblyMemberCnf(
                    config=ArticleOnlyTunedCnf(),
                    thresholds_path='/home/prokop/Git/entity-enhance-classification/results/saved_models/best_article_only_tuned_20260513_230745/custom_thresholds.json',
                    label='article_only_tuned+attention',
                ),
            ),
        )
    )

@dataclass(frozen=True)
class AssemblyDebug(BaseCnf):
    """Dual-model assembly demo using two ``DebugCnf`` instances.

    Each member directly carries a full pipeline config instance. Threshold
    files come from prior single-model debug runs in ``results/saved_models``.
    """
    assembly: AssemblyCnf = field(
        default_factory=lambda: AssemblyCnf(
            enabled=True,
            members=(
                AssemblyMemberCnf(
                    config=DebugCnf(),
                    thresholds_path='/home/prokop/Git/entity-enhance-classification/results/saved_models/debug_20260508_182714/custom_thresholds.json',
                    label='debug1',
                ),
                AssemblyMemberCnf(
                    config=DebugCnf(),
                    thresholds_path='/home/prokop/Git/entity-enhance-classification/results/saved_models/debug_20260508_182714/custom_thresholds.json',
                    label='debug2',
                ),
            ),
        )
    )

def _iter_subclasses(base_cls: type[Any]) -> tuple[type[Any], ...]:
    """Return all transitive subclasses of ``base_cls``."""
    found: dict[type[Any], None] = {}
    stack = list(base_cls.__subclasses__())
    while stack:
        sub_cls = stack.pop()
        if sub_cls in found:
            continue
        found[sub_cls] = None
        stack.extend(sub_cls.__subclasses__())
    return tuple(found.keys())


def _validate_config_dataclass_decorators() -> None:
    """Fail fast when config subclasses are not declared as frozen dataclasses."""
    for cls in _iter_subclasses(BaseCnf):
        if cls.__module__ != __name__:
            continue

        if '__dataclass_params__' not in cls.__dict__:
            raw_fields = [
                name for name, value in cls.__dict__.items()
                if isinstance(value, Field)
            ]
            details = f', raw fields={raw_fields}' if raw_fields else ''
            raise TypeError(
                f'{cls.__name__} must declare @dataclass(frozen=True){details}'
            )

        if not cls.__dataclass_params__.frozen:
            raise TypeError(f'{cls.__name__} must declare @dataclass(frozen=True)')
 
     
@dataclass(frozen=True)
class TunningLearningRateCnf(BaseCnfWithHPO3):
    """  """
    
    

@dataclass(frozen=True)
class DebugAttentionCnf(DebugCnf):
    """Debug configuration for quick local runs."""
    model: ModelCnf = field(
        default_factory=lambda: replace(ModelCnf(), nn_type='entity_attention_mlp')
    )
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_pooling='no_pooling')
    )


@dataclass(frozen=True)
class TunningLearningRateF1Cnf(BaseCnfWithHPO3):
    """  """
    train: TrainingCnf = field(default_factory=lambda: replace(TrainingCnf(), early_stopping_metric='f1'))
    
def _config_map() -> dict[str, BaseCnf]:
    """Return supported config instances."""
    return {
        # testing longtail features
        'article_only': ArticleOnlyCnf(),
        'article_only_gelu': ArticleOnlyGeluCnf(),
        'article_only_skip': ArticleOnlySkipCnf(),
        'article_only_leaky': ArticleOnlyLeakyCnf(),
        'article_only_prior': ArticleOnlyPriorCnf(),
        'article_only_tuned': ArticleOnlyTunedCnf(),
        
        'wpentities': WpEntitiesCnf(),
        'wpentities_gelu': WpEntitiesGeluCnf(),
        'wpentities_skip': WpEntitiesSkipCnf(),
        'wpentities_leaky': WpEntitiesLeakyCnf(),
        'wpentities_prior': WpEntitiesPriorCnf(),
        'wpentities_tuned': WpEntitiesTunedCnf(),
        
        # comparing entity entity description aproaches 
        'wp_entity_only': EntityOnlyCnf(),
        'wikipedia2vec_entity_only': Wikipedia2VecEntityOnlyCnf(),
        'wikidata_description_entity_only': WikidataDescriptionEntityOnlyCnf(),
        'wikipedia_intro_entity_only': WikipediaIntroEntityOnlyCnf(),
        'wikipedia_article_entity_only': WikipediaArticleEntityOnlyCnf(),
        
        'wikipedia2vec_entities': Wikipedia2VecEntitiesCnf(),
        'wikidata_description_entities': WikidataDescriptionEntitiesCnf(),
        'wikipedia_intro_entities': WikipediaIntroEntitiesCnf(),
        'wikipedia_article_entities': WikipediaArticleEntitiesCnf(),

        # testing different embeddings
        'wpentities_jina_v3_cls': WpEntitiesJV3ClsTunedCnf(),
        'wpentities_jina_v5_cls': WpEntitiesJV5ClsTunedCnf(),
        'wpentities_pmm': WpEntitiesPmmTunedCnf(),
        
    
        'article_only_tuned_diff_thresholds': ArticleOnlyTunedDiffThresholdsCnf(),
        
        
        # entity-pooling 
        'debug_attention': DebugAttentionCnf(),
        'wpentities_attention_hpo': WPEntitiesAttentionHPOCnf(),
        'wpentities_relevance_weighted_sum': WPEntitiesRelevanceWeightedSumCnf(),
        'wpentities_mention_weighted_sum': WPEntitiesMentionWeightedSumCnf(),
        'wpentities_weighted_mean': WPEntitiesWeightedMeanCnf(),
        'wpentities_pmm_attention': WPEntitiesPmmAttentionHPOCnf(),
        'w2vec_attention': W2VecAttentionHPOCnf(),
        'wikipedia_article_entities_attention': WArticleAtentionCnf(),
 
        # rozběhnout ještě hpo na tunning a normal article_only a wpentities
        'debug': DebugCnf(),
        'article_only': ArticleOnlyCnf(),
        'entity_only': EntityOnlyCnf(),
        'no_embeddings': NoEmbeddingsCnf(),
        'wpentities': WpEntitiesCnf(),
        
        'wpentities_rel_th_5': WPEntitiesRelTH5(),
        'wpentities_attention': BestWpEntitiesAttentionCnf(),
        'wpentities_attention_hpo': WPEntitiesAttentionHPOCnf(),
        'wpentities_attention_hpo_2': WPEntitiesAttentionHPOCnf2(),
        
        'wpentities_en_nl': WPEntitiesEnNlCnf(),
        'wpentities_nl': WPEntitiesNlCnf(),
        'wpentities_all_langs': WPEntitiesAllLangsCnf(),
        
        'best_wpentities': BestWpEntitiesCnf(),
        #'best_wpentities_f1': BestWpEntitiesF1Cnf(),
        'best_wpentities_tuned': WpEntitiesTunedCnf(),
        'best_wpentities_tuned_2': BestWpEntitiesTunedCnf2(),
        'best_wpentities_tuned_3': BestWpEntitiesTunedCnf3(),
        'best_wpentities_tuned_4': BestWpEntitiesTunedCnf4(),
        'best_wpentities_tuned_5': BestWpEntitiesTunedCnf5(),
        'best_article_only': BestArticleOnlyCnf(),
        
        'best_article_only_tuned': ArticleOnlyTunedCnf(),
        'best_article_only_tuned_2': BestArticleOnlyTunedCnf2(),
        'best_article_only_tuned_3': BestArticleOnlyTunedCnf3(),
        'best_article_only_tuned_4': BestArticleOnlyTunedCnf4(),
        'best_article_only_tuned_5': BestArticleOnlyTunedCnf5(),
        'best_wpentities_all_langs': BestWpentitiesAllLangsCnf(),
        'best_wpentities_nl': BestWpentitiesNlCnf(),
        'best_wpentities_en_nl': BestWPEntitiesENNLCnf(),
        
        'wikipedia2vec_entities_all_langs': Wikipedia2VecEntitiesAllLangsCnf(),
        'best_wikipedia2vec_entities': BestWikipedia2VecEntitiesCnf(),
        
        
        #'wpentities_2': WpEntitiesCnf2(),
        #'wpentities_3': WpEntitiesCnf3(),
        #'wpentities_4': WpEntitiesCnf4(),
        #'wpentities_5': WpEntitiesCnf5(),
        #'article_only_2': ArticleOnlyCnf2(),
        #'article_only_3': ArticleOnlyCnf3(),
        #'article_only_4': ArticleOnlyCnf4(),
        #'article_only_5': ArticleOnlyCnf5(),
        #'learning_rate': TunningLearningRateCnf(),
        #'learning_rate_f1': TunningLearningRateF1Cnf(),
        'assembly1': Assembly1Cnf(),
        'assembly_debug': AssemblyDebug(),
        'assembly2': Assembly2Cnf(),
        'assembly_attention': Assembly3Cnf(),
    }



def get_config(config_name: str) -> BaseCnf:
    """
    Return config variant by name.

    Supported names:
    - ``debug``: minimal config loading from ``data/debug`` for fast local testing.
    - ``wpentities``: entity-enhanced default setup (gold-chrono-per-dataset).
    - ``article_only``: article embeddings only (entity embeddings disabled).
    - ``entity_only``: entity embeddings only (article embeddings disabled).

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
_validate_config_dataclass_decorators()
PipelineCnf = BaseCnf
