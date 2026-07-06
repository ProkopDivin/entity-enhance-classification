"""Configuration dataclasses for the IPTC entity-enhanced pipeline."""

from dataclasses import Field, asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal, Mapping

DATA_ROOT = '/home/prokop/Git/entity-enhance-classification/data'
ALL_ENTITY_LANGS = ('en', 'de','fr' , 'nl', 'es' , 'cs')
GOLD_ORIGIN_TRAIN_CSV = f'{DATA_ROOT}/gold-origin/all-corpora-train-entities.csv'
GOLD_ORIGIN_TEST_CSV = f'{DATA_ROOT}/gold-origin/all-corpora-test-entities.csv'


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


def _gold_origin_paths(**overrides: Any) -> PathsCnf:
    """Build :class:`PathsCnf` pointing at gold-origin train/test entity CSVs."""
    return replace(
        PathsCnf(),
        train_csv=GOLD_ORIGIN_TRAIN_CSV,
        test_csv=GOLD_ORIGIN_TEST_CSV,
        **overrides,
    )


@dataclass(frozen=True)
class EmbeddingCnf:
    """Embedding loading and fallback-computation parameters."""

    article_model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2-300-0.3'
    article_embedding_dim: int = 384
    embed_svc_url: str = 'http://tau.g:5533'
    entity_lang: str = 'en'
    entity_langs: tuple[str, ...] = ()
    entity_lang_mode: Literal['average', 'fallback'] = 'average'
    entity_relevance_threshold: float = 0.0
    use_entity_relevance_weights: bool = False
    use_article_embeddings: bool = True
    use_entity_embeddings: bool = True
    combine_method: Literal['concat', 'sum'] = 'concat'
    entity_pooling: Literal[
        'sum',
        'mean',
        'weighted_mean',
        'weighted_sum',
        'weighted_mean_relevance',
        'weighted_sum_relevance',
        'no_pooling',
    ] = 'mean'


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
    attention_num_heads: int = 1
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
    early_stopping_min_delta: float = 0.0000000001 # because of small classes the improvement can be very small
    # because the validation set is different than test set - not splited chronologically 
    # we do not want to overfit it - not expecting all the articles be the same
    early_stopping_metric: Literal['loss', 'f1'] = 'loss'
    # If False, skip the extra forward pass over the full training set each epoch (dev
    # validation unchanged). Saves wall time and avoids holding large per-batch logits
    # during that pass; per-epoch train curves in ClearML stay empty.
    train_validation: bool = False


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
        default_factory=lambda: replace(TrainingCnf(), train_validation=False)
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
class WPEntitiesMeanCnf(BaseCnfWithHPO):
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
            entity_pooling='weighted_mean_relevance',
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
            entity_pooling='weighted_mean_relevance',
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
class WPEntitiesRelTH1(BaseCnfWithHPO):
    """Entity-enhanced configuration with English, German and Czech entity embeddings."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            entity_langs=('en',),
            entity_relevance_threshold=1.0,
        )
    )

@dataclass(frozen=True)
class WPEntitiesRelTH2(BaseCnfWithHPO):
    """Entity-enhanced configuration with English, German and Czech entity embeddings."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            entity_langs=('en',),
            entity_relevance_threshold=2.0,
        )
    )
    
@dataclass(frozen=True)
class WPEntitiesRelTH3(BaseCnfWithHPO):
    """Entity-enhanced configuration with English, German and Czech entity embeddings."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            entity_langs=('en',),
            entity_relevance_threshold=3.0,
        )
    )

@dataclass(frozen=True)
class WPEntitiesRelTH4(BaseCnfWithHPO):
    """Entity-enhanced configuration with English, German and Czech entity embeddings."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            entity_langs=('en',),
            entity_relevance_threshold=4.0,
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
    
    
@dataclass(frozen=True)
class WPEntitiesRelTH6(BaseCnfWithHPO):
    """Entity-enhanced configuration with English, German and Czech entity embeddings."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            entity_langs=('en',),
            entity_relevance_threshold=6.0,
        )
    )
    
@dataclass(frozen=True)
class WPEntitiesRelTH7(BaseCnfWithHPO):
    """Entity-enhanced configuration with English, German and Czech entity embeddings."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            entity_langs=('en',),
            entity_relevance_threshold=7.0,
        )
    )
    
    
@dataclass(frozen=True)
class WPEntitiesRelTH8(BaseCnfWithHPO):
    """Entity-enhanced configuration with English, German and Czech entity embeddings."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            entity_langs=('en',),
            entity_relevance_threshold=8.0,
        )
    )
    
    
@dataclass(frozen=True)
class WPEntitiesRelTH9(BaseCnfWithHPO):
    """Entity-enhanced configuration with English, German and Czech entity embeddings."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            entity_langs=('en',),
            entity_relevance_threshold=9.0,
        )
    )

@dataclass(frozen=True)
class WPEntitiesRelTH10(BaseCnfWithHPO):
    """Entity-enhanced configuration with English, German and Czech entity embeddings."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            entity_langs=('en',),
            entity_relevance_threshold=10.0,
        )
    )
    
@dataclass(frozen=True)
class WPEntitiesRelTH11(BaseCnfWithHPO):
    """Entity-enhanced configuration with English, German and Czech entity embeddings."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            entity_langs=('en',),
            entity_relevance_threshold=11.0,
        )
    )
    
    
@dataclass(frozen=True)
class WPEntitiesRelTH12(BaseCnfWithHPO):
    """Entity-enhanced configuration with English, German and Czech entity embeddings."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            entity_langs=('en',),
            entity_relevance_threshold=12.0,
        )
    )
    
    
@dataclass(frozen=True)
class WPEntitiesRelTH13(BaseCnfWithHPO):
    """Entity-enhanced configuration with English, German and Czech entity embeddings."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            entity_langs=('en',),
            entity_relevance_threshold=13.0,
        )
    )
    
@dataclass(frozen=True)
class WPEntitiesRelTH14(BaseCnfWithHPO):
    """Entity-enhanced configuration with English, German and Czech entity embeddings."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            entity_langs=('en',),
            entity_relevance_threshold=14.0,
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
class WPEntitiesRelTH15(BaseCnfWithHPO):
    """Entity-enhanced configuration with English, German and Czech entity embeddings."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            entity_langs=('en',),
            entity_relevance_threshold=15.0,
        )
    )

    
@dataclass(frozen=True)
class WPEntitiesRelTH20(BaseCnfWithHPO):
    """Entity-enhanced configuration with English, German and Czech entity embeddings."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            entity_langs=('en',),
            entity_relevance_threshold=20.0,
        )
    )


@dataclass(frozen=True)
class WPEntitiesRelTH25(BaseCnfWithHPO):
    """Entity-enhanced configuration with English, German and Czech entity embeddings."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            entity_langs=('en',),
            entity_relevance_threshold=25.0,
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
class WPEntitiesMentionWeightedMeanCnf(PreBaseCnfWithHPO):
    """Entity-enhanced configuration with mention-weighted mean pooling enabled."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            entity_pooling='weighted_mean',
        )
    )
    tuning: ThresholdTuningCnf = field(
        default_factory=lambda: replace(ThresholdTuningCnf(), enabled=True)
    )


@dataclass(frozen=True)
class WPEntitiesRelevanceWeightedMeanCnf(PreBaseCnfWithHPO):
    """Entity-enhanced configuration with relevance-weighted mean pooling enabled."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            entity_pooling='weighted_mean_relevance',
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
class BestWpEntitiesTunedCnf(BaseCnfWithHPO):
    hparam: HyperparamSpace = field(
        default_factory=lambda: replace(
            HyperparamSpace(),
            hidden_dims=(4096, ),
            dropouts1=(0.0,),
            dropouts2=(0.5, ),
            learning_rates=(0.00037,),
        )
    )


@dataclass(frozen=True)
class WpEntitiesTunedF1Cnf(PreBaseCnfWithHPO):
    """Best entity-enhanced config with per-class threshold tuning enabled.

    The dev folds are scanned over a 17-point sigmoid grid (0.10..0.90 by 0.05)
    and per-class thresholds are aggregated by mean across folds, then reused
    when evaluating the final model on test.
    """

    train: TrainingCnf = field(
        default_factory=lambda: replace(TrainingCnf(), early_stopping_metric='f1')
    )

@dataclass(frozen=True)
class BestWpEntitiesTunedF1Cnf(BestWpEntitiesTunedCnf):
    """Best entity-enhanced config with per-class threshold tuning enabled.

    The dev folds are scanned over a 17-point sigmoid grid (0.10..0.90 by 0.05)
    and per-class thresholds are aggregated by mean across folds, then reused
    when evaluating the final model on test.
    """

    train: TrainingCnf = field(
        default_factory=lambda: replace(TrainingCnf(), train_validation=True)
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
class WpEntitiesPmmTunedCnf(BaseCnfWithHPO):
    """Best entity-enhanced config with per-class threshold tuning enabled.

    The dev folds are scanned over a 17-point sigmoid grid (0.10..0.90 by 0.05)
    and per-class thresholds are aggregated by mean across folds, then reused
    when evaluating the final model on test.
    """
    paths: PathsCnf = field(
        default_factory=lambda: replace(
            PathsCnf(),
            entity_embeddings_dir=f'{DATA_ROOT}/entity_embeddings/entity_embeddings_pmm',
        )
    )


@dataclass(frozen=True)
class WpEntitiesPmmTunedOnlyCnf(WpEntitiesPmmTunedCnf):
    """Best entity-enhanced config with per-class threshold tuning enabled.

    The dev folds are scanned over a 17-point sigmoid grid (0.10..0.90 by 0.05)
    and per-class thresholds are aggregated by mean across folds, then reused
    when evaluating the final model on test.
    """
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), use_article_embeddings=False)
    )
    
  
    
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
class BestWpEntitiesAttention2Cnf(WpEntitiesTunedCnf):
    """Entity-enhanced config with per-entity attention plus pooled-entity gating."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            entity_pooling='no_pooling',
        )
    )
    model: ModelCnf = field(
        default_factory=lambda: replace(
            ModelCnf(),
            nn_type='entity_attention2_mlp',
            attention_hidden_dim=512,
        )
    )


@dataclass(frozen=True)
class BestWpEntitiesMhaAttentionCnf(WpEntitiesTunedCnf):
    """Best entity-enhanced config with MHA over entities."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            entity_pooling='no_pooling',
        )
    )
    model: ModelCnf = field(
        default_factory=lambda: replace(
            ModelCnf(),
            nn_type='entity_mha_attention_mlp',
            attention_num_heads=8,
        )
    )


@dataclass(frozen=True)
class BestWpEntitiesMhaAttention2Cnf(WpEntitiesTunedCnf):
    """Best entity-enhanced config with MHA and attention-weight scaling."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            entity_pooling='no_pooling',
        )
    )
    model: ModelCnf = field(
        default_factory=lambda: replace(
            ModelCnf(),
            nn_type='entity_mha_attention2_mlp',
            attention_num_heads=8,
        )
    )

@dataclass(frozen=True)
class   WikiIntroAttentionHPOCnf(WpEntitiesTunedCnf):
    """Entity-enhanced configuration with explicit attention over entities."""
    paths: PathsCnf = field(
        default_factory=lambda: replace(
            PathsCnf(),
            entity_embeddings_dir=f'{DATA_ROOT}/entity_embeddings/cuted-article-embeddings',
        )
    )
    
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
            dropouts1=(0.1,),
            dropouts2=(0.5,),
            attention_hidden_dims=(64, 128, 256, 512),  
            attention_dropouts=(0.0, 0.3,),

        )
    )

@dataclass(frozen=True)
class   WPEntitiesAttentionHPOCnf(WpEntitiesTunedCnf):
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
            attention_hidden_dims=(64, 256, 512), # 128, #TODO: return this back 
            attention_dropouts=(0.0, 0.3,),

        )
    )
    
    
@dataclass(frozen=True)
class BestWpEntitiesAttentionHPOCnf(WpEntitiesTunedCnf):
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
            attention_hidden_dim=512,
        )
    )
    hparam: HyperparamSpace = field(
        default_factory=lambda: replace(
            HyperparamSpace(),
            hidden_dims=(4096,),
            dropouts1=(0.0,),
            dropouts2=(0.5,),
            attention_hidden_dims=(512,),
            attention_dropouts=(0.3,),

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
class WPEntitiesAttention2HPOCnf(WPEntitiesAttentionHPOCnf):
    """Entity-enhanced configuration with attention plus pooled-entity gating."""
    model: ModelCnf = field(
        default_factory=lambda: replace(
            ModelCnf(),
            nn_type='entity_attention2_mlp',
        )
    )



@dataclass(frozen=True)
class WPEntitiesPmmAttention2HPOCnf(WPEntitiesAttentionHPOCnf):
    """PMM entity embeddings with attention2 architecture and HPO."""

    model: ModelCnf = field(
        default_factory=lambda: replace(
            ModelCnf(),
            nn_type='entity_attention2_mlp',
        )
    )


@dataclass(frozen=True)
class WArticleAttention2Cnf(WikipediaArticleEntitiesCnf):
    """Wikipedia article entity embeddings with attention2 architecture."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            entity_pooling='no_pooling',
        )
    )
    model: ModelCnf = field(
        default_factory=lambda: replace(
            ModelCnf(),
            nn_type='entity_attention2_mlp',
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
            attention_dropouts=(0.0, 0.3),
        )
    )


@dataclass(frozen=True)
class W2VecAttention2HPOCnf(Wikipedia2VecEntitiesCnf):
    """Wikipedia2Vec entity embeddings with attention2 architecture."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(
            EmbeddingCnf(),
            entity_pooling='no_pooling',
        )
    )
    model: ModelCnf = field(
        default_factory=lambda: replace(
            ModelCnf(),
            nn_type='entity_attention2_mlp',
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
            attention_dropouts=(0.0, 0.3),
        )
    )

###--------------------------------------------------TRY RUNS FOR ATTENTION----------------------
@dataclass(frozen=True)
class TryAttentionBase(WpEntitiesTunedCnf):
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
            attention_hidden_dims=(512,), # 128, #TODO: return this back
            attention_dropouts=(0.3,),

        )
    )
@dataclass(frozen=True)
class WPEntitiesAttention2TryCnf(TryAttentionBase):
    """Entity-enhanced configuration with attention plus pooled-entity gating."""
    model: ModelCnf = field(
        default_factory=lambda: replace(
            ModelCnf(),
            nn_type='entity_attention2_mlp',
        )
    )
    


@dataclass(frozen=True)
class WPEntitiesPmmAttention2TryCnf(TryAttentionBase):
    """PMM entity embeddings with attention2 architecture and HPO."""
    paths: PathsCnf = field(
        default_factory=lambda: replace(
            PathsCnf(),
            entity_embeddings_dir=f'{DATA_ROOT}/entity_embeddings/entity_embeddings_pmm',
        )
    )
    model: ModelCnf = field(
        default_factory=lambda: replace(
            ModelCnf(),
            nn_type='entity_attention2_mlp',
        )
    )
    
@dataclass(frozen=True)
class  WPEntitiesAttentionTryCnf(TryAttentionBase):
    """Entity-enhanced configuration with explicit attention over entities."""
    model: ModelCnf = field(
        default_factory=lambda: replace(
            ModelCnf(),
            nn_type='entity_attention_mlp',
            attention_hidden_dim=128,
        )
    )

    
@dataclass(frozen=True)
class WPEntitiesPmmAttentionTryCnf(TryAttentionBase):
    """Entity-enhanced configuration with explicit attention over entities."""
    paths: PathsCnf = field(
        default_factory=lambda: replace(
            PathsCnf(),
            entity_embeddings_dir=f'{DATA_ROOT}/entity_embeddings/entity_embeddings_pmm',
        )
    )

    model: ModelCnf = field(
        default_factory=lambda: replace(
            ModelCnf(),
            nn_type='entity_attention_mlp',
            attention_hidden_dim=128,
        )
    )


@dataclass(frozen=True)
class WPEntitiesAttention3TryCnf(TryAttentionBase):
    """Entity-enhanced configuration with two-stage softmax gated attention."""
    model: ModelCnf = field(
        default_factory=lambda: replace(
            ModelCnf(),
            nn_type='entity_attention3_mlp',
        )
    )


@dataclass(frozen=True)
class WPEntitiesPmmAttention3TryCnf(TryAttentionBase):
    """PMM entity embeddings with two-stage softmax gated attention."""
    paths: PathsCnf = field(
        default_factory=lambda: replace(
            PathsCnf(),
            entity_embeddings_dir=f'{DATA_ROOT}/entity_embeddings/entity_embeddings_pmm',
        )
    )
    model: ModelCnf = field(
        default_factory=lambda: replace(
            ModelCnf(),
            nn_type='entity_attention3_mlp',
        )
    )


@dataclass(frozen=True)
class WPEntitiesMhaAttentionTryCnf(TryAttentionBase):
    """Entity-enhanced configuration with MHA pooling."""
    model: ModelCnf = field(
        default_factory=lambda: replace(
            ModelCnf(),
            nn_type='entity_mha_attention_mlp',
            attention_num_heads=8,
        )
    )


@dataclass(frozen=True)
class WPEntitiesPmmMhaAttentionTryCnf(TryAttentionBase):
    """PMM entity embeddings with MHA pooling."""
    paths: PathsCnf = field(
        default_factory=lambda: replace(
            PathsCnf(),
            entity_embeddings_dir=f'{DATA_ROOT}/entity_embeddings/entity_embeddings_pmm',
        )
    )
    model: ModelCnf = field(
        default_factory=lambda: replace(
            ModelCnf(),
            nn_type='entity_mha_attention_mlp',
            attention_num_heads=8,
        )
    )


@dataclass(frozen=True)
class WPEntitiesMhaAttention2TryCnf(TryAttentionBase):
    """Entity-enhanced configuration with MHA pooling and gated fusion."""
    model: ModelCnf = field(
        default_factory=lambda: replace(
            ModelCnf(),
            nn_type='entity_mha_attention2_mlp',
            attention_num_heads=8,
        )
    )


@dataclass(frozen=True)
class WPEntitiesPmmMhaAttention2TryCnf(TryAttentionBase):
    """PMM entity embeddings with MHA pooling and gated fusion."""
    paths: PathsCnf = field(
        default_factory=lambda: replace(
            PathsCnf(),
            entity_embeddings_dir=f'{DATA_ROOT}/entity_embeddings/entity_embeddings_pmm',
        )
    )
    model: ModelCnf = field(
        default_factory=lambda: replace(
            ModelCnf(),
            nn_type='entity_mha_attention2_mlp',
            attention_num_heads=8,
        )
    )


@dataclass(frozen=True)
class WPEntitiesMhaAttentionTry1HeadCnf(WPEntitiesMhaAttentionTryCnf):
    """Entity-enhanced MHA pooling with 1 attention head."""
    model: ModelCnf = field(
        default_factory=lambda: replace(
            ModelCnf(),
            nn_type='entity_mha_attention_mlp',
            attention_num_heads=1,
        )
    )


@dataclass(frozen=True)
class WPEntitiesPmmMhaAttentionTry1HeadCnf(WPEntitiesPmmMhaAttentionTryCnf):
    """PMM entity embeddings with MHA pooling and 1 attention head."""
    model: ModelCnf = field(
        default_factory=lambda: replace(
            ModelCnf(),
            nn_type='entity_mha_attention_mlp',
            attention_num_heads=1,
        )
    )


@dataclass(frozen=True)
class WPEntitiesMhaAttention2Try1HeadCnf(WPEntitiesMhaAttention2TryCnf):
    """Entity-enhanced MHA pooling + gated fusion with 1 attention head."""
    model: ModelCnf = field(
        default_factory=lambda: replace(
            ModelCnf(),
            nn_type='entity_mha_attention2_mlp',
            attention_num_heads=1,
        )
    )


@dataclass(frozen=True)
class WPEntitiesPmmMhaAttention2Try1HeadCnf(WPEntitiesPmmMhaAttention2TryCnf):
    """PMM entity embeddings with MHA pooling + gated fusion and 1 attention head."""
    model: ModelCnf = field(
        default_factory=lambda: replace(
            ModelCnf(),
            nn_type='entity_mha_attention2_mlp',
            attention_num_heads=1,
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
class BestArticleOnlyTunedCnf(ArticleOnlyTunedCnf):
    tuning: ThresholdTuningCnf = field(
        default_factory=lambda: replace(ThresholdTuningCnf(), enabled=True)
    )
    hparam: HyperparamSpace = field(default_factory=lambda: replace(
            HyperparamSpace(),
            hidden_dims=(4096,),
            dropouts1=(0.0,),
            dropouts2=(0.5, ),
            learning_rates=(0.00037,),
        )
    )

@dataclass(frozen=True)
class BestArticleOnlyTunedF1Cnf(BestArticleOnlyCnf):
    train: TrainingCnf = field(
        default_factory=lambda: replace(TrainingCnf(), train_validation=True)
    )
    
    train: TrainingCnf = field(
        default_factory=lambda: replace(TrainingCnf(), early_stopping_metric='f1')
    )
    tuning: ThresholdTuningCnf = field(
        default_factory=lambda: replace(ThresholdTuningCnf(), enabled=True)
    )
        
@dataclass(frozen=True)
class BestArticleOnlyTunedCnf(BestArticleOnlyCnf):
    tuning: ThresholdTuningCnf = field(
        default_factory=lambda: replace(ThresholdTuningCnf(), enabled=True)
    )
    train: TrainingCnf = field(
        default_factory=lambda: replace(TrainingCnf(), train_validation=True)
    )
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


#### Entity-source comparison configs on gold-origin train/test corpora ####


@dataclass(frozen=True)
class GoldOriginEntityOnlyCnf(EntityOnlyCnf):
    """WikidataProject entity-only config on gold-origin corpora."""

    paths: PathsCnf = field(default_factory=_gold_origin_paths)


@dataclass(frozen=True)
class GoldOriginWikipedia2VecEntityOnlyCnf(Wikipedia2VecEntityOnlyCnf):
    """Wikipedia2Vec entity-only config on gold-origin corpora."""

    paths: PathsCnf = field(
        default_factory=lambda: _gold_origin_paths(
            entity_embeddings_dir=f'{DATA_ROOT}/entity_embeddings/wikipedia2vec_old',
        )
    )


@dataclass(frozen=True)
class GoldOriginWikidataDescriptionEntityOnlyCnf(WikidataDescriptionEntityOnlyCnf):
    """Wikidata description entity-only config on gold-origin corpora."""

    paths: PathsCnf = field(
        default_factory=lambda: _gold_origin_paths(
            entity_embeddings_dir=f'{DATA_ROOT}/entity_embeddings/WikidataDescription',
        )
    )


@dataclass(frozen=True)
class GoldOriginWikipediaIntroEntityOnlyCnf(WikipediaIntroEntityOnlyCnf):
    """Cuted-article entity-only config on gold-origin corpora."""

    paths: PathsCnf = field(
        default_factory=lambda: _gold_origin_paths(
            entity_embeddings_dir=f'{DATA_ROOT}/entity_embeddings/cuted-article-embeddings',
        )
    )


@dataclass(frozen=True)
class GoldOriginWikipediaArticleEntityOnlyCnf(WikipediaArticleEntityOnlyCnf):
    """Selected-article entity-only config on gold-origin corpora."""

    paths: PathsCnf = field(
        default_factory=lambda: _gold_origin_paths(
            entity_embeddings_dir=f'{DATA_ROOT}/entity_embeddings/selected-article-embeddings',
        )
    )


@dataclass(frozen=True)
class GoldOriginWpEntitiesPmmTunedOnlyCnf(WpEntitiesPmmTunedOnlyCnf):
    """PMM entity-only config on gold-origin corpora."""

    paths: PathsCnf = field(
        default_factory=lambda: _gold_origin_paths(
            entity_embeddings_dir=f'{DATA_ROOT}/entity_embeddings/entity_embeddings_pmm',
        )
    )


@dataclass(frozen=True)
class GoldOriginWpEntitiesPmmTunedCnf(WpEntitiesPmmTunedCnf):
    """PMM entity-enhanced config on gold-origin corpora."""

    paths: PathsCnf = field(
        default_factory=lambda: _gold_origin_paths(
            entity_embeddings_dir=f'{DATA_ROOT}/entity_embeddings/entity_embeddings_pmm',
        )
    )


@dataclass(frozen=True)
class GoldOriginWpEntitiesTunedCnf(WpEntitiesTunedCnf):
    """PMM entity-enhanced config on gold-origin corpora."""

    paths: PathsCnf = field(
        default_factory=lambda: _gold_origin_paths(
            entity_embeddings_dir=f'{DATA_ROOT}/entity_embeddings/WikidataProject',
        )
    )

@dataclass(frozen=True)
class GoldOriginWikipedia2VecEntitiesCnf(Wikipedia2VecEntitiesCnf):
    """Wikipedia2Vec entity-enhanced config on gold-origin corpora."""

    paths: PathsCnf = field(
        default_factory=lambda: _gold_origin_paths(
            entity_embeddings_dir=f'{DATA_ROOT}/entity_embeddings/wikipedia2vec_old',
        )
    )


@dataclass(frozen=True)
class GoldOriginWikidataDescriptionEntitiesCnf(WikidataDescriptionEntitiesCnf):
    """Wikidata description entity-enhanced config on gold-origin corpora."""

    paths: PathsCnf = field(
        default_factory=lambda: _gold_origin_paths(
            entity_embeddings_dir=f'{DATA_ROOT}/entity_embeddings/WikidataDescription',
        )
    )


@dataclass(frozen=True)
class GoldOriginWikipediaIntroEntitiesCnf(WikipediaIntroEntitiesCnf):
    """Cuted-article entity-enhanced config on gold-origin corpora."""

    paths: PathsCnf = field(
        default_factory=lambda: _gold_origin_paths(
            entity_embeddings_dir=f'{DATA_ROOT}/entity_embeddings/cuted-article-embeddings',
        )
    )


@dataclass(frozen=True)
class GoldOriginWikipediaArticleEntitiesCnf(WikipediaArticleEntitiesCnf):
    """Selected-article entity-enhanced config on gold-origin corpora."""

    paths: PathsCnf = field(
        default_factory=lambda: _gold_origin_paths(
            entity_embeddings_dir=f'{DATA_ROOT}/entity_embeddings/selected-article-embeddings',
        )
    )


@dataclass(frozen=True)
class WPEntityOnlyMeanCnf(EntityOnlyCnf):
    """Entity-only configuration with mean entity pooling."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), use_article_embeddings=False, entity_pooling='mean')
    )


@dataclass(frozen=True)
class Wikipedia2VecEntityOnlyMeanCnf(Wikipedia2VecEntityOnlyCnf):
    """Wikipedia2Vec entity-only configuration with mean entity pooling."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), use_article_embeddings=False, entity_pooling='mean')
    )


@dataclass(frozen=True)
class WikidataDescriptionEntityOnlyMeanCnf(WikidataDescriptionEntityOnlyCnf):
    """Wikidata description entity-only configuration with mean entity pooling."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), use_article_embeddings=False, entity_pooling='mean')
    )


@dataclass(frozen=True)
class WikipediaIntroEntityOnlyMeanCnf(WikipediaIntroEntityOnlyCnf):
    """Wikipedia intro entity-only configuration with mean entity pooling."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), use_article_embeddings=False, entity_pooling='mean')
    )


@dataclass(frozen=True)
class WikipediaArticleEntityOnlyMeanCnf(WikipediaArticleEntityOnlyCnf):
    """Wikipedia article entity-only configuration with mean entity pooling."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), use_article_embeddings=False, entity_pooling='mean')
    )


@dataclass(frozen=True)
class Wikipedia2VecEntitiesMeanCnf(Wikipedia2VecEntitiesCnf):
    """Wikipedia2Vec entity-enhanced configuration with mean entity pooling."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_pooling='mean')
    )


@dataclass(frozen=True)
class WikidataDescriptionEntitiesMeanCnf(WikidataDescriptionEntitiesCnf):
    """Wikidata description entity-enhanced configuration with mean entity pooling."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_pooling='mean')
    )
    
    
@dataclass(frozen=True)
class WikidataDescriptionEntitiesMeanJinaCnf(WikidataDescriptionEntitiesCnf):
    """Wikidata description entity-enhanced configuration with mean entity pooling."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_pooling='mean')
    )
    paths: PathsCnf = field(
        default_factory=lambda: replace(
            PathsCnf(),
            entity_embeddings_dir=f'{DATA_ROOT}/entity_embeddings/WikidataDescription_jina',
        )
    )
@dataclass(frozen=True)
class WikidataDescriptionEntitiesMeanJinaAllLangsFallbackCnf(WikidataDescriptionEntitiesCnf):
    """Wikidata description entity-enhanced configuration with mean entity pooling."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_pooling='mean',entity_langs=ALL_ENTITY_LANGS, entity_lang_mode='fallback')
    )
    paths: PathsCnf = field(
        default_factory=lambda: replace(
            PathsCnf(),
            entity_embeddings_dir=f'{DATA_ROOT}/entity_embeddings/WikidataDescription_jina',
        )
    )

@dataclass(frozen=True)
class WikidataDescriptionAttentionCnf(WikidataDescriptionEntitiesCnf):
    """Wikidata description entity-enhanced configuration with mean entity pooling."""

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
class WikipediaIntroEntitiesMeanCnf(WikipediaIntroEntitiesCnf):
    """Wikipedia intro entity-enhanced configuration with mean entity pooling."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_pooling='mean')
    )


@dataclass(frozen=True)
class W2VecRelevanceWeightedMeanCnf(WPEntitiesRelevanceWeightedMeanCnf):
    """Wikipedia2Vec embeddings with relevance-weighted mean pooling."""

    paths: PathsCnf = field(default_factory=lambda: replace(Wikipedia2VecEntitiesCnf().paths))


@dataclass(frozen=True)
class W2VecMentionWeightedMeanCnf(WPEntitiesMentionWeightedMeanCnf):
    """Wikipedia2Vec embeddings with mention-weighted mean pooling."""

    paths: PathsCnf = field(default_factory=lambda: replace(Wikipedia2VecEntitiesCnf().paths))


@dataclass(frozen=True)
class WikiIntroRelevanceWeightedMeanCnf(WPEntitiesRelevanceWeightedMeanCnf):
    """Wikipedia intro embeddings with relevance-weighted mean pooling."""

    paths: PathsCnf = field(default_factory=lambda: replace(WikiIntroAttentionHPOCnf().paths))


@dataclass(frozen=True)
class WikiIntroMentionWeightedMeanCnf(WPEntitiesMentionWeightedMeanCnf):
    """Wikipedia intro embeddings with mention-weighted mean pooling."""

    paths: PathsCnf = field(default_factory=lambda: replace(WikiIntroAttentionHPOCnf().paths))


@dataclass(frozen=True)
class WikipediaArticleEntitiesMeanCnf(WikipediaArticleEntitiesCnf):
    """Wikipedia article entity-enhanced configuration with mean entity pooling."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_pooling='mean')
    )


@dataclass(frozen=True)
class WpEntitiesPmmTunedSumCnf(WpEntitiesPmmTunedCnf):
    """PMM entity-enhanced configuration with sum feature fusion."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), combine_method='sum')
    )


@dataclass(frozen=True)
class WikidataDescriptionEntitiesSumCnf(WikidataDescriptionEntitiesCnf):
    """Wikidata description entity-enhanced configuration with sum feature fusion."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), combine_method='sum')
    )


@dataclass(frozen=True)
class WikipediaIntroEntitiesSumCnf(WikipediaIntroEntitiesCnf):
    """Wikipedia intro entity-enhanced configuration with sum feature fusion."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), combine_method='sum')
    )


@dataclass(frozen=True)
class WikipediaArticleEntitiesSumCnf(WikipediaArticleEntitiesCnf):
    """Wikipedia article entity-enhanced configuration with sum feature fusion."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), combine_method='sum')
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
        default_factory=lambda: replace(ModelCnf(), nn_type='entity_attention2_mlp')
    )
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_pooling='no_pooling')
    )


@dataclass(frozen=True)
class DebugAttention3Cnf(DebugCnf):
    """Debug configuration for two-stage softmax gated attention."""
    model: ModelCnf = field(
        default_factory=lambda: replace(ModelCnf(), nn_type='entity_attention3_mlp')
    )
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_pooling='no_pooling')
    )


@dataclass(frozen=True)
class DebugMhaAttentionCnf(DebugCnf):
    """Debug configuration for MHA entity pooling."""
    model: ModelCnf = field(
        default_factory=lambda: replace(
            ModelCnf(),
            nn_type='entity_mha_attention_mlp',
            attention_num_heads=8,
        )
    )
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_pooling='no_pooling')
    )


@dataclass(frozen=True)
class DebugMhaAttention2Cnf(DebugCnf):
    """Debug configuration for MHA entity pooling with gated fusion."""
    model: ModelCnf = field(
        default_factory=lambda: replace(
            ModelCnf(),
            nn_type='entity_mha_attention2_mlp',
            attention_num_heads=8,
        )
    )
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_pooling='no_pooling')
    )


@dataclass(frozen=True)
class TunningLearningRateF1Cnf(BaseCnfWithHPO3):
    """  """
    train: TrainingCnf = field(default_factory=lambda: replace(TrainingCnf(), early_stopping_metric='f1'))

#### Best entity-enhanced configs with per-class threshold tuning enabled and specific languages ####

@dataclass(frozen=True)
class BWpEntitiesTunedAllLangsCnf(BestWpEntitiesTunedCnf):
    """Best entity-enhanced config with per-class threshold tuning enabled and all languages."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_langs=ALL_ENTITY_LANGS)
    )
    
@dataclass(frozen=True)
class BWpEntitiesTunedCsCnf(BestWpEntitiesTunedCnf):
    """Best entity-enhanced config with per-class threshold tuning enabled and all languages."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_langs=('cs',))
    )
    
@dataclass(frozen=True)
class BWpEntitiesTunedDeCnf(BestWpEntitiesTunedCnf):
    """Best entity-enhanced config with per-class threshold tuning enabled and all languages."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_langs=('de',))
    )

@dataclass(frozen=True)
class BWpEntitiesTunedNlCnf(BestWpEntitiesTunedCnf):
    """Best entity-enhanced config with per-class threshold tuning enabled and all languages."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_langs=('nl',))
    )

@dataclass(frozen=True)
class BWpEntitiesTunedFrCnf(BestWpEntitiesTunedCnf):
    """Best entity-enhanced config with per-class threshold tuning enabled and all languages."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_langs=('fr',))
    )

@dataclass(frozen=True)
class BWpEntitiesTunedEsCnf(BestWpEntitiesTunedCnf):
    """Best entity-enhanced config with per-class threshold tuning enabled and all languages."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_langs=('es',))
    )

@dataclass(frozen=True)
class BWpEntitiesTunedEnCnf(BestWpEntitiesTunedCnf):
    """Best entity-enhanced config with per-class threshold tuning enabled and all languages."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_langs=('en',))
    )
    
@dataclass(frozen=True)
class BWpEntitiesTunedEnDeCnf(BestWpEntitiesTunedCnf):
    """Best entity-enhanced config with per-class threshold tuning enabled and all languages."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_langs=('de','en'))
    )

@dataclass(frozen=True)
class BWpEntitiesTunedEnNlCnf(BestWpEntitiesTunedCnf):
    """Best entity-enhanced config with per-class threshold tuning enabled and all languages."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_langs=('nl','en'))
    )  
    
#### Best entity-enhanced configs with per-class threshold tuning enabled and specific languages fallback mode ####

@dataclass(frozen=True)
class BWpEntitiesTunedAllLangsFallbackCnf(BaseCnfWithHPO):
    """Best entity-enhanced config with per-class threshold tuning enabled and all languages."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_langs=ALL_ENTITY_LANGS, entity_lang_mode='fallback')
    )

@dataclass(frozen=True)
class BWpEntitiesTunedAllLangsFallbackAttentionCnf(BaseCnfWithHPO):
    """Best entity-enhanced config with per-class threshold tuning enabled and all languages."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_langs=ALL_ENTITY_LANGS, entity_lang_mode='fallback', entity_pooling='no_pooling')
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
            hidden_dims=(2048,),
            dropouts1=(0.1,),
            dropouts2=(0.5,),
            attention_hidden_dims=(64, 256, 512), # 128, #TODO: return this back 
            attention_dropouts=(0.0, 0.3,),

        )
    )

@dataclass(frozen=True)
class BWpEntitiesTunedEnFallbackCnf(BaseCnfWithHPO):
    """Best entity-enhanced config with per-class threshold tuning enabled and all languages."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_langs=('en',), entity_lang_mode='fallback')
    )
    
@dataclass(frozen=True)
class BWpEntitiesTunedEnDeFallbackCnf(BaseCnfWithHPO):
    """Best entity-enhanced config with per-class threshold tuning enabled and all languages."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_langs=('en', 'de'), entity_lang_mode='fallback')
    )

@dataclass(frozen=True)
class BWpEntitiesTunedNlEnFallbackCnf(BaseCnfWithHPO):
    """Best entity-enhanced config with per-class threshold tuning enabled and all languages."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_langs=('nl','en'), entity_lang_mode='fallback')
    )  
    
@dataclass(frozen=True)
class BWpEntitiesTunedEnNlFallbackCnf(BaseCnfWithHPO):
    """Best entity-enhanced config with per-class threshold tuning enabled and all languages."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_langs=('en','nl'), entity_lang_mode='fallback')
    )  
    
#### Gold-origin twins of the article-only, entity-pooling and language configs ####


def _gold_origin_paths_from(parent: type[BaseCnf]) -> PathsCnf:
    """Build gold-origin paths from a parent config, swapping only the train/test CSVs.

    :param parent: config class whose ``paths`` (entity embeddings dir, etc.) should be preserved
    :return: paths pointing at the gold-origin train/test entity CSVs
    """
    return replace(
        parent().paths,
        train_csv=GOLD_ORIGIN_TRAIN_CSV,
        test_csv=GOLD_ORIGIN_TEST_CSV,
    )


# article-only family
@dataclass(frozen=True)
class GoldOriginArticleOnlyCnf(ArticleOnlyCnf):
    """Article-only config on gold-origin corpora."""

    paths: PathsCnf = field(default_factory=lambda: _gold_origin_paths_from(ArticleOnlyCnf))


@dataclass(frozen=True)
class GoldOriginArticleOnlyGeluCnf(ArticleOnlyGeluCnf):
    """Article-only gelu config on gold-origin corpora."""

    paths: PathsCnf = field(default_factory=lambda: _gold_origin_paths_from(ArticleOnlyGeluCnf))


@dataclass(frozen=True)
class GoldOriginArticleOnlySkipCnf(ArticleOnlySkipCnf):
    """Article-only skip config on gold-origin corpora."""

    paths: PathsCnf = field(default_factory=lambda: _gold_origin_paths_from(ArticleOnlySkipCnf))


@dataclass(frozen=True)
class GoldOriginArticleOnlyLeakyCnf(ArticleOnlyLeakyCnf):
    """Article-only leaky config on gold-origin corpora."""

    paths: PathsCnf = field(default_factory=lambda: _gold_origin_paths_from(ArticleOnlyLeakyCnf))


@dataclass(frozen=True)
class GoldOriginArticleOnlyPriorCnf(ArticleOnlyPriorCnf):
    """Article-only prior config on gold-origin corpora."""

    paths: PathsCnf = field(default_factory=lambda: _gold_origin_paths_from(ArticleOnlyPriorCnf))


@dataclass(frozen=True)
class GoldOriginArticleOnlyTunedCnf(ArticleOnlyTunedCnf):
    """Article-only tuned config on gold-origin corpora."""

    paths: PathsCnf = field(default_factory=lambda: _gold_origin_paths_from(ArticleOnlyTunedCnf))


# wpentities family
@dataclass(frozen=True)
class GoldOriginWpEntitiesBaseCnf(WpEntitiesCnf):
    """WpEntities base config on gold-origin corpora."""

    paths: PathsCnf = field(default_factory=lambda: _gold_origin_paths_from(WpEntitiesCnf))


@dataclass(frozen=True)
class GoldOriginWpEntitiesGeluCnf(WpEntitiesGeluCnf):
    """WpEntities gelu config on gold-origin corpora."""

    paths: PathsCnf = field(default_factory=lambda: _gold_origin_paths_from(WpEntitiesGeluCnf))


@dataclass(frozen=True)
class GoldOriginWpEntitiesSkipCnf(WpEntitiesSkipCnf):
    """WpEntities skip config on gold-origin corpora."""

    paths: PathsCnf = field(default_factory=lambda: _gold_origin_paths_from(WpEntitiesSkipCnf))


@dataclass(frozen=True)
class GoldOriginWpEntitiesLeakyCnf(WpEntitiesLeakyCnf):
    """WpEntities leaky config on gold-origin corpora."""

    paths: PathsCnf = field(default_factory=lambda: _gold_origin_paths_from(WpEntitiesLeakyCnf))


@dataclass(frozen=True)
class GoldOriginWpEntitiesPriorCnf(WpEntitiesPriorCnf):
    """WpEntities prior config on gold-origin corpora."""

    paths: PathsCnf = field(default_factory=lambda: _gold_origin_paths_from(WpEntitiesPriorCnf))


# sum-method family
@dataclass(frozen=True)
class GoldOriginWpEntitiesPmmTunedSumCnf(WpEntitiesPmmTunedSumCnf):
    """PMM sum-pooled entity-enhanced config on gold-origin corpora."""

    paths: PathsCnf = field(default_factory=lambda: _gold_origin_paths_from(WpEntitiesPmmTunedSumCnf))


@dataclass(frozen=True)
class GoldOriginWikidataDescriptionEntitiesSumCnf(WikidataDescriptionEntitiesSumCnf):
    """Wikidata description sum-pooled entity-enhanced config on gold-origin corpora."""

    paths: PathsCnf = field(default_factory=lambda: _gold_origin_paths_from(WikidataDescriptionEntitiesSumCnf))


@dataclass(frozen=True)
class GoldOriginWikipediaIntroEntitiesSumCnf(WikipediaIntroEntitiesSumCnf):
    """Cuted-article sum-pooled entity-enhanced config on gold-origin corpora."""

    paths: PathsCnf = field(default_factory=lambda: _gold_origin_paths_from(WikipediaIntroEntitiesSumCnf))


@dataclass(frozen=True)
class GoldOriginWikipediaArticleEntitiesSumCnf(WikipediaArticleEntitiesSumCnf):
    """Selected-article sum-pooled entity-enhanced config on gold-origin corpora."""

    paths: PathsCnf = field(default_factory=lambda: _gold_origin_paths_from(WikipediaArticleEntitiesSumCnf))


# entity-pooling with mean - entity-only
@dataclass(frozen=True)
class GoldOriginWPEntityOnlyMeanCnf(WPEntityOnlyMeanCnf):
    """Mean-pooled entity-only config on gold-origin corpora."""

    paths: PathsCnf = field(default_factory=lambda: _gold_origin_paths_from(WPEntityOnlyMeanCnf))


@dataclass(frozen=True)
class GoldOriginWikipedia2VecEntityOnlyMeanCnf(Wikipedia2VecEntityOnlyMeanCnf):
    """Wikipedia2Vec mean-pooled entity-only config on gold-origin corpora."""

    paths: PathsCnf = field(default_factory=lambda: _gold_origin_paths_from(Wikipedia2VecEntityOnlyMeanCnf))


@dataclass(frozen=True)
class GoldOriginWikidataDescriptionEntityOnlyMeanCnf(WikidataDescriptionEntityOnlyMeanCnf):
    """Wikidata description mean-pooled entity-only config on gold-origin corpora."""

    paths: PathsCnf = field(default_factory=lambda: _gold_origin_paths_from(WikidataDescriptionEntityOnlyMeanCnf))


@dataclass(frozen=True)
class GoldOriginWikipediaIntroEntityOnlyMeanCnf(WikipediaIntroEntityOnlyMeanCnf):
    """Cuted-article mean-pooled entity-only config on gold-origin corpora."""

    paths: PathsCnf = field(default_factory=lambda: _gold_origin_paths_from(WikipediaIntroEntityOnlyMeanCnf))


@dataclass(frozen=True)
class GoldOriginWikipediaArticleEntityOnlyMeanCnf(WikipediaArticleEntityOnlyMeanCnf):
    """Selected-article mean-pooled entity-only config on gold-origin corpora."""

    paths: PathsCnf = field(default_factory=lambda: _gold_origin_paths_from(WikipediaArticleEntityOnlyMeanCnf))


# entity-pooling with mean - entity-enhanced
@dataclass(frozen=True)
class GoldOriginWPEntitiesMeanCnf(WPEntitiesMeanCnf):
    """Mean-pooled entity-enhanced config on gold-origin corpora."""

    paths: PathsCnf = field(default_factory=lambda: _gold_origin_paths_from(WPEntitiesMeanCnf))


@dataclass(frozen=True)
class GoldOriginWikipedia2VecEntitiesMeanCnf(Wikipedia2VecEntitiesMeanCnf):
    """Wikipedia2Vec mean-pooled entity-enhanced config on gold-origin corpora."""

    paths: PathsCnf = field(default_factory=lambda: _gold_origin_paths_from(Wikipedia2VecEntitiesMeanCnf))


@dataclass(frozen=True)
class GoldOriginWikidataDescriptionEntitiesMeanCnf(WikidataDescriptionEntitiesMeanCnf):
    """Wikidata description mean-pooled entity-enhanced config on gold-origin corpora."""

    paths: PathsCnf = field(default_factory=lambda: _gold_origin_paths_from(WikidataDescriptionEntitiesMeanCnf))


@dataclass(frozen=True)
class GoldOriginWikipediaIntroEntitiesMeanCnf(WikipediaIntroEntitiesMeanCnf):
    """Cuted-article mean-pooled entity-enhanced config on gold-origin corpora."""

    paths: PathsCnf = field(default_factory=lambda: _gold_origin_paths_from(WikipediaIntroEntitiesMeanCnf))


@dataclass(frozen=True)
class GoldOriginWikipediaArticleEntitiesMeanCnf(WikipediaArticleEntitiesMeanCnf):
    """Selected-article mean-pooled entity-enhanced config on gold-origin corpora."""

    paths: PathsCnf = field(default_factory=lambda: _gold_origin_paths_from(WikipediaArticleEntitiesMeanCnf))


# attention and weighted-pooling family
@dataclass(frozen=True)
class GoldOriginWPEntitiesAttentionHPOCnf(WPEntitiesAttentionHPOCnf):
    """Attention-pooled entity-enhanced config on gold-origin corpora."""

    paths: PathsCnf = field(default_factory=lambda: _gold_origin_paths_from(WPEntitiesAttentionHPOCnf))


@dataclass(frozen=True)
class GoldOriginWPEntitiesRelevanceWeightedSumCnf(WPEntitiesRelevanceWeightedSumCnf):
    """Relevance-weighted sum entity-enhanced config on gold-origin corpora."""

    paths: PathsCnf = field(default_factory=lambda: _gold_origin_paths_from(WPEntitiesRelevanceWeightedSumCnf))


@dataclass(frozen=True)
class GoldOriginWPEntitiesMentionWeightedSumCnf(WPEntitiesMentionWeightedSumCnf):
    """Mention-weighted sum entity-enhanced config on gold-origin corpora."""

    paths: PathsCnf = field(default_factory=lambda: _gold_origin_paths_from(WPEntitiesMentionWeightedSumCnf))


@dataclass(frozen=True)
class GoldOriginWPEntitiesRelevanceWeightedMeanCnf(WPEntitiesRelevanceWeightedMeanCnf):
    """Relevance-weighted mean entity-enhanced config on gold-origin corpora."""

    paths: PathsCnf = field(default_factory=lambda: _gold_origin_paths_from(WPEntitiesRelevanceWeightedMeanCnf))


@dataclass(frozen=True)
class GoldOriginWPEntitiesMentionWeightedMeanCnf(WPEntitiesMentionWeightedMeanCnf):
    """Mention-weighted mean entity-enhanced config on gold-origin corpora."""

    paths: PathsCnf = field(default_factory=lambda: _gold_origin_paths_from(WPEntitiesMentionWeightedMeanCnf))


@dataclass(frozen=True)
class GoldOriginWPEntitiesWeightedMeanCnf(WPEntitiesWeightedMeanCnf):
    """Weighted mean entity-enhanced config on gold-origin corpora."""

    paths: PathsCnf = field(default_factory=lambda: _gold_origin_paths_from(WPEntitiesWeightedMeanCnf))


@dataclass(frozen=True)
class GoldOriginWPEntitiesPmmAttentionHPOCnf(WPEntitiesPmmAttentionHPOCnf):
    """PMM attention-pooled entity-enhanced config on gold-origin corpora."""

    paths: PathsCnf = field(default_factory=lambda: _gold_origin_paths_from(WPEntitiesPmmAttentionHPOCnf))


# language fallback family
@dataclass(frozen=True)
class GoldOriginBWpEntitiesTunedEnFallbackCnf(BWpEntitiesTunedEnFallbackCnf):
    """Best tuned en-fallback entity-enhanced config on gold-origin corpora."""

    paths: PathsCnf = field(default_factory=lambda: _gold_origin_paths_from(BWpEntitiesTunedEnFallbackCnf))


@dataclass(frozen=True)
class GoldOriginBWpEntitiesTunedEnDeFallbackCnf(BWpEntitiesTunedEnDeFallbackCnf):
    """Best tuned en-de-fallback entity-enhanced config on gold-origin corpora."""

    paths: PathsCnf = field(default_factory=lambda: _gold_origin_paths_from(BWpEntitiesTunedEnDeFallbackCnf))


@dataclass(frozen=True)
class GoldOriginBWpEntitiesTunedEnNlFallbackCnf(BWpEntitiesTunedEnNlFallbackCnf):
    """Best tuned en-nl-fallback entity-enhanced config on gold-origin corpora."""

    paths: PathsCnf = field(default_factory=lambda: _gold_origin_paths_from(BWpEntitiesTunedEnNlFallbackCnf))


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
        'best_article_only_tuned': BestArticleOnlyTunedCnf(),
        #'article_only_tuned_f1': ArticleOnlyTunedF1Cnf(),
        
        'wpentities': WpEntitiesCnf(),
        'wpentities_gelu': WpEntitiesGeluCnf(),
        'wpentities_skip': WpEntitiesSkipCnf(),
        'wpentities_leaky': WpEntitiesLeakyCnf(),
        'wpentities_prior': WpEntitiesPriorCnf(),
        'wpentities_tuned': WpEntitiesTunedCnf(),
        #'wpentities_tuned_f1': WpEntitiesTunedF1Cnf(),
        
        # comparing entity entitysources 
        'wp_entity_only': EntityOnlyCnf(),
        'wikipedia2vec_entity_only': Wikipedia2VecEntityOnlyCnf(),
        'wikidata_description_entity_only': WikidataDescriptionEntityOnlyCnf(),
        'wikipedia_intro_entity_only': WikipediaIntroEntityOnlyCnf(),
        'wikipedia_article_entity_only': WikipediaArticleEntityOnlyCnf(),
        'wpentities_pmm_tuned_only': WpEntitiesPmmTunedOnlyCnf(),
        
        'wpentities_pmm': WpEntitiesPmmTunedCnf(),
        'wikipedia2vec_entities': Wikipedia2VecEntitiesCnf(),
        'wikidata_description_entities': WikidataDescriptionEntitiesCnf(),
        'wikipedia_intro_entities': WikipediaIntroEntitiesCnf(),
        'wikipedia_article_entities': WikipediaArticleEntitiesCnf(),
        
        
        # comparing entity entitysources on original gold dataset
        'gold_origin_wp_entity_only': GoldOriginEntityOnlyCnf(),
        'gold_origin_wpentities_pmm': GoldOriginWpEntitiesPmmTunedCnf(),
        'gold_origin_wpentities': GoldOriginWpEntitiesTunedCnf(),
        
        'gold_origin_wikipedia2vec_entity_only': GoldOriginWikipedia2VecEntityOnlyCnf(),
        'gold_origin_wikidata_description_entity_only': GoldOriginWikidataDescriptionEntityOnlyCnf(),
        'gold_origin_wikipedia_intro_entity_only': GoldOriginWikipediaIntroEntityOnlyCnf(),
        'gold_origin_wikipedia_article_entity_only': GoldOriginWikipediaArticleEntityOnlyCnf(),
        'gold_origin_wpentities_pmm_tuned_only': GoldOriginWpEntitiesPmmTunedOnlyCnf(),
        
        'gold_origin_wikipedia2vec_entities': GoldOriginWikipedia2VecEntitiesCnf(),
        'gold_origin_wikidata_description_entities': GoldOriginWikidataDescriptionEntitiesCnf(),
        'gold_origin_wikipedia_intro_entities': GoldOriginWikipediaIntroEntitiesCnf(),
        'gold_origin_wikipedia_article_entities': GoldOriginWikipediaArticleEntitiesCnf(),

        # gold-origin twins of article-only, wpentities, pooling, attention and language configs
        'gold_origin_article_only': GoldOriginArticleOnlyCnf(),
        'gold_origin_article_only_gelu': GoldOriginArticleOnlyGeluCnf(),
        'gold_origin_article_only_skip': GoldOriginArticleOnlySkipCnf(),
        'gold_origin_article_only_leaky': GoldOriginArticleOnlyLeakyCnf(),
        'gold_origin_article_only_prior': GoldOriginArticleOnlyPriorCnf(),
        'gold_origin_article_only_tuned': GoldOriginArticleOnlyTunedCnf(),

        'gold_origin_wpentities_base': GoldOriginWpEntitiesBaseCnf(),
        'gold_origin_wpentities_gelu': GoldOriginWpEntitiesGeluCnf(),
        'gold_origin_wpentities_skip': GoldOriginWpEntitiesSkipCnf(),
        'gold_origin_wpentities_leaky': GoldOriginWpEntitiesLeakyCnf(),
        'gold_origin_wpentities_prior': GoldOriginWpEntitiesPriorCnf(),
        'gold_origin_wpentities_tuned': GoldOriginWpEntitiesTunedCnf(),

        'gold_origin_wpentities_pmm_sum': GoldOriginWpEntitiesPmmTunedSumCnf(),
        'gold_origin_wikidata_description_entities_sum': GoldOriginWikidataDescriptionEntitiesSumCnf(),
        'gold_origin_wikipedia_intro_entities_sum': GoldOriginWikipediaIntroEntitiesSumCnf(),
        'gold_origin_wikipedia_article_entities_sum': GoldOriginWikipediaArticleEntitiesSumCnf(),

        'gold_origin_wp_entity_only_mean': GoldOriginWPEntityOnlyMeanCnf(),
        'gold_origin_wikipedia2vec_entity_only_mean': GoldOriginWikipedia2VecEntityOnlyMeanCnf(),
        'gold_origin_wikidata_description_entity_only_mean': GoldOriginWikidataDescriptionEntityOnlyMeanCnf(),
        'gold_origin_wikipedia_intro_entity_only_mean': GoldOriginWikipediaIntroEntityOnlyMeanCnf(),
        'gold_origin_wikipedia_article_entity_only_mean': GoldOriginWikipediaArticleEntityOnlyMeanCnf(),
        'gold_origin_wp_entities_mean': GoldOriginWPEntitiesMeanCnf(),
        'gold_origin_wikipedia2vec_entities_mean': GoldOriginWikipedia2VecEntitiesMeanCnf(),
        'gold_origin_wikidata_description_entities_mean': GoldOriginWikidataDescriptionEntitiesMeanCnf(),
        'gold_origin_wikipedia_intro_entities_mean': GoldOriginWikipediaIntroEntitiesMeanCnf(),
        'gold_origin_wikipedia_article_entities_mean': GoldOriginWikipediaArticleEntitiesMeanCnf(),

        'gold_origin_wpentities_attention_hpo': GoldOriginWPEntitiesAttentionHPOCnf(),
        'gold_origin_wpentities_relevance_weighted_sum': GoldOriginWPEntitiesRelevanceWeightedSumCnf(),
        'gold_origin_wpentities_mention_weighted_sum': GoldOriginWPEntitiesMentionWeightedSumCnf(),
        'gold_origin_wpentities_relevance_weighted_mean': GoldOriginWPEntitiesRelevanceWeightedMeanCnf(),
        'gold_origin_wpentities_mention_weighted_mean': GoldOriginWPEntitiesMentionWeightedMeanCnf(),
        'gold_origin_wpentities_weighted_mean': GoldOriginWPEntitiesWeightedMeanCnf(),
        'gold_origin_wpentities_mean': GoldOriginWPEntitiesMeanCnf(),
        'gold_origin_wpentities_pmm_attention': GoldOriginWPEntitiesPmmAttentionHPOCnf(),

        'gold_origin_wpentities_tuned_en_fallback': GoldOriginBWpEntitiesTunedEnFallbackCnf(),
        'gold_origin_wpentities_tuned_en_de_fallback': GoldOriginBWpEntitiesTunedEnDeFallbackCnf(),
        'gold_origin_wpentities_tuned_en_nl_fallback': GoldOriginBWpEntitiesTunedEnNlFallbackCnf(),

        # sum method
        'wpentities_pmm_sum': WpEntitiesPmmTunedSumCnf(),
        'wikidata_description_entities_sum': WikidataDescriptionEntitiesSumCnf(),
        'wikipedia_intro_entities_sum': WikipediaIntroEntitiesSumCnf(),
        'wikipedia_article_entities_sum': WikipediaArticleEntitiesSumCnf(),
        
        # entity-pooling with mean
        'wp_entity_only_mean': WPEntityOnlyMeanCnf(),
        'wikipedia2vec_entity_only_mean': Wikipedia2VecEntityOnlyMeanCnf(),
        'wikidata_description_entity_only_mean': WikidataDescriptionEntityOnlyMeanCnf(),
        'wikipedia_intro_entity_only_mean': WikipediaIntroEntityOnlyMeanCnf(),
        'wikipedia_article_entity_only_mean': WikipediaArticleEntityOnlyMeanCnf(),
        'wp_entities_mean': WPEntitiesMeanCnf(),
        'wikipedia2vec_entities_mean': Wikipedia2VecEntitiesMeanCnf(),
        'wikidata_description_entities_mean': WikidataDescriptionEntitiesMeanCnf(),
        'wikipedia_intro_entities_mean': WikipediaIntroEntitiesMeanCnf(),
        'wikipedia_article_entities_mean': WikipediaArticleEntitiesMeanCnf(),
        'wikidata_description_jina': WikidataDescriptionEntitiesMeanJinaCnf(),
        'wikidata_description_jina_all_langs_fallback': WikidataDescriptionEntitiesMeanJinaAllLangsFallbackCnf(),


        # testing different embeddings
        'wpentities_jina_v3_cls': WpEntitiesJV3ClsTunedCnf(),
        'wpentities_jina_v5_cls': WpEntitiesJV5ClsTunedCnf(),
        
        
    
        'article_only_tuned_diff_thresholds': ArticleOnlyTunedDiffThresholdsCnf(),
        
        #try runs
        'try_wpentities_attention2': WPEntitiesAttention2TryCnf(),
        'try_wpentities_pmm_attention2': WPEntitiesPmmAttention2TryCnf(),
        'try_wpentities_attention': WPEntitiesAttentionTryCnf(),
        'try_wpentities_pmm_attention': WPEntitiesPmmAttentionTryCnf(),
        'try_wpentities_attention3': WPEntitiesAttention3TryCnf(),
        'try_wpentities_pmm_attention3': WPEntitiesPmmAttention3TryCnf(),
        'try_wpentities_mha_attention_h1': WPEntitiesMhaAttentionTry1HeadCnf(),
        'try_wpentities_pmm_mha_attention_h1': WPEntitiesPmmMhaAttentionTry1HeadCnf(),
        'try_wpentities_mha_attention2_h1': WPEntitiesMhaAttention2Try1HeadCnf(),
        'try_wpentities_pmm_mha_attention2_h1': WPEntitiesPmmMhaAttention2Try1HeadCnf(),
        'try_wpentities_mha_attention_h8': WPEntitiesMhaAttentionTryCnf(),
        'try_wpentities_pmm_mha_attention_h8': WPEntitiesPmmMhaAttentionTryCnf(),
        'try_wpentities_mha_attention2_h8': WPEntitiesMhaAttention2TryCnf(),
        'try_wpentities_pmm_mha_attention2_h8': WPEntitiesPmmMhaAttention2TryCnf(),
        
        # entity-pooling 
        'debug_attention': DebugAttentionCnf(),
        'debug_attention3': DebugAttention3Cnf(),
        'debug_mha_attention': DebugMhaAttentionCnf(),
        'debug_mha_attention2': DebugMhaAttention2Cnf(),
        'wpentities_attention_hpo': WPEntitiesAttentionHPOCnf(),
        'best_wpentities_attention_hpo': BestWpEntitiesAttentionHPOCnf(),
        
        
        'wpentities_relevance_weighted_sum': WPEntitiesRelevanceWeightedSumCnf(),
        'wpentities_mention_weighted_sum': WPEntitiesMentionWeightedSumCnf(),
        'wpentities_relevance_weighted_mean': WPEntitiesRelevanceWeightedMeanCnf(),
        'wpentities_mention_weighted_mean': WPEntitiesMentionWeightedMeanCnf(),
        'w2vec_relevance_weighted_mean': W2VecRelevanceWeightedMeanCnf(),
        'w2vec_mention_weighted_mean': W2VecMentionWeightedMeanCnf(),
        'wikipedia_intro_relevance_weighted_mean': WikiIntroRelevanceWeightedMeanCnf(),
        'wikipedia_intro_mention_weighted_mean': WikiIntroMentionWeightedMeanCnf(),
        'wpentities_weighted_mean': WPEntitiesWeightedMeanCnf(),
        'wpentities_mean': WPEntitiesMeanCnf(),
        'wpentities_pmm_attention': WPEntitiesPmmAttentionHPOCnf(),
        'wikipedia_intro_attention': WikiIntroAttentionHPOCnf(),
        'w2vec_attention': W2VecAttentionHPOCnf(),
        'wikipedia_article_entities_attention': WArticleAtentionCnf(),
        'wikidata_description_attention': WikidataDescriptionAttentionCnf(),
        
        
        # entity-pooling with mean
        
        # relevance treshold (all tuned )
        'wpentities_rel_th_1': WPEntitiesRelTH1(),
        'wpentities_rel_th_2': WPEntitiesRelTH2(),
        'wpentities_rel_th_3': WPEntitiesRelTH3(),
        'wpentities_rel_th_4': WPEntitiesRelTH4(),
        'wpentities_rel_th_5': WPEntitiesRelTH5(),
        'wpentities_rel_th_6': WPEntitiesRelTH6(),
        'wpentities_rel_th_7': WPEntitiesRelTH7(),
        'wpentities_rel_th_8': WPEntitiesRelTH8(),
        'wpentities_rel_th_9': WPEntitiesRelTH9(),
        'wpentities_rel_th_10': WPEntitiesRelTH10(),
        'wpentities_rel_th_11': WPEntitiesRelTH11(),
        'wpentities_rel_th_12': WPEntitiesRelTH12(),
        'wpentities_rel_th_13': WPEntitiesRelTH13(),
        'wpentities_rel_th_14': WPEntitiesRelTH14(),
        'wpentities_rel_th_15': WPEntitiesRelTH15(),
        'wpentities_rel_th_20': WPEntitiesRelTH20(),
        'wpentities_rel_th_25': WPEntitiesRelTH25(),
        
        # languages tests
      
        'wpentities_tuned_en_fallback': BWpEntitiesTunedEnFallbackCnf(),
        'wpentities_tuned_en_de_fallback': BWpEntitiesTunedEnDeFallbackCnf(),
        'wpentities_tuned_en_nl_fallback': BWpEntitiesTunedEnNlFallbackCnf(),
        'wpentities_tuned_nl_en_fallback': BWpEntitiesTunedNlEnFallbackCnf(),
        'wpentities_tuned_all_langs_fallback': BWpEntitiesTunedAllLangsFallbackCnf(),
        
        'wpentities_tuned_cs': BWpEntitiesTunedCsCnf(),
        'wpentities_tuned_de': BWpEntitiesTunedDeCnf(),
        'wpentities_tuned_nl': BWpEntitiesTunedNlCnf(),
        'wpentities_tuned_fr': BWpEntitiesTunedFrCnf(),
        'wpentities_tuned_es': BWpEntitiesTunedEsCnf(),
        'wpentities_tuned_en': BWpEntitiesTunedEnCnf(),
        'wpentities_tuned_en_de': BWpEntitiesTunedEnDeCnf(),
        'wpentities_tuned_en_nl': BWpEntitiesTunedEnNlCnf(),
        'wpentities_tuned_all_langs': BWpEntitiesTunedAllLangsCnf(),
        
        # all the best
        'wpentities_all_langs_fallback_attention': BWpEntitiesTunedAllLangsFallbackAttentionCnf(),
        
        
        # rozběhnout ještě hpo na tunning a normal article_only a wpentities
        'debug': DebugCnf(),
        'article_only': ArticleOnlyCnf(),
        'entity_only': EntityOnlyCnf(),
        'no_embeddings': NoEmbeddingsCnf(),
        'wpentities': WpEntitiesCnf(),
          
        'wpentities_rel_th_5': WPEntitiesRelTH5(),
        'wpentities_attention': BestWpEntitiesAttentionCnf(),
        'wpentities_mha_attention': BestWpEntitiesMhaAttentionCnf(),
        'wpentities_mha_attention2': BestWpEntitiesMhaAttention2Cnf(),
        'wpentities_attention_hpo': WPEntitiesAttentionHPOCnf(),
        
        'wpentities_en_nl': WPEntitiesEnNlCnf(),
        'wpentities_nl': WPEntitiesNlCnf(),
        'wpentities_all_langs': WPEntitiesAllLangsCnf(),
        
        
        #'best_wpentities_f1': BestWpEntitiesF1Cnf(),
        'best_wpentities_tuned': BestWpEntitiesTunedCnf(), 

        #'best_article_only': BestArticleOnlyCnf(),
        #
        #'best_article_only_tuned': BestArticleOnlyTunedCnf(),
        #'best_article_only_tuned_f1': BestArticleOnlyTunedF1Cnf(),
        #'best_article_only_tuned_2': BestArticleOnlyTunedCnf2(),
        #'best_article_only_tuned_3': BestArticleOnlyTunedCnf3(),
        #'best_article_only_tuned_4': BestArticleOnlyTunedCnf4(),
        #'best_article_only_tuned_5': BestArticleOnlyTunedCnf5(),
        #'best_wpentities_all_langs': BestWpentitiesAllLangsCnf(),
        #'best_wpentities_nl': BestWpentitiesNlCnf(),
        #'best_wpentities_en_nl': BestWPEntitiesENNLCnf(),
        
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


