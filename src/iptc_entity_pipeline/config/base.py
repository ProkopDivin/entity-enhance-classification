"""Core config dataclasses, constants, and path helpers."""

# standard library
import os
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal, Mapping

# Constants: this file lives at src/iptc_entity_pipeline/config/base.py
_DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[3] / 'data'
DATA_ROOT = os.getenv('IPTC_DATA_ROOT', str(_DEFAULT_DATA_ROOT))
ALL_ENTITY_LANGS = ('en', 'de', 'fr', 'nl', 'es', 'cs')
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
    # in original pipeline a corpora was to donsampled during datapreparation i removed the corpora from train data,
    # this is not needed now, im leaving this here  to future use 
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
    entity_lang_mode: Literal['average', 'fallback'] = 'average'
    entity_relevance_threshold: float = 0.0
    remove_types: tuple[str, ...] = ()
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
    early_stopping_min_delta: float = 0.000000001 # because of small classes the improvement can be very small
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
    model_path: str | None = None


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
    hparam: HyperparamSpace = field(
        default_factory=lambda: replace(
            HyperparamSpace(),
            hidden_dims=(100, 384, 1024, 2048, 4096, 8192, 16384,),
            dropouts1=(0.0, 0.1,),
            dropouts2=(0.0, 0.15, 0.3, 0.5,),
            learning_rates=(0.00037,),
        )
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
    resolved_model_path = (
        str(root_path / config.model_path)
        if config.model_path and not Path(config.model_path).is_absolute()
        else config.model_path
    )
    assembly = getattr(config, 'assembly', None)
    if assembly is None:
        return replace(config, paths=resolved_paths, model_path=resolved_model_path)
    resolved_assembly = _resolve_assembly(assembly=assembly, root_path=root_path)
    return replace(config, paths=resolved_paths, assembly=resolved_assembly, model_path=resolved_model_path)


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


