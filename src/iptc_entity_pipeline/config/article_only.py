"""Article-only experiment configs."""

# standard library
from dataclasses import dataclass, field, replace

# project
from iptc_entity_pipeline.config.base import (
    EmbeddingCnf,
    HyperparamSpace,
    ModelCnf,
    PreBaseCnfWithHPO,
    ThresholdTuningCnf,
    TrainingCnf,
)

@dataclass(frozen=True)
class ArticleOnlyCnf(PreBaseCnfWithHPO):
    """Article-only configuration without entity embeddings."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), use_entity_embeddings=False)
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


@dataclass(frozen=True)
class ArticleOnlyTunedCnf(ArticleOnlyCnf):
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


