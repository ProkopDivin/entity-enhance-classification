"""WikidataProject / PMM entity experiment configs."""

# standard library
from dataclasses import dataclass, field, replace

# project
from iptc_entity_pipeline.config.base import (
    DATA_ROOT,
    BaseCnfWithHPO,
    BaseCnfWithHPO2,
    EmbeddingCnf,
    HyperparamSpace,
    ModelCnf,
    PathsCnf,
    PreBaseCnfWithHPO,
    ThresholdTuningCnf,
)
from iptc_entity_pipeline.data_loading import remove_types_except

@dataclass(frozen=True)
class WpEntitiesCnf(BaseCnfWithHPO2):
    """Default entity-enhanced configuration."""


def _wp_entities_mean_emb(*, remove_types: tuple[str, ...] = ()) -> EmbeddingCnf:
    """Build mean-pooled embedding config with optional type filtering."""
    return replace(EmbeddingCnf(), entity_pooling='mean', remove_types=remove_types)


def _wp_entities_mean_only_type_emb(*, keep_type: str) -> EmbeddingCnf:
    """Build mean-pooled embedding config that keeps a single entity type."""
    return _wp_entities_mean_emb(remove_types=remove_types_except(keep_type=keep_type))


@dataclass(frozen=True)
class WPEntitiesMeanCnf(BaseCnfWithHPO):
    """Entity-enhanced configuration with mean pooling."""

    emb: EmbeddingCnf = field(default_factory=lambda: _wp_entities_mean_emb())


@dataclass(frozen=True)
class WPEntitiesMeanNoLocationCnf(WPEntitiesMeanCnf):
    """Mean-pooled entity-enhanced config with location entities removed."""

    emb: EmbeddingCnf = field(default_factory=lambda: _wp_entities_mean_emb(remove_types=('location',)))


@dataclass(frozen=True)
class WPEntitiesMeanOnlyEventCnf(WPEntitiesMeanCnf):
    """Mean-pooled config keeping only event entities."""

    emb: EmbeddingCnf = field(default_factory=lambda: _wp_entities_mean_only_type_emb(keep_type='event'))


@dataclass(frozen=True)
class WPEntitiesMeanOnlyGeneralCnf(WPEntitiesMeanCnf):
    """Mean-pooled config keeping only general entities."""

    emb: EmbeddingCnf = field(default_factory=lambda: _wp_entities_mean_only_type_emb(keep_type='general'))


@dataclass(frozen=True)
class WPEntitiesMeanOnlyLocationCnf(WPEntitiesMeanCnf):
    """Mean-pooled config keeping only location entities."""

    emb: EmbeddingCnf = field(default_factory=lambda: _wp_entities_mean_only_type_emb(keep_type='location'))


@dataclass(frozen=True)
class WPEntitiesMeanOnlyOrganizationCnf(WPEntitiesMeanCnf):
    """Mean-pooled config keeping only organization entities."""

    emb: EmbeddingCnf = field(default_factory=lambda: _wp_entities_mean_only_type_emb(keep_type='organization'))


@dataclass(frozen=True)
class WPEntitiesMeanOnlyPersonCnf(WPEntitiesMeanCnf):
    """Mean-pooled config keeping only person entities."""

    emb: EmbeddingCnf = field(default_factory=lambda: _wp_entities_mean_only_type_emb(keep_type='person'))


@dataclass(frozen=True)
class WPEntitiesMeanOnlyProductCnf(WPEntitiesMeanCnf):
    """Mean-pooled config keeping only product entities."""

    emb: EmbeddingCnf = field(default_factory=lambda: _wp_entities_mean_only_type_emb(keep_type='product'))


@dataclass(frozen=True)
class WPEntitiesMeanOnlyOtherCnf(WPEntitiesMeanCnf):
    """Mean-pooled config keeping only other entities."""

    emb: EmbeddingCnf = field(default_factory=lambda: _wp_entities_mean_only_type_emb(keep_type='other'))


@dataclass(frozen=True)
class EntityOnlyCnf(BaseCnfWithHPO):
    """Entity-only configuration without article embeddings."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), use_article_embeddings=False)
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


@dataclass(frozen=True)
class WpEntitiesTunedCnf(PreBaseCnfWithHPO):
    """Best entity-enhanced config with per-class threshold tuning enabled.

    The dev folds are scanned over a 9-point grid (0.25..0.65 by 0.05) and
    per-class thresholds are aggregated by mean across folds, then reused
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
class WpEntitiesPmmTunedCnf(BaseCnfWithHPO):
    """Entity-enhanced config using PMM entity embeddings.

    The dev folds are scanned over a 9-point grid (0.25..0.65 by 0.05) and
    per-class thresholds are aggregated by mean across folds, then reused
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
    """PMM entity embeddings without article embeddings."""
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
class WPEntityOnlyMeanCnf(EntityOnlyCnf):
    """Entity-only configuration with mean entity pooling."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), use_article_embeddings=False, entity_pooling='mean')
    )


@dataclass(frozen=True)
class WpEntitiesPmmMentionWeightedMeanCnf(WPEntitiesMentionWeightedMeanCnf):
    """PMM entity embeddings with mention-weighted mean pooling."""

    paths: PathsCnf = field(default_factory=lambda: replace(WpEntitiesPmmTunedCnf().paths))


@dataclass(frozen=True)
class WpEntitiesPmmTunedSumCnf(WpEntitiesPmmTunedCnf):
    """PMM entity-enhanced configuration with sum feature fusion."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), combine_method='sum')
    )

