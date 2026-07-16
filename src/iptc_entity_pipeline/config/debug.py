"""Debug / smoke-test experiment configs."""

# standard library
from dataclasses import dataclass, field, replace

# project
from iptc_entity_pipeline.config.base import (
    DATA_ROOT,
    BaseCnf,
    EmbeddingCnf,
    HyperparamSpace,
    ModelCnf,
    PathsCnf,
    ThresholdTuningCnf,
)

@dataclass(frozen=True)
class DebugCnf(BaseCnf):
    """Debug configuration for quick local runs."""
    paths: PathsCnf = field(
        default_factory=lambda: PathsCnf(
            train_csv=f'{DATA_ROOT}/debug/all-corpora-train-entities.sample_4plus1.csv',
            test_csv=f'{DATA_ROOT}/debug/all-corpora-test-entities.sample_4plus1.csv',
            wdid_mapping_tsv=f'{DATA_ROOT}/wd-id_mapping_debug.tsv',
            article_embeddings_dir=f'{DATA_ROOT}/article_embeddings_debug',
            entity_embeddings_dir=f'{DATA_ROOT}/entity_embeddings/debug'
        )
    )
    model: ModelCnf = field(default_factory=lambda: replace(ModelCnf(), dropouts1=0.1))
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
            entity_pooling='weighted_mean_relevance',
        )
    )

    random_seed: int = 2
    tuning: ThresholdTuningCnf = field(
        default_factory=lambda: replace(ThresholdTuningCnf(), enabled=True)
    )


@dataclass(frozen=True)
class DebugEvalCnf(DebugCnf):
    """Debug configuration that loads a pre-trained model and skips CV/training."""
    model_path: str | None = 'models/debug_20260706_215424'


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


