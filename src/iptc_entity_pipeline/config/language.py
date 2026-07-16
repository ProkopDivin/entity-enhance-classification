"""Language and language-fallback experiment configs."""

# standard library
from dataclasses import dataclass, field, replace

# project
from iptc_entity_pipeline.config.base import (
    ALL_ENTITY_LANGS,
    BaseCnfWithHPO,
    EmbeddingCnf,
    HyperparamSpace,
    ModelCnf,
)
from iptc_entity_pipeline.config.wpentities import BestWpEntitiesTunedCnf

@dataclass(frozen=True)
class BWpEntitiesTunedAllLangsCnf(BestWpEntitiesTunedCnf):
    """Best entity-enhanced config with all supported entity languages."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_langs=ALL_ENTITY_LANGS)
    )


@dataclass(frozen=True)
class BWpEntitiesTunedCsCnf(BestWpEntitiesTunedCnf):
    """Best entity-enhanced config with Czech entity embeddings."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_langs=('cs',))
    )


@dataclass(frozen=True)
class BWpEntitiesTunedDeCnf(BestWpEntitiesTunedCnf):
    """Best entity-enhanced config with German entity embeddings."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_langs=('de',))
    )


@dataclass(frozen=True)
class BWpEntitiesTunedNlCnf(BestWpEntitiesTunedCnf):
    """Best entity-enhanced config with Dutch entity embeddings."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_langs=('nl',))
    )


@dataclass(frozen=True)
class BWpEntitiesTunedFrCnf(BestWpEntitiesTunedCnf):
    """Best entity-enhanced config with French entity embeddings."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_langs=('fr',))
    )


@dataclass(frozen=True)
class BWpEntitiesTunedEsCnf(BestWpEntitiesTunedCnf):
    """Best entity-enhanced config with Spanish entity embeddings."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_langs=('es',))
    )


@dataclass(frozen=True)
class BWpEntitiesTunedEnCnf(BestWpEntitiesTunedCnf):
    """Best entity-enhanced config with English entity embeddings."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_langs=('en',))
    )


@dataclass(frozen=True)
class BWpEntitiesTunedEnDeCnf(BestWpEntitiesTunedCnf):
    """Best entity-enhanced config with German and English entity embeddings."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_langs=('de','en'))
    )


@dataclass(frozen=True)
class BWpEntitiesTunedEnNlCnf(BestWpEntitiesTunedCnf):
    """Best entity-enhanced config with Dutch and English entity embeddings."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_langs=('nl','en'))
    )


@dataclass(frozen=True)
class BWpEntitiesTunedAllLangsFallbackCnf(BaseCnfWithHPO):
    """Best entity-enhanced config with all languages in fallback mode."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_langs=ALL_ENTITY_LANGS, entity_lang_mode='fallback')
    )


@dataclass(frozen=True)
class BWpEntitiesTunedAllLangsFallbackAttentionCnf(BaseCnfWithHPO):
    """Best entity-enhanced config with all languages in fallback mode and attention pooling."""
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
            attention_hidden_dims=(64, 128, 256, 512),
            attention_dropouts=(0.0, 0.3,),

        )
    )


@dataclass(frozen=True)
class BWpEntitiesTunedEnFallbackCnf(BaseCnfWithHPO):
    """Best entity-enhanced config with English in fallback mode."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_langs=('en',), entity_lang_mode='fallback')
    )


@dataclass(frozen=True)
class BWpEntitiesTunedEnDeFallbackCnf(BaseCnfWithHPO):
    """Best entity-enhanced config with English and German in fallback mode."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_langs=('en', 'de'), entity_lang_mode='fallback')
    )


@dataclass(frozen=True)
class BWpEntitiesTunedNlEnFallbackCnf(BaseCnfWithHPO):
    """Best entity-enhanced config with Dutch and English in fallback mode."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_langs=('nl','en'), entity_lang_mode='fallback')
    )


@dataclass(frozen=True)
class BWpEntitiesTunedEnNlFallbackCnf(BaseCnfWithHPO):
    """Best entity-enhanced config with English and Dutch in fallback mode."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_langs=('en','nl'), entity_lang_mode='fallback')
    )


