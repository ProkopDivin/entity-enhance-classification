"""Alternate entity-embedding source experiment configs."""

# standard library
from dataclasses import dataclass, field, replace

# project
from iptc_entity_pipeline.config.base import (
    ALL_ENTITY_LANGS,
    DATA_ROOT,
    BaseCnfWithHPO,
    EmbeddingCnf,
    HyperparamSpace,
    ModelCnf,
    PathsCnf,
)
from iptc_entity_pipeline.config.wpentities import (
    WPEntitiesMentionWeightedMeanCnf,
    WPEntitiesRelevanceWeightedMeanCnf,
    WpEntitiesTunedCnf,
)

@dataclass(frozen=True)
class WikiIntroAttentionHPOCnf(WpEntitiesTunedCnf):
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
class WikipediaArticleEntitiesCnf(BaseCnfWithHPO):
    """Entity-enhanced configuration using full Wikipedia article embeddings."""

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
            entity_embeddings_dir=f'{DATA_ROOT}/entity_embeddings/wikipedia2vec',
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
    """Entity-enhanced configuration using Wikipedia intro text embeddings."""

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
class WikidataDescriptionMentionWeightedMeanCnf(WPEntitiesMentionWeightedMeanCnf):
    """Wikidata description embeddings with mention-weighted mean pooling."""

    paths: PathsCnf = field(default_factory=lambda: replace(WikidataDescriptionEntitiesCnf().paths))


@dataclass(frozen=True)
class WikipediaArticleMentionWeightedMeanCnf(WPEntitiesMentionWeightedMeanCnf):
    """Wikipedia article embeddings with mention-weighted mean pooling."""

    paths: PathsCnf = field(default_factory=lambda: replace(WikipediaArticleEntitiesCnf().paths))


@dataclass(frozen=True)
class WikipediaArticleEntitiesMeanCnf(WikipediaArticleEntitiesCnf):
    """Wikipedia article entity-enhanced configuration with mean entity pooling."""

    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), entity_pooling='mean')
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


