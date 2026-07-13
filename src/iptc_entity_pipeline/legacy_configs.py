"""Legacy experiment config variants not in the active config registry."""

from dataclasses import dataclass, field, replace
from typing import Any

from iptc_entity_pipeline.config import (
    ALL_ENTITY_LANGS,
    ArticleOnlyCnf,
    ArticleOnlyGeluCnf,
    ArticleOnlyLeakyCnf,
    ArticleOnlyPriorCnf,
    ArticleOnlySkipCnf,
    ArticleOnlyTunedCnf,
    AssemblyCnf,
    AssemblyMemberCnf,
    BaseCnf,
    BaseCnfWithHPO,
    BaseCnfWithHPO2,
    BaseCnfWithHPO3,
    BestArticleOnlyCnf,
    BestWpEntitiesAttentionCnf,
    BestWpEntitiesTunedCnf,
    BWpEntitiesTunedEnDeFallbackCnf,
    BWpEntitiesTunedEnFallbackCnf,
    BWpEntitiesTunedEnNlFallbackCnf,
    DATA_ROOT,
    DebugCnf,
    EmbeddingCnf,
    EntityOnlyCnf,
    GOLD_ORIGIN_TEST_CSV,
    GOLD_ORIGIN_TRAIN_CSV,
    HyperparamSpace,
    ModelCnf,
    PathsCnf,
    PreBaseCnfWithHPO,
    ThresholdTuningCnf,
    TrainingCnf,
    Wikipedia2VecEntitiesCnf,
    Wikipedia2VecEntityOnlyCnf,
    WikidataDescriptionEntitiesCnf,
    WikidataDescriptionEntityOnlyCnf,
    WikipediaArticleEntitiesCnf,
    WikipediaArticleEntityOnlyCnf,
    WikipediaIntroEntitiesCnf,
    WikipediaIntroEntityOnlyCnf,
    WPEntitiesAttentionHPOCnf,
    WPEntitiesPmmAttentionHPOCnf,
    WPEntitiesMeanCnf,
    WPEntitiesMeanNoLocationCnf,
    WPEntitiesMentionWeightedMeanCnf,
    WPEntitiesMentionWeightedSumCnf,
    WPEntitiesRelevanceWeightedMeanCnf,
    WPEntitiesRelevanceWeightedSumCnf,
    WPEntitiesWeightedMeanCnf,
    WPEntityOnlyMeanCnf,
    WpEntitiesCnf,
    WpEntitiesGeluCnf,
    WpEntitiesLeakyCnf,
    WpEntitiesPmmTunedCnf,
    WpEntitiesPmmTunedOnlyCnf,
    WpEntitiesPmmTunedSumCnf,
    WpEntitiesPriorCnf,
    WpEntitiesSkipCnf,
    WpEntitiesTunedCnf,
    Wikipedia2VecEntitiesMeanCnf,
    WikidataDescriptionEntitiesMeanCnf,
    WikipediaIntroEntitiesMeanCnf,
    WikipediaArticleEntitiesMeanCnf,
    Wikipedia2VecEntityOnlyMeanCnf,
    WikidataDescriptionEntityOnlyMeanCnf,
    WikipediaIntroEntityOnlyMeanCnf,
    WikipediaArticleEntityOnlyMeanCnf,
    WikidataDescriptionEntitiesSumCnf,
    WikipediaIntroEntitiesSumCnf,
    WikipediaArticleEntitiesSumCnf,
)


def _gold_origin_paths(**overrides: Any) -> PathsCnf:
    """Build :class:`PathsCnf` pointing at gold-origin train/test entity CSVs."""
    return replace(
        PathsCnf(),
        train_csv=GOLD_ORIGIN_TRAIN_CSV,
        test_csv=GOLD_ORIGIN_TEST_CSV,
        **overrides,
    )


@dataclass(frozen=True)
class NoEmbeddingsCnf(BaseCnf):
    """No embeddings configuration."""
    emb: EmbeddingCnf = field(
        default_factory=lambda: replace(EmbeddingCnf(), use_article_embeddings=False, use_entity_embeddings=False)
    )


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
class Assembly1Cnf(BaseCnf):
    """Dual-model assembly demo using two tuned single-model configs.

    Each member directly carries a full pipeline config instance. The
    ``thresholds_path`` files come from prior single-model runs in
    ``results/saved_models`` and drive both per-fold evaluation and the
    final stitched per-class thresholds.

    Each member runs the full configured k-fold ``run_cv`` loop.
    """
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


@dataclass(frozen=True)
class TunningLearningRateCnf(BaseCnfWithHPO3):
    """  """
@dataclass(frozen=True)
class TunningLearningRateF1Cnf(BaseCnfWithHPO3):
    """  """
    train: TrainingCnf = field(default_factory=lambda: replace(TrainingCnf(), early_stopping_metric='f1'))
    
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
@dataclass(frozen=True)
class GoldOriginWPEntitiesMeanCnf(WPEntitiesMeanCnf):
    """Mean-pooled entity-enhanced config on gold-origin corpora."""

    paths: PathsCnf = field(default_factory=lambda: _gold_origin_paths_from(WPEntitiesMeanCnf))
@dataclass(frozen=True)
class GoldOriginWPEntitiesMeanNoLocationCnf(WPEntitiesMeanNoLocationCnf):
    """Mean-pooled entity-enhanced config on gold-origin corpora without location entities."""

    paths: PathsCnf = field(default_factory=lambda: _gold_origin_paths_from(WPEntitiesMeanNoLocationCnf))
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

def _legacy_config_map() -> dict[str, BaseCnf]:
    """Return legacy experiment config instances."""
    return {
        'best_wpentities_f1': BestWpEntitiesTunedF1Cnf(),
        'best_wpentities_tuned': BestWpEntitiesTunedCnf(),
        'best_article_only': BestArticleOnlyCnf(),
        'article_only_tuned_f1': BestArticleOnlyTunedF1Cnf(),
        'best_article_only_tuned_f1': BestArticleOnlyTunedF1Cnf(),
        'best_article_only_tuned_2': BestArticleOnlyTunedCnf2(),
        'best_article_only_tuned_3': BestArticleOnlyTunedCnf3(),
        'best_article_only_tuned_4': BestArticleOnlyTunedCnf4(),
        'best_article_only_tuned_5': BestArticleOnlyTunedCnf5(),
        'best_wpentities_all_langs': BestWpentitiesAllLangsCnf(),
        'best_wpentities_nl': BestWpentitiesNlCnf(),
        'best_wpentities_en_nl': BestWPEntitiesENNLCnf(),
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
        'gold_origin_wp_entities_mean_no_location': GoldOriginWPEntitiesMeanNoLocationCnf(),
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
        'gold_origin_wpentities_mean_no_location': GoldOriginWPEntitiesMeanNoLocationCnf(),
        'gold_origin_wpentities_pmm_attention': GoldOriginWPEntitiesPmmAttentionHPOCnf(),
        'gold_origin_wpentities_tuned_en_fallback': GoldOriginBWpEntitiesTunedEnFallbackCnf(),
        'gold_origin_wpentities_tuned_en_de_fallback': GoldOriginBWpEntitiesTunedEnDeFallbackCnf(),
        'gold_origin_wpentities_tuned_en_nl_fallback': GoldOriginBWpEntitiesTunedEnNlFallbackCnf(),
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
        'wpentities_rel_th_1': WPEntitiesRelTH1(),
        'wpentities_rel_th_2': WPEntitiesRelTH2(),
        'wpentities_rel_th_3': WPEntitiesRelTH3(),
        'wpentities_rel_th_4': WPEntitiesRelTH4(),
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
        'wpentities_jina_v3_cls': WpEntitiesJV3ClsTunedCnf(),
        'wpentities_jina_v5_cls': WpEntitiesJV5ClsTunedCnf(),
        'wpentities_2': WpEntitiesCnf2(),
        'wpentities_3': WpEntitiesCnf3(),
        'wpentities_4': WpEntitiesCnf4(),
        'wpentities_5': WpEntitiesCnf5(),
        'article_only_2': ArticleOnlyCnf2(),
        'article_only_3': ArticleOnlyCnf3(),
        'article_only_4': ArticleOnlyCnf4(),
        'article_only_5': ArticleOnlyCnf5(),
        'learning_rate': TunningLearningRateCnf(),
        'learning_rate_f1': TunningLearningRateF1Cnf(),
        'assembly1': Assembly1Cnf(),
        'assembly_debug': AssemblyDebug(),
        'assembly2': Assembly2Cnf(),
        'assembly_attention': Assembly3Cnf(),
        'entity_only': EntityOnlyCnf(),
        'no_embeddings': NoEmbeddingsCnf(),
    }


def get_legacy_config(config_name: str) -> BaseCnf:
    """
    Return legacy config variant by name.

    :param config_name: Legacy config variant name.
    :return: Selected config object.
    :raises ValueError: If ``config_name`` is unknown.
    """
    name = config_name.strip().lower()
    config_map = _legacy_config_map()
    if name not in config_map:
        raise ValueError(f'Unsupported legacy config_name: {config_name}')
    return config_map[name]


def list_legacy_config_names() -> tuple[str, ...]:
    """
    Return names of legacy config variants.

    :return: Tuple of legacy config names.
    """
    return tuple(_legacy_config_map().keys())


from iptc_entity_pipeline.config import _validate_config_dataclass_decorators

_validate_config_dataclass_decorators(__name__)
