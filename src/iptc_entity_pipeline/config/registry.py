"""Config name registry and lookup helpers."""

# standard library
from dataclasses import Field
from typing import Any

# project
from iptc_entity_pipeline.config.article_only import (
    ArticleOnlyCnf,
    ArticleOnlyGeluCnf,
    ArticleOnlyLeakyCnf,
    ArticleOnlyPriorCnf,
    ArticleOnlySkipCnf,
    ArticleOnlyTunedCnf,
    ArticleOnlyTunedDiffThresholdsCnf,
    BestArticleOnlyTunedCnf,
)
from iptc_entity_pipeline.config.base import BaseCnf
from iptc_entity_pipeline.config.debug import (
    DebugAttention3Cnf,
    DebugAttentionCnf,
    DebugCnf,
    DebugEvalCnf,
    DebugMhaAttention2Cnf,
    DebugMhaAttentionCnf,
)
from iptc_entity_pipeline.config.language import (
    BWpEntitiesTunedAllLangsCnf,
    BWpEntitiesTunedAllLangsFallbackAttentionCnf,
    BWpEntitiesTunedAllLangsFallbackCnf,
    BWpEntitiesTunedCsCnf,
    BWpEntitiesTunedDeCnf,
    BWpEntitiesTunedEnCnf,
    BWpEntitiesTunedEnDeCnf,
    BWpEntitiesTunedEnDeFallbackCnf,
    BWpEntitiesTunedEnFallbackCnf,
    BWpEntitiesTunedEnNlCnf,
    BWpEntitiesTunedEnNlFallbackCnf,
    BWpEntitiesTunedEsCnf,
    BWpEntitiesTunedFrCnf,
    BWpEntitiesTunedNlCnf,
    BWpEntitiesTunedNlEnFallbackCnf,
)
from iptc_entity_pipeline.config.sources import (
    W2VecAttentionHPOCnf,
    W2VecMentionWeightedMeanCnf,
    W2VecRelevanceWeightedMeanCnf,
    WArticleAtentionCnf,
    WikiIntroAttentionHPOCnf,
    WikiIntroMentionWeightedMeanCnf,
    WikiIntroRelevanceWeightedMeanCnf,
    WikidataDescriptionAttentionCnf,
    WikidataDescriptionEntitiesCnf,
    WikidataDescriptionEntitiesMeanCnf,
    WikidataDescriptionEntitiesMeanJinaAllLangsFallbackCnf,
    WikidataDescriptionEntitiesMeanJinaCnf,
    WikidataDescriptionEntitiesSumCnf,
    WikidataDescriptionEntityOnlyCnf,
    WikidataDescriptionEntityOnlyMeanCnf,
    WikidataDescriptionMentionWeightedMeanCnf,
    Wikipedia2VecEntitiesCnf,
    Wikipedia2VecEntitiesMeanCnf,
    Wikipedia2VecEntityOnlyCnf,
    Wikipedia2VecEntityOnlyMeanCnf,
    WikipediaArticleEntitiesCnf,
    WikipediaArticleEntitiesMeanCnf,
    WikipediaArticleEntitiesSumCnf,
    WikipediaArticleEntityOnlyCnf,
    WikipediaArticleEntityOnlyMeanCnf,
    WikipediaArticleMentionWeightedMeanCnf,
    WikipediaIntroEntitiesCnf,
    WikipediaIntroEntitiesMeanCnf,
    WikipediaIntroEntitiesSumCnf,
    WikipediaIntroEntityOnlyCnf,
    WikipediaIntroEntityOnlyMeanCnf,
)
from iptc_entity_pipeline.config.wpentities import (
    BestWpEntitiesAttentionCnf,
    BestWpEntitiesAttentionHPOCnf,
    BestWpEntitiesMhaAttention2Cnf,
    BestWpEntitiesMhaAttentionCnf,
    EntityOnlyCnf,
    WPEntitiesAttentionHPOCnf,
    WPEntitiesMeanCnf,
    WPEntitiesMeanNoLocationCnf,
    WPEntitiesMeanOnlyEventCnf,
    WPEntitiesMeanOnlyGeneralCnf,
    WPEntitiesMeanOnlyLocationCnf,
    WPEntitiesMeanOnlyOrganizationCnf,
    WPEntitiesMeanOnlyOtherCnf,
    WPEntitiesMeanOnlyPersonCnf,
    WPEntitiesMeanOnlyProductCnf,
    WPEntitiesMentionWeightedMeanCnf,
    WPEntitiesPmmAttentionHPOCnf,
    WPEntitiesRelevanceWeightedMeanCnf,
    WPEntitiesRelevanceWeightedSumCnf,
    WPEntityOnlyMeanCnf,
    WpEntitiesCnf,
    WpEntitiesGeluCnf,
    WpEntitiesLeakyCnf,
    WpEntitiesPmmMentionWeightedMeanCnf,
    WpEntitiesPmmTunedCnf,
    WpEntitiesPmmTunedOnlyCnf,
    WpEntitiesPmmTunedSumCnf,
    WpEntitiesPriorCnf,
    WpEntitiesSkipCnf,
    WpEntitiesTunedCnf,
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


def _validate_config_dataclass_decorators(
    *,
    package_prefix: str = 'iptc_entity_pipeline.config',
) -> None:
    """Fail fast when config subclasses are not declared as frozen dataclasses."""
    for cls in _iter_subclasses(BaseCnf):
        if not cls.__module__.startswith(package_prefix):
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


def _config_map() -> dict[str, BaseCnf]:
    """Return supported config instances."""
    return {
        # pre base testing
        'article_only': ArticleOnlyCnf(),
        'article_only_gelu': ArticleOnlyGeluCnf(),
        'article_only_skip': ArticleOnlySkipCnf(),
        'article_only_leaky': ArticleOnlyLeakyCnf(),
        'article_only_prior': ArticleOnlyPriorCnf(),
        'article_only_tuned': ArticleOnlyTunedCnf(),
        'best_article_only_tuned': BestArticleOnlyTunedCnf(),
        
        # pre base test with entities(doeas it change anything... no)
        'wpentities': WpEntitiesCnf(),
        'wpentities_gelu': WpEntitiesGeluCnf(),
        'wpentities_skip': WpEntitiesSkipCnf(),
        'wpentities_leaky': WpEntitiesLeakyCnf(),
        'wpentities_prior': WpEntitiesPriorCnf(),
        'wpentities_tuned': WpEntitiesTunedCnf(),
        
        # entity-only (no article)
        'wp_entity_only': EntityOnlyCnf(),
        'wikipedia2vec_entity_only': Wikipedia2VecEntityOnlyCnf(),
        'wikidata_description_entity_only': WikidataDescriptionEntityOnlyCnf(),
        'wikipedia_intro_entity_only': WikipediaIntroEntityOnlyCnf(),
        'wikipedia_article_entity_only': WikipediaArticleEntityOnlyCnf(),
        'wpentities_pmm_tuned_only': WpEntitiesPmmTunedOnlyCnf(),

        # article + entities (default aggregation) - mean
        'wpentities_pmm': WpEntitiesPmmTunedCnf(),
        'wikipedia2vec_entities': Wikipedia2VecEntitiesCnf(),
        'wikidata_description_entities': WikidataDescriptionEntitiesCnf(),
        'wikipedia_intro_entities': WikipediaIntroEntitiesCnf(),
        'wikipedia_article_entities': WikipediaArticleEntitiesCnf(),

        # entity-only mean
        'wp_entity_only_mean': WPEntityOnlyMeanCnf(),
        'wikipedia2vec_entity_only_mean': Wikipedia2VecEntityOnlyMeanCnf(),
        'wikidata_description_entity_only_mean': WikidataDescriptionEntityOnlyMeanCnf(),
        'wikipedia_intro_entity_only_mean': WikipediaIntroEntityOnlyMeanCnf(),
        'wikipedia_article_entity_only_mean': WikipediaArticleEntityOnlyMeanCnf(),

        # entities mean
        'wp_entities_mean': WPEntitiesMeanCnf(),
        'wikipedia2vec_entities_mean': Wikipedia2VecEntitiesMeanCnf(),
        'wikidata_description_entities_mean': WikidataDescriptionEntitiesMeanCnf(),
        'wikipedia_intro_entities_mean': WikipediaIntroEntitiesMeanCnf(),
        'wikipedia_article_entities_mean': WikipediaArticleEntitiesMeanCnf(),

        # entities sum
        'wpentities_pmm_sum': WpEntitiesPmmTunedSumCnf(),
        'wikidata_description_entities_sum': WikidataDescriptionEntitiesSumCnf(),
        'wikipedia_intro_entities_sum': WikipediaIntroEntitiesSumCnf(),
        'wikipedia_article_entities_sum': WikipediaArticleEntitiesSumCnf(),

        # mention-weighted mean
        'wpentities_mention_weighted_mean': WPEntitiesMentionWeightedMeanCnf(),
        'wpentities_pmm_mention_weighted_mean': WpEntitiesPmmMentionWeightedMeanCnf(),
        'wikipedia2vec_mention_weighted_mean': W2VecMentionWeightedMeanCnf(),
        'wikidata_description_mention_weighted_mean': WikidataDescriptionMentionWeightedMeanCnf(),
        'wikipedia_intro_mention_weighted_mean': WikiIntroMentionWeightedMeanCnf(),
        'wikipedia_article_mention_weighted_mean': WikipediaArticleMentionWeightedMeanCnf(),

        # relevance / weighted aggregation
        'wpentities_relevance_weighted_sum': WPEntitiesRelevanceWeightedSumCnf(),
        'wpentities_relevance_weighted_mean': WPEntitiesRelevanceWeightedMeanCnf(),
        'w2vec_relevance_weighted_mean': W2VecRelevanceWeightedMeanCnf(),
        'wikipedia_intro_relevance_weighted_mean': WikiIntroRelevanceWeightedMeanCnf(),

        # attention aggregation
        'debug_attention': DebugAttentionCnf(),
        'debug_attention3': DebugAttention3Cnf(),
        'debug_mha_attention': DebugMhaAttentionCnf(),
        'debug_mha_attention2': DebugMhaAttention2Cnf(),
        'wpentities_attention_hpo': WPEntitiesAttentionHPOCnf(),
        'best_wpentities_attention_hpo': BestWpEntitiesAttentionHPOCnf(),
        'wpentities_pmm_attention': WPEntitiesPmmAttentionHPOCnf(),
        'w2vec_attention': W2VecAttentionHPOCnf(),
        'wikidata_description_attention': WikidataDescriptionAttentionCnf(),
        'wikipedia_intro_attention': WikiIntroAttentionHPOCnf(),
        'wikipedia_article_entities_attention': WArticleAtentionCnf(),

        # entity-type ablations (mean)
        'wp_entities_mean_no_location': WPEntitiesMeanNoLocationCnf(),
        'wp_entities_mean_only_event': WPEntitiesMeanOnlyEventCnf(),
        'wp_entities_mean_only_general': WPEntitiesMeanOnlyGeneralCnf(),
        'wp_entities_mean_only_location': WPEntitiesMeanOnlyLocationCnf(),
        'wp_entities_mean_only_organization': WPEntitiesMeanOnlyOrganizationCnf(),
        'wp_entities_mean_only_person': WPEntitiesMeanOnlyPersonCnf(),
        'wp_entities_mean_only_product': WPEntitiesMeanOnlyProductCnf(),
        'wp_entities_mean_only_other': WPEntitiesMeanOnlyOtherCnf(),

        # jina embeddings
        'wikidata_description_jina': WikidataDescriptionEntitiesMeanJinaCnf(),
        'wikidata_description_jina_all_langs_fallback': WikidataDescriptionEntitiesMeanJinaAllLangsFallbackCnf(),

        # threshold variants
        'article_only_tuned_diff_thresholds': ArticleOnlyTunedDiffThresholdsCnf(),
        # language fallback runs
        'wpentities_tuned_en_fallback': BWpEntitiesTunedEnFallbackCnf(),
        'wpentities_tuned_en_de_fallback': BWpEntitiesTunedEnDeFallbackCnf(),
        'wpentities_tuned_en_nl_fallback': BWpEntitiesTunedEnNlFallbackCnf(),
        'wpentities_tuned_nl_en_fallback': BWpEntitiesTunedNlEnFallbackCnf(),
        'wpentities_tuned_all_langs_fallback': BWpEntitiesTunedAllLangsFallbackCnf(),
        
        # language tests with averaging
        'wpentities_tuned_cs': BWpEntitiesTunedCsCnf(),
        'wpentities_tuned_de': BWpEntitiesTunedDeCnf(),
        'wpentities_tuned_nl': BWpEntitiesTunedNlCnf(),
        'wpentities_tuned_fr': BWpEntitiesTunedFrCnf(),
        'wpentities_tuned_es': BWpEntitiesTunedEsCnf(),
        'wpentities_tuned_en': BWpEntitiesTunedEnCnf(),
        'wpentities_tuned_en_de': BWpEntitiesTunedEnDeCnf(),
        'wpentities_tuned_en_nl': BWpEntitiesTunedEnNlCnf(),
        'wpentities_tuned_all_langs': BWpEntitiesTunedAllLangsCnf(),
        
        # debug runs
        'debug': DebugCnf(),
        'debug_eval': DebugEvalCnf(),
         
        # attention runs
        'wpentities_attention': BestWpEntitiesAttentionCnf(),
        'wpentities_mha_attention': BestWpEntitiesMhaAttentionCnf(),
        'wpentities_mha_attention2': BestWpEntitiesMhaAttention2Cnf(),
        
        # best
        'wpentities_all_langs_fallback_attention': BWpEntitiesTunedAllLangsFallbackAttentionCnf(),
    }


def get_config(config_name: str) -> BaseCnf:
    """
    Return config variant by name.

    Configs are resolved from :func:`_config_map`.

    :param config_name: Config variant name.
    :return: Selected config object.
    :raises ValueError: If ``config_name`` is unknown.
    """
    name = config_name.strip().lower()
    config_map = _config_map()
    if name in config_map:
        return config_map[name]
    raise ValueError(f'Unsupported config_name: {config_name}')


def list_config_names() -> tuple[str, ...]:
    """
    Return names of supported config variants.

    :return: Tuple of supported config names.
    """
    return tuple(_config_map().keys())


_validate_config_dataclass_decorators()
PipelineCnf = BaseCnf


