"""Tests for config dataclasses and helpers."""
from __future__ import annotations

import pytest

from iptc_entity_pipeline.config import (
    ArticleOnlyCnf,
    BaseCnf,
    BestArticleOnlyCnf,
    BestArticleOnlyTunedCnf,
    ArticleOnlyGeluCnf,
    ArticleOnlyLeakyCnf,
    ArticleOnlyPriorCnf,
    ArticleOnlySkipCnf,
    BestWpEntitiesTunedCnf,
    DebugCnf,
    EmbeddingCnf,
    EntityOnlyCnf,
    HyperparamSpace,
    ModelCnf,
    PathsCnf,
    TrainingCnf,
    WPEntitiesMentionWeightedMeanCnf,
    WPEntitiesRelevanceWeightedMeanCnf,
    WPEntitiesRelevanceWeightedSumCnf,
    WikidataDescriptionEntitiesCnf,
    WpEntitiesCnf,
    WpEntitiesGeluCnf,
    WpEntitiesLeakyCnf,
    WpEntitiesPriorCnf,
    WpEntitiesSkipCnf,
    conf_from_dict,
    get_config,
    list_config_names,
    resolve_paths,
)


def test_config_from_dict_filters_unknown_keys() -> None:
    cfg = conf_from_dict(
        ModelCnf,
        {'hidden_dim': 512, 'bias_from_prior': False, 'unknown': 'ignored'},
    )
    assert cfg.hidden_dim == 512
    assert cfg.dropouts1 == 0.0
    assert cfg.bias_from_prior is False


@pytest.mark.parametrize('name,expected_cls', [
    ('debug', DebugCnf),
    ('wpentities', WpEntitiesCnf),
    ('wpentities_relevance_weighted_sum', WPEntitiesRelevanceWeightedSumCnf),
    ('wpentities_relevance_weighted_mean', WPEntitiesRelevanceWeightedMeanCnf),
    ('wpentities_mention_weighted_mean', WPEntitiesMentionWeightedMeanCnf),
    ('article_only', ArticleOnlyCnf),
    ('wp_entity_only', EntityOnlyCnf),
    ('best_article_only_tuned', BestArticleOnlyTunedCnf),
    ('article_only_skip', ArticleOnlySkipCnf),
    ('article_only_leaky', ArticleOnlyLeakyCnf),
    ('wpentities_skip', WpEntitiesSkipCnf),
    ('wpentities_leaky', WpEntitiesLeakyCnf),
    ('wikidata_description_entities', WikidataDescriptionEntitiesCnf),
    ('  Debug  ', DebugCnf),
])
def test_get_config_valid(name: str, expected_cls: type) -> None:
    assert isinstance(get_config(name), expected_cls)


def test_get_config_invalid() -> None:
    with pytest.raises(ValueError, match='Unsupported'):
        get_config('nonexistent')


def test_list_config_names() -> None:
    assert {
        'debug',
        'wpentities',
        'wpentities_relevance_weighted_sum',
        'article_only',
        'wp_entity_only',
        'best_article_only_tuned',
        'article_only_skip',
        'article_only_leaky',
        'wpentities_skip',
        'wpentities_leaky',
        'wikidata_description_entities',
    }.issubset(set(list_config_names()))


def test_resolve_paths_prepends_root() -> None:
    cfg = BaseCnf(paths=PathsCnf(
        train_csv='data/train.csv',
        test_csv='data/test.csv',
        wdid_mapping_tsv='data/map.tsv',
        article_embeddings_dir='data/emb',
        entity_embeddings_dir='data/ent',
        downsampling_order_cache_json='data/cache.json',
    ))
    resolved = resolve_paths(cfg, '/root')
    assert resolved.paths.train_csv == '/root/data/train.csv'
    assert resolved.paths.entity_embeddings_dir == '/root/data/ent'
    assert resolved.paths.removed_cat_ids == cfg.paths.removed_cat_ids


def test_config_variants() -> None:
    assert ArticleOnlyCnf().emb.use_entity_embeddings is False
    assert EntityOnlyCnf().emb.use_article_embeddings is False
    assert BestArticleOnlyCnf().emb.use_entity_embeddings is False
    assert ArticleOnlyGeluCnf().model.nn_type == 'mlp_gelu'
    assert WpEntitiesGeluCnf().model.nn_type == 'mlp_gelu'
    assert ArticleOnlySkipCnf().model.nn_type == 'skip_mlp'
    assert ArticleOnlyLeakyCnf().model.nn_type == 'leaky_mlp'
    assert WpEntitiesSkipCnf().model.nn_type == 'skip_mlp'
    assert WpEntitiesLeakyCnf().model.nn_type == 'leaky_mlp'
    assert WPEntitiesRelevanceWeightedMeanCnf().emb.entity_pooling == 'weighted_mean_relevance'
    assert WPEntitiesMentionWeightedMeanCnf().emb.entity_pooling == 'weighted_mean'
    assert WPEntitiesRelevanceWeightedSumCnf().emb.entity_pooling == 'weighted_sum_relevance'
    assert DebugCnf().train.epochs == TrainingCnf().epochs
    assert WpEntitiesCnf().hparam.learning_rates == (0.00037,)
    assert ArticleOnlyCnf().hparam.learning_rates == (0.00037,)
    assert BestWpEntitiesTunedCnf().hparam.hidden_dims == (4096,)
    assert BaseCnf().model.bias_from_prior is False
    assert ArticleOnlyPriorCnf().model.bias_from_prior is True
    assert WpEntitiesPriorCnf().model.bias_from_prior is True
    assert ArticleOnlyPriorCnf().model.nn_type == 'mlp'
    assert WpEntitiesPriorCnf().model.nn_type == 'mlp'
    d = BaseCnf().to_clearml_mapping()
    assert isinstance(d, dict) and 'paths' in d and 'model' in d


def test_embedding_config_supports_multi_language_entities() -> None:
    cfg = EmbeddingCnf(entity_lang='en', entity_langs=('en', 'de', 'cs'))
    assert cfg.entity_lang == 'en'
    assert cfg.entity_langs == ('en', 'de', 'cs')


def test_embedding_config_defaults_to_single_language_fallback() -> None:
    cfg = EmbeddingCnf()
    assert cfg.entity_lang == 'en'
    assert cfg.entity_langs == ()
    assert cfg.entity_lang_mode == 'average'
    assert cfg.entity_relevance_threshold == 0.0
    assert cfg.use_article_embeddings is True
    assert cfg.use_entity_embeddings is True
    assert cfg.entity_pooling == 'mean'


def test_hyperparam_space_default_types_are_scalar_sequences() -> None:
    space = HyperparamSpace()
    assert all(isinstance(value, int) for value in space.hidden_dims)
    assert all(isinstance(value, float) for value in space.dropouts2)


def test_wikidata_description_entities_config_uses_description_embedding_dir() -> None:
    cfg = get_config('wikidata_description_entities')
    assert cfg.paths.entity_embeddings_dir.endswith('/data/entity_embeddings/WikidataDescription')
