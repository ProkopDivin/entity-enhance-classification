"""Tests for config dataclasses and helpers."""
from __future__ import annotations

import pytest

from iptc_entity_pipeline.config import (
    ALL_ENTITY_LANGS,
    ArticleOnlyCnf,
    BaseCnf,
    BestArticleOnlyCnf,
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
    WPEntitiesAllLangsCnf,
    WPEntitiesRelTH5,
    WPEntitiesMentionWeightedMeanCnf,
    WPEntitiesMentionWeightedSumCnf,
    WPEntitiesNlCnf,
    WPEntitiesRelevanceWeightedMeanCnf,
    WPEntitiesRelevanceWeightedSumCnf,
    WPEntitiesWeightedMeanCnf,
    WPEntitiesEnNlCnf,
    Wikipedia2VecEntitiesAllLangsCnf,
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
    ('wpentities_weighted_mean', WPEntitiesWeightedMeanCnf),
    ('wpentities_relevance_weighted_sum', WPEntitiesRelevanceWeightedSumCnf),
    ('wpentities_mention_weighted_sum', WPEntitiesMentionWeightedSumCnf),
    ('wpentities_relevance_weighted_mean', WPEntitiesRelevanceWeightedMeanCnf),
    ('wpentities_mention_weighted_mean', WPEntitiesMentionWeightedMeanCnf),
    ('article_only', ArticleOnlyCnf),
    ('entity_only', EntityOnlyCnf),
    ('wpentities_en_nl', WPEntitiesEnNlCnf),
    ('wpentities_nl', WPEntitiesNlCnf),
    ('wpentities_all_langs', WPEntitiesAllLangsCnf),
    ('wpentities_rel_th_5', WPEntitiesRelTH5),
    ('best_wpentities_tuned', BestWpEntitiesTunedCnf),
    ('best_article_only', BestArticleOnlyCnf),
    ('article_only_skip', ArticleOnlySkipCnf),
    ('article_only_leaky', ArticleOnlyLeakyCnf),
    ('wpentities_skip', WpEntitiesSkipCnf),
    ('wpentities_leaky', WpEntitiesLeakyCnf),
    ('wikipedia2vec_entities_all_langs', Wikipedia2VecEntitiesAllLangsCnf),
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
        'wpentities_weighted_mean',
        'wpentities_relevance_weighted_sum',
        'wpentities_mention_weighted_sum',
        'article_only',
        'entity_only',
        'wpentities_en_nl',
        'wpentities_nl',
        'wpentities_all_langs',
        'wpentities_rel_th_5',
        'best_wpentities_tuned',
        'best_article_only',
        'article_only_skip',
        'article_only_leaky',
        'wpentities_skip',
        'wpentities_leaky',
        'wikipedia2vec_entities_all_langs',
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
    assert WPEntitiesWeightedMeanCnf().emb.entity_pooling == 'weighted_mean_relevance'
    assert WPEntitiesRelevanceWeightedMeanCnf().emb.entity_pooling == 'weighted_mean_relevance'
    assert WPEntitiesMentionWeightedMeanCnf().emb.entity_pooling == 'weighted_mean'
    assert WPEntitiesRelevanceWeightedSumCnf().emb.entity_pooling == 'weighted_sum_relevance'
    assert WPEntitiesMentionWeightedSumCnf().emb.entity_pooling == 'weighted_sum'
    assert DebugCnf().train.epochs == TrainingCnf().epochs
    assert WpEntitiesCnf().hparam.learning_rates == (0.00037,)
    assert ArticleOnlyCnf().hparam.learning_rates == (0.00037,)
    assert BaseCnf().model.bias_from_prior is False
    assert ArticleOnlyPriorCnf().model.bias_from_prior is True
    assert WpEntitiesPriorCnf().model.bias_from_prior is True
    assert ArticleOnlyPriorCnf().model.nn_type == 'mlp'
    assert WpEntitiesPriorCnf().model.nn_type == 'mlp'
    from iptc_entity_pipeline.config import BestWpEntitiesAttention2Cnf, WPEntitiesAttention2HPOCnf

    assert BestWpEntitiesAttention2Cnf().model.nn_type == 'entity_attention2_mlp'
    assert BestWpEntitiesAttention2Cnf().emb.entity_pooling == 'no_pooling'
    assert WPEntitiesAttention2HPOCnf().model.nn_type == 'entity_attention2_mlp'
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


@pytest.mark.parametrize('name,expected_threshold', [
    ('wpentities_rel_th_5', 5.0),
])
def test_relevance_threshold_configs_map_to_embedding_threshold(name: str, expected_threshold: float) -> None:
    cfg = get_config(name)
    assert cfg.emb.entity_relevance_threshold == expected_threshold


@pytest.mark.parametrize('name', [
    'wpentities_rel_th_5',
])
def test_relevance_threshold_configs_keep_single_language_tuple(name: str) -> None:
    cfg = get_config(name)
    assert cfg.emb.entity_langs == ('en',)


def test_wikidata_description_entities_config_uses_description_embedding_dir() -> None:
    cfg = get_config('wikidata_description_entities')
    assert cfg.paths.entity_embeddings_dir.endswith('/data/entity_embeddings/WikidataDescription')


def test_wikipedia2vec_entities_all_langs_config_uses_wikipedia2vec_dir_and_all_langs() -> None:
    cfg = get_config('wikipedia2vec_entities_all_langs')
    assert cfg.paths.entity_embeddings_dir.endswith('/data/entity_embeddings/wikipedia2vec')
    assert cfg.emb.entity_langs == ALL_ENTITY_LANGS
