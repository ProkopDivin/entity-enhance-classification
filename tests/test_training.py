"""Tests for training payload builders."""
from __future__ import annotations

import json
import numpy as np
import pytest
import torch

from geneea.catlib.model.nnet import EntityAttention2MLP, EntityAttentionMLP, LeakyMLP, MLP, NeuralCategModel, SkipMLP
from iptc_entity_pipeline.config import ModelCnf, TrainingCnf
from iptc_entity_pipeline.legacy_reuse import _resolve_nn_type, wgh_labels_from_score_matrix
from iptc_entity_pipeline.training import _class_prior_logits, combo_params_json, model_payload, train_payload


def test_payload_builders() -> None:
    mc = ModelCnf(hidden_dim=256, dropouts1=0.1, dropouts2=0.3)
    tc = TrainingCnf(epochs=10, batch_size=64, learning_rate=0.001, early_stopping_metric='f1')

    params = json.loads(combo_params_json(model_config=mc, training_config=tc))
    assert params['hidden_dim'] == 256
    assert params['batch_size'] == 64
    assert params['learning_rate'] == 0.001

    mp = model_payload(model_config=mc)
    assert mp['hiddenDim'] == 256
    assert mp['dropouts1'] == 0.1
    assert mp['dropouts2'] == 0.3
    assert mp['biasFromPrior'] is False

    tp = train_payload(training_config=tc)
    assert tp['epochs'] == 10
    assert tp['batchSize'] == 64
    assert tp['optimizerConfig']['adamConfig']['lr'] == 0.001
    assert tp['lrSchedulerConfig']['name'] == 'stepLR'
    assert tp['lossConfig']['name'] == 'bceWithLogitsLoss'
    assert tp['earlyStoppingMetric'] == 'f1'

    tc_fast = TrainingCnf(epochs=3, train_validation=False)
    tp_fast = train_payload(training_config=tc_fast)
    assert tp_fast['trainValidation'] is False


def test_class_prior_logits_clip_edge_priors() -> None:
    class DummyData:
        Y = np.array([
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ], dtype=np.float64)

    logits = np.asarray(_class_prior_logits(train_data=DummyData(), eps=1e-6))
    assert np.isfinite(logits).all()
    assert logits.shape == (2,)
    assert logits[0] < 0.0
    assert logits[1] > 0.0


def test_resolve_nn_type_skip_and_leaky() -> None:
    assert _resolve_nn_type({'nnType': 'skip_mlp'}) is SkipMLP
    assert _resolve_nn_type({'nnType': 'mlp_skip'}) is SkipMLP
    assert _resolve_nn_type({'nnType': 'leaky_mlp'}) is LeakyMLP
    assert _resolve_nn_type({'nnType': 'mlp_leaky'}) is LeakyMLP


def test_resolve_nn_type_entity_attention_variants() -> None:
    assert _resolve_nn_type({'nnType': 'entity_attention_mlp'}) is EntityAttentionMLP
    assert _resolve_nn_type({'nnType': 'entity_attention2_mlp'}) is EntityAttention2MLP
    assert _resolve_nn_type({'nnType': 'entity_attention_2_mlp'}) is EntityAttention2MLP


def test_entity_attention2_mlp_forward_shape() -> None:
    model = EntityAttention2MLP(
        embDim=4,
        outDim=3,
        hiddenDim=8,
        entityDim=2,
        attentionHiddenDim=6,
    )
    batch = {
        'article_embeddings': torch.randn(2, 4),
        'entity_embeddings': torch.randn(2, 5, 2),
        'entity_mask': torch.tensor([
            [True, True, False, False, False],
            [True, True, True, False, False],
        ]),
    }
    logits = model(batch)
    assert logits.shape == (2, 3)


def test_output_bias_init_leaky_mlp() -> None:
    target_logits = [0.7, -0.4, 1.5]
    model = NeuralCategModel.create(
        embDim=4,
        outDim=3,
        hiddenDim=8,
        nnType=LeakyMLP,
        textVectorizer='not None',
        biasFromPriorLogits=target_logits,
    )
    bias = model._nn.linear_leaky_relu_stack[-1].bias.detach().cpu()
    assert torch.allclose(bias, torch.tensor(target_logits, dtype=bias.dtype))


def test_wgh_labels_from_score_matrix_includes_all_cats_by_default() -> None:
    score_matrix = np.asarray(
        [[0.7, 0.2, 0.5], [0.1, 0.9, 0.4]],
        dtype=np.float32,
    )
    cat_list = ['A', 'B', 'C']

    labels = list(wgh_labels_from_score_matrix(score_matrix=score_matrix, cat_list=cat_list))

    assert len(labels) == 2
    assert len(labels[0]) == 3
    assert labels[0][0][0] == 'A'
    assert labels[0][0][1] == pytest.approx(0.7)
    assert labels[1][0][0] == 'B'
    assert labels[1][0][1] == pytest.approx(0.9)
    assert {cat for cat, _ in labels[0]} == set(cat_list)


def test_wgh_labels_from_score_matrix_threshold_filters_strictly() -> None:
    score_matrix = np.asarray([[0.2, 0.5, 0.8]], dtype=np.float32)

    kept = list(
        wgh_labels_from_score_matrix(score_matrix=score_matrix, cat_list=['A', 'B', 'C'], thr=0.5)
    )
    cat_ids = [cat for cat, _ in kept[0]]

    assert cat_ids == ['C']


def test_wgh_labels_from_score_matrix_returns_generator() -> None:
    import types

    gen = wgh_labels_from_score_matrix(
        score_matrix=np.zeros((2, 2), dtype=np.float32),
        cat_list=['A', 'B'],
    )

    assert isinstance(gen, types.GeneratorType)


def test_wgh_labels_from_score_matrix_validates_eagerly() -> None:
    with pytest.raises(ValueError, match='columns'):
        wgh_labels_from_score_matrix(
            score_matrix=np.zeros((2, 5), dtype=np.float32),
            cat_list=['A', 'B'],
        )


def test_output_bias_init_enabled_vs_disabled() -> None:
    target_logits = [2.0, -1.0, 0.5]

    torch.manual_seed(0)
    disabled_model = NeuralCategModel.create(
        embDim=4,
        outDim=3,
        hiddenDim=8,
        nnType=MLP,
        textVectorizer='not None',
        biasFromPriorLogits=None,
    )
    disabled_bias = disabled_model._nn.linear_relu_stack[-1].bias.detach().cpu()

    torch.manual_seed(0)
    enabled_model = NeuralCategModel.create(
        embDim=4,
        outDim=3,
        hiddenDim=8,
        nnType=MLP,
        textVectorizer='not None',
        biasFromPriorLogits=target_logits,
    )
    enabled_bias = enabled_model._nn.linear_relu_stack[-1].bias.detach().cpu()
    target_tensor = torch.tensor(target_logits, dtype=enabled_bias.dtype)

    assert torch.allclose(enabled_bias, target_tensor)
    assert not torch.allclose(disabled_bias, target_tensor)
