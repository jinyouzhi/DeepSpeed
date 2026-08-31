# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Test that partition_config receives correct full hierarchical module paths.

The bug: AutoTP._replace_module built ``full_name`` from ``prev_name`` (the
immediate parent only) instead of ``class_name`` (the accumulated hierarchical
path).  Patterns like ``model.layers.0.self_attn.q_proj`` never matched
because the name was just ``0.self_attn.q_proj``.
"""

import pytest
import torch.nn as nn

from deepspeed.module_inject.auto_tp import AutoTP, AutoTPConfig, PartitionType, TPLayerSpec
from deepspeed.module_inject.layers import LinearLayer, VocabParallelLinear
from deepspeed.module_inject.tp_plan_converter import TPPlanConverter
from deepspeed.sequence.cross_entropy import VocabParallelCausalLMLoss, configure_vocab_parallel_loss


class SubAttn(nn.Module):

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(32, 32, bias=False)
        self.k_proj = nn.Linear(32, 32, bias=False)
        self.v_proj = nn.Linear(32, 32, bias=False)
        self.o_proj = nn.Linear(32, 32, bias=False)


class DecoderLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.self_attn = SubAttn()
        self.mlp = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32))


class DummyModel(nn.Module):

    def __init__(self, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(100, 32)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(num_layers)])
        self.head = nn.Linear(32, 100, bias=False)


class OutputModel(nn.Module):

    def __init__(self, tied):
        super().__init__()
        self.config = type("Config", (), {"tie_word_embeddings": not tied})()
        self.embed_tokens = nn.Embedding(100, 32)
        self.lm_head = nn.Linear(32, 100, bias=False)
        if tied:
            self.lm_head.weight = self.embed_tokens.weight


def _build_config():
    """Partition config that matches q_proj and o_proj via regex."""
    return AutoTPConfig(layer_specs=[
        TPLayerSpec(patterns=[r".*\.self_attn\.q_proj"], partition_type=PartitionType.COLUMN),
        TPLayerSpec(patterns=[r".*\.self_attn\.o_proj"], partition_type=PartitionType.ROW),
    ])


def _capture_matched_names(model, config):
    """Run _replace_module and capture full_name values that match a spec."""
    matched_names = []
    original = AutoTP._replace_with_config

    def capture(self, child, full_name):
        # Only capture if a spec actually matches
        param_name = full_name + ".weight"
        model_type = self._get_model_type() if hasattr(self, '_get_model_type') else None
        spec = config.find_matching_spec(param_name, model_type)
        if spec is not None:
            matched_names.append(full_name)
        return None

    AutoTP._replace_with_config = capture
    try:
        autotp = AutoTP(
            module=model,
            all_reduce_linears=[],
            prefix="model",
            state_dict=None,
            linear_layer_setting=None,
            orig_layer_impl=None,
            partition_config=config,
        )
        autotp._replace_module(model)
    finally:
        AutoTP._replace_with_config = original
    return matched_names


def test_partition_config_receives_full_path():
    """Verify that pattern matching sees the full hierarchical path."""
    model = DummyModel(num_layers=2)
    config = _build_config()
    matched_names = _capture_matched_names(model, config)

    for layer_idx in range(2):
        assert f"layers.{layer_idx}.self_attn.q_proj" in matched_names, \
            f"Expected 'layers.{layer_idx}.self_attn.q_proj', got: {matched_names}"
        assert f"layers.{layer_idx}.self_attn.o_proj" in matched_names, \
            f"Expected 'layers.{layer_idx}.self_attn.o_proj', got: {matched_names}"


def test_no_truncated_paths():
    """Ensure paths are never truncated to just the immediate parent prefix."""
    model = DummyModel(num_layers=3)
    config = _build_config()
    matched_names = _capture_matched_names(model, config)

    for name in matched_names:
        assert name.startswith("layers."), \
            f"Path should start with 'layers.', got: {name}"
        assert ".self_attn." in name, \
            f"Path should contain '.self_attn.', got: {name}"
        assert name.count(".") >= 3, \
            f"Path should have at least 3 dots (layers.N.self_attn.X_proj), got: {name}"


def test_nested_depth_correct():
    """Verify correct count and paths at 3 layers deep."""
    model = DummyModel(num_layers=3)
    config = _build_config()
    matched_names = _capture_matched_names(model, config)

    expected_count = 3 * 2  # 3 layers × (q_proj + o_proj)
    assert len(matched_names) == expected_count, \
        f"Expected {expected_count} matches, got {len(matched_names)}: {matched_names}"

    for layer_idx in range(3):
        assert f"layers.{layer_idx}.self_attn.q_proj" in matched_names
        assert f"layers.{layer_idx}.self_attn.o_proj" in matched_names


def _build_gathered_lm_head_autotp(model, mp_size=1):
    config = AutoTPConfig(layer_specs=[
        TPLayerSpec(
            patterns=[r".*lm_head\.weight$"],
            partition_type=PartitionType.COLUMN,
            gather_output=True,
        ),
    ])
    autotp = AutoTP(
        module=model,
        all_reduce_linears=[],
        prefix="",
        state_dict=None,
        linear_layer_setting=None,
        orig_layer_impl=None,
        partition_config=config,
    )
    autotp.set_tensor_parallel_config(mp_size, None)
    autotp.update_linear_policies()
    return autotp


def _build_local_lm_head_autotp(model, vocab_parallel_lm_head=True):
    config = AutoTPConfig(layer_specs=[
        TPLayerSpec(patterns=[r".*lm_head\.weight$"], partition_type=PartitionType.COLUMN),
    ])
    autotp = AutoTP(
        module=model,
        all_reduce_linears=[],
        prefix="",
        state_dict=None,
        linear_layer_setting=None,
        orig_layer_impl=None,
        partition_config=config,
        vocab_parallel_lm_head=vocab_parallel_lm_head,
    )
    autotp.set_tensor_parallel_config(1, None)
    autotp.update_linear_policies()
    return autotp


def test_gathered_lm_head_uses_column_parallel_layer_when_untied():
    model = OutputModel(tied=False)
    _build_gathered_lm_head_autotp(model)._replace_module(model)

    assert isinstance(model.lm_head, LinearLayer)
    assert model.lm_head.gather_output


def test_gathered_lm_head_falls_back_for_runtime_parameter_tie():
    model = OutputModel(tied=True)
    assert model.lm_head.weight is model.embed_tokens.weight

    _build_gathered_lm_head_autotp(model)._replace_module(model)

    assert isinstance(model.embed_tokens, nn.Embedding)
    assert isinstance(model.lm_head, nn.Linear)
    assert model.lm_head.weight is model.embed_tokens.weight


def test_tied_embedding_plan_leaves_lm_head_and_embedding_replicated():
    model = OutputModel(tied=True)
    specs = TPPlanConverter.convert({"embed_tokens": "embedding_rowwise", "lm_head": "colwise_gather_output"})

    autotp = AutoTP(
        module=model,
        all_reduce_linears=[],
        prefix="",
        state_dict=None,
        linear_layer_setting=None,
        orig_layer_impl=None,
        partition_config=AutoTPConfig(layer_specs=specs),
    )
    autotp.set_tensor_parallel_config(1, None)
    autotp.update_linear_policies()
    autotp._replace_module(model)

    assert isinstance(model.embed_tokens, nn.Embedding)
    assert isinstance(model.lm_head, nn.Linear)
    assert model.lm_head.weight is model.embed_tokens.weight


def test_gathered_lm_head_uses_column_parallel_layer_when_output_dim_is_uneven():
    model = OutputModel(tied=False)
    model.lm_head = nn.Linear(32, 101, bias=False)

    _build_gathered_lm_head_autotp(model, mp_size=2)._replace_module(model)

    assert isinstance(model.lm_head, LinearLayer)
    assert model.lm_head.gather_output


def test_vocab_parallel_linear_exposes_vocab_metadata():
    layer = VocabParallelLinear(nn.Linear(32, 101, bias=False), mp_group=None, name="lm_head")

    assert layer.vocab_size == 101
    assert layer.vocab_start_index == 0
    assert layer.vocab_end_index == 101
    assert not layer.gather_output


def test_plain_colwise_lm_head_uses_vocab_parallel_layer():
    model = OutputModel(tied=False)

    _build_local_lm_head_autotp(model, vocab_parallel_lm_head=True)._replace_module(model)

    assert isinstance(model.lm_head, VocabParallelLinear)


def test_plain_colwise_lm_head_without_flag_keeps_local_logits():
    model = OutputModel(tied=False)

    _build_local_lm_head_autotp(model, vocab_parallel_lm_head=False)._replace_module(model)

    assert isinstance(model.lm_head, LinearLayer)
    assert not isinstance(model.lm_head, VocabParallelLinear)
    assert not model.lm_head.gather_output


@pytest.mark.parametrize("name,expected", [
    ("lm_head", True),
    ("embed_out", True),
    ("model.lm_head", True),
    ("lm_head_proj", False),
    ("model.lm_head_proj.weight", False),
    ("inner_lm_head.block", False),
])
def test_lm_head_name_matching_ignores_projections(name, expected):
    assert AutoTP._is_lm_head_name(name) is expected


def test_plain_colwise_lm_head_rejects_tied_weights():
    model = OutputModel(tied=True)

    with pytest.raises(ValueError, match="requires untied"):
        _build_local_lm_head_autotp(model)._replace_module(model)


def test_configure_vocab_parallel_loss_installs_and_preserves_hook():
    model = OutputModel(tied=False)
    model.loss_function = nn.CrossEntropyLoss()
    original_loss_function = model.loss_function
    _build_local_lm_head_autotp(model)._replace_module(model)

    configure_vocab_parallel_loss(model, model.lm_head)

    assert isinstance(model.loss_function, VocabParallelCausalLMLoss)
    assert model._deepspeed_original_loss_function is original_loss_function


def test_configure_vocab_parallel_loss_requires_hook():
    model = OutputModel(tied=False)
    _build_local_lm_head_autotp(model)._replace_module(model)

    with pytest.raises(ValueError, match="requires a writable loss_function"):
        configure_vocab_parallel_loss(model, model.lm_head)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
