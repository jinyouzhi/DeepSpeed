# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import deepspeed.comm as dist
import deepspeed
import importlib.util
from copy import deepcopy
from torch import nn

from unit.common import DistributedTest, preferred_dtype
from deepspeed.accelerator import get_accelerator
from deepspeed.utils import groups
from deepspeed.module_inject.layers import (LinearAllreduce, LinearLayer, SubParamLinearLayer, VocabParallelLinear,
                                            fused_LinearLayer, set_autotp_mode)
from deepspeed.module_inject.autotp_config import AutoTPConfig
from deepspeed.module_inject.auto_tp import AutoTP
from deepspeed.sequence.cross_entropy import VocabParallelCausalLMLoss, vocab_parallel_cross_entropy


def skip_on_device():
    if get_accelerator().device_name() == 'xpu':
        pytest.skip("XPU requires a higher version for test")


class SequentialLinearModel(torch.nn.Module):

    def __init__(self, hidden_dim, nlayers=1):
        super(SequentialLinearModel, self).__init__()
        self.linears = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(nlayers)])

    def forward(self, x):
        for layer in self.linears:
            x = layer(x)
        return x


class CustomLinearModule(torch.nn.Module):

    def __init__(self, hidden_dim):
        super(CustomLinearModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.bias = torch.nn.Parameter(torch.empty(hidden_dim))
        torch.nn.init.uniform_(self.weight, -0.02, 0.02)
        torch.nn.init.uniform_(self.bias, -0.02, 0.02)

    def forward(self, x):
        return torch.matmul(x, self.weight.transpose(-1, -2)) + self.bias


class CustomLinearModel(torch.nn.Module):

    def __init__(self, hidden_dim):
        super(CustomLinearModel, self).__init__()
        self.custom = CustomLinearModule(hidden_dim)

    def forward(self, x):
        return self.custom(x)


class QKVLinearModule(torch.nn.Module):

    def __init__(self, hidden_dim):
        super(QKVLinearModule, self).__init__()
        self.qkv_proj = torch.nn.Linear(hidden_dim, hidden_dim * 3)

    def forward(self, x):
        return self.qkv_proj(x)


class QKVLinearModel(torch.nn.Module):

    def __init__(self, hidden_dim):
        super(QKVLinearModel, self).__init__()
        self.self_attn = QKVLinearModule(hidden_dim)

    def forward(self, x):
        return self.self_attn(x)


class DeepAttention(torch.nn.Module):
    """Mimics HF attention module with separate projection layers."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.q_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.o_proj(self.q_proj(x))


class DeepBlock(torch.nn.Module):
    """Mimics a single HF transformer block."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.self_attn = DeepAttention(hidden_dim)

    def forward(self, x):
        return self.self_attn(x)


class DeepModel(torch.nn.Module):
    """Mimics HF transformer structure: model.layers.[N].self_attn.{q,o}_proj.

    This creates a 4-level-deep module hierarchy to test that _replace_module
    correctly propagates the full module path during recursion.
    """

    def __init__(self, hidden_dim, nlayers=2):
        super().__init__()
        self.layers = torch.nn.ModuleList([DeepBlock(hidden_dim) for _ in range(nlayers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class UntiedCausalLM(torch.nn.Module):
    """Small causal LM used to exercise the vocab-parallel LM-head path."""

    def __init__(self, hidden_dim=8, vocab_size=16):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.vocab_size = vocab_size
        self.config = type("Config", (), {
            "model_type": "untied_test",
            "tie_word_embeddings": False,
        })()
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels=None):
        hidden_states = self.proj(self.embed_tokens(input_ids))
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.vocab_size)
        return type("CausalLMOutput", (), {"loss": loss, "logits": logits})()


class TiedCausalLM(UntiedCausalLM):

    def __init__(self, hidden_dim=8, vocab_size=16):
        super().__init__(hidden_dim=hidden_dim, vocab_size=vocab_size)
        self.lm_head.weight = self.embed_tokens.weight
        self.config.tie_word_embeddings = True


def init_tp_engine(tp_size, partition_config=None):
    config_dict = {
        "train_micro_batch_size_per_gpu": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-6
            }
        },
        "tensor_parallel": {
            "autotp_size": tp_size,
        },
        "zero_optimization": {
            "stage": 0,
        }
    }
    if partition_config is not None:
        config_dict["tensor_parallel"]["partition_config"] = partition_config
    else:
        config_dict["tensor_parallel"]["partition_config"] = {
            "use_default_specs": False,
            "layer_specs": [{
                "patterns": [".*\\.weight$"],
                "partition_type": "skip",
            }],
        }
    if preferred_dtype() is torch.float16:
        config_dict["fp16"] = {"enabled": True}
    elif preferred_dtype() is torch.bfloat16:
        config_dict["bf16"] = {"enabled": True}

    model = SequentialLinearModel(hidden_dim=8)
    deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)


def apply_autotp_with_partition_config(model, tp_size, partition_config):
    groups._init_tp_mesh_device(tensor_model_parallel_size=tp_size)
    autotp_config = AutoTPConfig.from_dict(partition_config)
    autotp = AutoTP(module=model,
                    all_reduce_linears=[],
                    prefix="",
                    state_dict=None,
                    linear_layer_setting=None,
                    orig_layer_impl=None,
                    keep_module_on_host=False,
                    partition_config=autotp_config)
    autotp.set_tensor_parallel_config(tp_size, groups.get_tensor_model_parallel_group())
    autotp.update_linear_policies()
    autotp._replace_module(model)
    return model


def gather_subparam_output(output, subparam_sizes, mp_group):
    tp_world_size = dist.get_world_size(group=mp_group)
    local_sizes = [size // tp_world_size for size in subparam_sizes]
    output_chunks = torch.split(output, local_sizes, dim=-1)
    gathered_chunks = []
    for chunk in output_chunks:
        chunk = chunk.contiguous()
        gathered = [torch.empty_like(chunk) for _ in range(tp_world_size)]
        dist.all_gather(gathered, chunk, group=mp_group)
        gathered_chunks.append(torch.cat(gathered, dim=-1))
    return torch.cat(gathered_chunks, dim=-1)


def assert_close_for_preferred_dtype(actual, expected):
    atol = 1e-3
    rtol = 2e-2
    if preferred_dtype() is torch.float32:
        atol = 1e-5
        rtol = 1e-5
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


class TestAutoTPCustomPatterns(DistributedTest):
    world_size = 2
    reuse_dist_env = False

    def test_custom_pattern_replacement(self):
        skip_on_device()
        partition_config = {
            "use_default_specs":
            False,
            "layer_specs": [
                {
                    "patterns": [".*linears\\.0\\.weight$"],
                    "partition_type": "row",
                },
                {
                    "patterns": [".*linears\\.1\\.weight$"],
                    "partition_type": "column",
                },
                {
                    "patterns": [".*linears\\.2\\.weight$"],
                    "partition_type": "skip",
                },
            ],
        }
        model = SequentialLinearModel(hidden_dim=16, nlayers=3)
        model = apply_autotp_with_partition_config(model, tp_size=2, partition_config=partition_config)

        assert isinstance(model.linears[0], LinearAllreduce)
        assert isinstance(model.linears[1], LinearLayer)
        assert isinstance(model.linears[2], nn.Linear)

    def test_custom_patterns_applied_via_config(self):
        skip_on_device()
        partition_config = {
            "use_default_specs":
            False,
            "layer_specs": [
                {
                    "patterns": [".*linears\\.0\\.weight$"],
                    "partition_type": "row",
                },
                {
                    "patterns": [".*linears\\.1\\.weight$"],
                    "partition_type": "column",
                },
                {
                    "patterns": [".*linears\\.2\\.weight$"],
                    "partition_type": "skip",
                },
            ],
        }
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-6
                }
            },
            "tensor_parallel": {
                "autotp_size": 2,
                "partition_config": partition_config,
            },
            "zero_optimization": {
                "stage": 0,
            }
        }
        if preferred_dtype() is torch.float16:
            config_dict["fp16"] = {"enabled": True}
        elif preferred_dtype() is torch.bfloat16:
            config_dict["bf16"] = {"enabled": True}

        model = SequentialLinearModel(hidden_dim=16, nlayers=3)
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
        assert isinstance(engine.module.linears[0], LinearAllreduce)
        assert isinstance(engine.module.linears[1], LinearLayer)
        assert isinstance(engine.module.linears[2], nn.Linear)

    def test_use_default_specs_false_skips_unmatched_layers(self):
        skip_on_device()
        # Verify unmatched layers remain unsharded when defaults are disabled.
        partition_config = {
            "use_default_specs":
            False,
            "layer_specs": [
                {
                    "patterns": [".*linears\\.0\\.weight$"],
                    "partition_type": "row",
                },
                {
                    "patterns": [".*linears\\.1\\.weight$"],
                    "partition_type": "column",
                },
            ],
        }
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-6
                }
            },
            "tensor_parallel": {
                "autotp_size": 2,
                "partition_config": partition_config,
            },
            "zero_optimization": {
                "stage": 0,
            }
        }
        if preferred_dtype() is torch.float16:
            config_dict["fp16"] = {"enabled": True}
        elif preferred_dtype() is torch.bfloat16:
            config_dict["bf16"] = {"enabled": True}

        model = SequentialLinearModel(hidden_dim=16, nlayers=3)
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
        assert isinstance(engine.module.linears[0], LinearAllreduce)
        assert isinstance(engine.module.linears[1], LinearLayer)
        assert isinstance(engine.module.linears[2], nn.Linear)

    def test_custom_module_replacement_with_patterns(self):
        skip_on_device()
        # Verify custom linear-like modules are partitioned via patterns.
        partition_config = {
            "use_default_specs": False,
            "layer_specs": [
                {
                    "patterns": [".*custom\\.weight$"],
                    "partition_type": "column",
                },
            ],
        }
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-6
                }
            },
            "tensor_parallel": {
                "autotp_size": 2,
                "partition_config": partition_config,
            },
            "zero_optimization": {
                "stage": 0,
            }
        }
        if preferred_dtype() is torch.float16:
            config_dict["fp16"] = {"enabled": True}
        elif preferred_dtype() is torch.bfloat16:
            config_dict["bf16"] = {"enabled": True}

        model = CustomLinearModel(hidden_dim=16)
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
        assert isinstance(engine.module.custom, LinearLayer)

    def test_custom_pattern_disables_fused_qkv_heuristic(self):
        skip_on_device()
        # Use a qkv_proj name that would trigger the fused-QKV heuristic, then
        # verify custom patterns override that path and preserve correctness.
        torch.manual_seed(1234)
        hidden_dim = 16
        qkv_sizes = (hidden_dim, hidden_dim, hidden_dim)
        partition_config = {
            "use_default_specs":
            False,
            "layer_specs": [
                {
                    "patterns": [".*self_attn\\.qkv_proj\\.weight$"],
                    "partition_type": "column",
                    "shape": [list(qkv_sizes), -1],
                    "partition_dim": 0,
                },
            ],
        }
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-6
                }
            },
            "tensor_parallel": {
                "autotp_size": 2,
                "partition_config": partition_config,
            },
            "zero_optimization": {
                "stage": 0,
            }
        }
        if preferred_dtype() is torch.float16:
            config_dict["fp16"] = {"enabled": True}
        elif preferred_dtype() is torch.bfloat16:
            config_dict["bf16"] = {"enabled": True}

        model = QKVLinearModel(hidden_dim=hidden_dim)
        baseline = deepcopy(model).to(get_accelerator().current_device(), dtype=preferred_dtype())
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)
        qkv_layer = engine.module.self_attn.qkv_proj
        # Custom pattern should force SubParamLinearLayer (shape-based path),
        # and avoid the legacy fused-QKV heuristic despite the qkv_proj name.
        assert isinstance(qkv_layer, SubParamLinearLayer)
        assert not isinstance(qkv_layer, fused_LinearLayer)

        assert qkv_layer.partition_dim == 0
        assert qkv_layer._subparam_sizes == qkv_sizes
        assert qkv_layer._orig_weight_shape == (hidden_dim * 3, hidden_dim)

        qkv_layer.gather_params([qkv_layer.weight, qkv_layer.bias])
        torch.testing.assert_close(qkv_layer.weight, baseline.self_attn.qkv_proj.weight)
        if qkv_layer.bias is not None:
            torch.testing.assert_close(qkv_layer.bias, baseline.self_attn.qkv_proj.bias)

        torch.manual_seed(4321)
        inputs = torch.randn(2, hidden_dim, dtype=preferred_dtype(), device=get_accelerator().current_device())
        full_output = baseline(inputs)
        tp_output = engine.module(inputs)
        assert_close_for_preferred_dtype(tp_output, full_output)

    def test_first_match_precedence(self):
        skip_on_device()
        partition_config = {
            "use_default_specs":
            False,
            "layer_specs": [
                {
                    "patterns": [".*linears\\.0\\.weight$"],
                    "partition_type": "skip",
                },
                {
                    "patterns": [".*linears\\.0\\.weight$"],
                    "partition_type": "column",
                },
            ],
        }
        model = SequentialLinearModel(hidden_dim=16, nlayers=1)
        model = apply_autotp_with_partition_config(model, tp_size=2, partition_config=partition_config)

        assert isinstance(model.linears[0], nn.Linear)

    def test_deep_model_full_path_propagation(self):
        """Verify _replace_module propagates accumulated paths through deep hierarchies.

        Uses a 4-level-deep model (layers.N.self_attn.{q,o}_proj) with patterns
        that require intermediate path components (layers.N). Without correct
        full_name propagation, the recursive path is truncated and patterns
        that include intermediate levels will silently fail to match.
        """
        skip_on_device()
        partition_config = {
            "use_default_specs":
            False,
            "layer_specs": [
                {
                    "patterns": [r".*layers\.\d+\.self_attn\.q_proj\.weight$"],
                    "partition_type": "column",
                },
                {
                    "patterns": [r".*layers\.\d+\.self_attn\.o_proj\.weight$"],
                    "partition_type": "row",
                },
            ],
        }
        model = DeepModel(hidden_dim=16, nlayers=2)
        model = apply_autotp_with_partition_config(model, tp_size=2, partition_config=partition_config)

        # All 4 projections (2 layers x {q_proj, o_proj}) must be replaced.
        # Before the full_name fix, 0 modules were replaced because the mangled
        # path "self_attn.q_proj.weight" could not match "layers.N.self_attn...".
        for i in range(2):
            assert isinstance(model.layers[i].self_attn.q_proj, LinearLayer), \
                f"layers.{i}.self_attn.q_proj was not replaced (path propagation bug?)"
            assert isinstance(model.layers[i].self_attn.o_proj, LinearAllreduce), \
                f"layers.{i}.self_attn.o_proj was not replaced (path propagation bug?)"

    def test_vocab_parallel_lm_head_tp1(self):
        skip_on_device()
        set_autotp_mode(training=True)
        try:
            torch.manual_seed(42)
            linear = nn.Linear(8, 16, bias=False)
            layer = VocabParallelLinear(linear, mp_group=None)
            inputs = torch.randn(2, 3, 8)

            torch.testing.assert_close(layer(inputs), linear(inputs))
            assert layer.weight.shape == (16, 8)
            assert layer.vocab_size == 16
            assert layer.vocab_start_index == 0
        finally:
            set_autotp_mode(training=False)

    def test_vocab_parallel_lm_head_rejects_tied_weights_tp2(self):
        skip_on_device()
        groups._init_tp_mesh_device(tensor_model_parallel_size=2)
        autotp = AutoTP(module=TiedCausalLM(),
                        all_reduce_linears=(),
                        prefix="",
                        state_dict=None,
                        linear_layer_setting=(torch.nn.Linear, torch.nn.Embedding),
                        orig_layer_impl=None,
                        keep_module_on_host=False,
                        vocab_parallel_lm_head=True)
        autotp.set_tensor_parallel_config(2, groups.get_tensor_model_parallel_group())
        set_autotp_mode(training=True)
        try:
            with pytest.raises(ValueError, match="untied"):
                autotp.replace_vocab_parallel_lm_head(autotp.module)
        finally:
            set_autotp_mode(training=False)

    def test_vocab_parallel_lm_head_rejects_nondivisible_vocab_tp2(self):
        skip_on_device()
        groups._init_tp_mesh_device(tensor_model_parallel_size=2)
        set_autotp_mode(training=True)
        try:
            device = get_accelerator().current_device()
            linear = nn.Linear(8, 17, bias=False, device=device)
            with pytest.raises(ValueError, match="divisible"):
                VocabParallelLinear(linear, groups.get_tensor_model_parallel_group(), name="lm_head")
        finally:
            set_autotp_mode(training=False)

    def test_vocab_parallel_cross_entropy_tp1_matches_torch(self):
        skip_on_device()
        torch.manual_seed(42)
        logits = torch.randn(2, 3, 16, requires_grad=True)
        reference_logits = logits.detach().clone().requires_grad_(True)
        target = torch.randint(0, 16, (2, 3))
        target[0, 0] = -100

        expected = torch.nn.functional.cross_entropy(reference_logits.view(-1, 16), target.view(-1))
        actual = vocab_parallel_cross_entropy(logits, target, reduction="mean")

        torch.testing.assert_close(actual, expected)
        actual.backward()
        expected.backward()
        torch.testing.assert_close(logits.grad, reference_logits.grad)

    def test_vocab_parallel_cross_entropy_tp2_matches_torch(self):
        skip_on_device()
        groups._init_tp_mesh_device(tensor_model_parallel_size=2)
        tp_group = groups.get_tensor_model_parallel_group()
        tp_rank = groups.get_tensor_model_parallel_rank()

        torch.manual_seed(42)
        full_logits = torch.randn(2, 3, 16, device=get_accelerator().current_device())
        reference_logits = full_logits.detach().clone().requires_grad_(True)
        vocab_start_index = tp_rank * 8
        local_logits = full_logits[..., vocab_start_index:vocab_start_index + 8].detach().clone().requires_grad_(True)
        target = torch.tensor([[0, 8, -100], [15, 7, 9]],
                              dtype=torch.long,
                              device=get_accelerator().current_device())

        expected_none = torch.nn.functional.cross_entropy(reference_logits.view(-1, 16),
                                                          target.view(-1),
                                                          reduction="none").view_as(target)
        actual_none = vocab_parallel_cross_entropy(local_logits,
                                                   target,
                                                   tp_group=tp_group,
                                                   vocab_start_index=vocab_start_index,
                                                   reduction="none")
        torch.testing.assert_close(actual_none, expected_none)

        expected = expected_none.sum() / (target != -100).sum()
        actual = vocab_parallel_cross_entropy(local_logits,
                                              target,
                                              tp_group=tp_group,
                                              vocab_start_index=vocab_start_index,
                                              reduction="mean")
        torch.testing.assert_close(actual, expected)

        actual.backward()
        expected.backward()
        expected_local_grad = reference_logits.grad[..., vocab_start_index:vocab_start_index + 8]
        torch.testing.assert_close(local_logits.grad, expected_local_grad)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_vocab_parallel_cross_entropy_tp2_dtype(self, dtype):
        skip_on_device()
        if dtype == torch.bfloat16 and not get_accelerator().is_bf16_supported():
            pytest.skip("BF16 is not supported by the active accelerator")

        groups._init_tp_mesh_device(tensor_model_parallel_size=2)
        tp_group = groups.get_tensor_model_parallel_group()
        tp_rank = groups.get_tensor_model_parallel_rank()
        device = get_accelerator().current_device()

        torch.manual_seed(314)
        full_logits = torch.randn(2, 4, 16, dtype=dtype, device=device)
        reference_logits = full_logits.detach().clone().requires_grad_(True)
        vocab_start_index = tp_rank * 8
        local_logits = full_logits[..., vocab_start_index:vocab_start_index + 8].detach().clone()
        local_logits.requires_grad_(True)
        assert local_logits.dtype == dtype
        target = torch.tensor([[0, 8, -100, 15], [7, 9, 3, -100]], dtype=torch.long, device=device)

        expected = torch.nn.functional.cross_entropy(reference_logits.float().view(-1, 16), target.view(-1))
        actual = vocab_parallel_cross_entropy(local_logits,
                                              target,
                                              tp_group=tp_group,
                                              vocab_start_index=vocab_start_index,
                                              reduction="mean")
        atol = 2e-2 if dtype == torch.bfloat16 else 1e-5
        rtol = 2e-2 if dtype == torch.bfloat16 else 1e-5
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

        actual.backward()
        expected.backward()
        expected_local_grad = reference_logits.grad[..., vocab_start_index:vocab_start_index + 8]
        assert local_logits.grad.dtype == dtype
        torch.testing.assert_close(local_logits.grad, expected_local_grad, atol=atol, rtol=rtol)

    def test_vocab_parallel_lm_head_tp2_matches_full_linear(self):
        skip_on_device()
        groups._init_tp_mesh_device(tensor_model_parallel_size=2)
        tp_group = groups.get_tensor_model_parallel_group()
        tp_rank = groups.get_tensor_model_parallel_rank()
        set_autotp_mode(training=True)

        try:
            torch.manual_seed(123)
            hidden_dim = 8
            vocab_size = 16
            device = get_accelerator().current_device()
            reference_linear = nn.Linear(hidden_dim, vocab_size, bias=False, dtype=torch.float32, device=device)
            sharded_linear = VocabParallelLinear(deepcopy(reference_linear), tp_group, name="lm_head")

            torch.manual_seed(456)
            reference_inputs = torch.randn(2, 3, hidden_dim, dtype=torch.float32, device=device, requires_grad=True)
            sharded_inputs = reference_inputs.detach().clone().requires_grad_(True)
            full_logits = reference_linear(reference_inputs)
            local_logits = sharded_linear(sharded_inputs)

            gathered_logits = [torch.empty_like(local_logits) for _ in range(2)]
            dist.all_gather(gathered_logits, local_logits.detach().contiguous(), group=tp_group)
            torch.testing.assert_close(torch.cat(gathered_logits, dim=-1), full_logits.detach())

            target = torch.tensor([[0, 8, -100], [15, 7, 9]], dtype=torch.long, device=device)
            expected = torch.nn.functional.cross_entropy(full_logits.view(-1, vocab_size), target.view(-1))
            actual = vocab_parallel_cross_entropy(local_logits,
                                                  target,
                                                  tp_group=tp_group,
                                                  vocab_start_index=sharded_linear.vocab_start_index,
                                                  reduction="mean")
            torch.testing.assert_close(actual, expected)

            actual.backward()
            expected.backward()
            local_weight_grad = reference_linear.weight.grad.narrow(0, tp_rank * 8, 8)
            torch.testing.assert_close(sharded_linear.weight.grad, local_weight_grad)
            torch.testing.assert_close(sharded_inputs.grad, reference_inputs.grad)
        finally:
            set_autotp_mode(training=False)

    def test_vocab_parallel_lm_head_sgd_step_tp2_matches_full_linear(self):
        skip_on_device()
        groups._init_tp_mesh_device(tensor_model_parallel_size=2)
        tp_group = groups.get_tensor_model_parallel_group()
        tp_rank = groups.get_tensor_model_parallel_rank()
        set_autotp_mode(training=True)

        try:
            torch.manual_seed(2718)
            device = get_accelerator().current_device()
            reference_linear = nn.Linear(8, 16, bias=False, dtype=torch.float32, device=device)
            sharded_linear = VocabParallelLinear(deepcopy(reference_linear), tp_group, name="lm_head")
            reference_inputs = torch.randn(2, 3, 8, dtype=torch.float32, device=device, requires_grad=True)
            sharded_inputs = reference_inputs.detach().clone().requires_grad_(True)
            target = torch.tensor([[0, 8, -100], [15, 7, 9]], dtype=torch.long, device=device)

            expected = torch.nn.functional.cross_entropy(reference_linear(reference_inputs).view(-1, 16),
                                                         target.view(-1))
            actual = vocab_parallel_cross_entropy(sharded_linear(sharded_inputs),
                                                  target,
                                                  tp_group=tp_group,
                                                  vocab_start_index=sharded_linear.vocab_start_index,
                                                  reduction="mean")
            actual.backward()
            expected.backward()

            expected_local_grad = reference_linear.weight.grad.narrow(0, tp_rank * 8, 8)
            torch.testing.assert_close(sharded_linear.weight.grad, expected_local_grad)
            torch.testing.assert_close(sharded_inputs.grad, reference_inputs.grad)

            reference_optimizer = torch.optim.SGD(reference_linear.parameters(), lr=0.05)
            sharded_optimizer = torch.optim.SGD(sharded_linear.parameters(), lr=0.05)
            reference_optimizer.step()
            sharded_optimizer.step()

            expected_local_weight = reference_linear.weight.detach().narrow(0, tp_rank * 8, 8)
            torch.testing.assert_close(sharded_linear.weight.detach(), expected_local_weight)
        finally:
            set_autotp_mode(training=False)

    def test_vocab_parallel_causal_lm_loss_tp2_matches_torch(self):
        skip_on_device()
        groups._init_tp_mesh_device(tensor_model_parallel_size=2)
        tp_group = groups.get_tensor_model_parallel_group()
        tp_rank = groups.get_tensor_model_parallel_rank()

        torch.manual_seed(789)
        full_logits = torch.randn(2, 5, 16, device=get_accelerator().current_device())
        reference_logits = full_logits.detach().clone().requires_grad_(True)
        vocab_start_index = tp_rank * 8
        local_logits = full_logits[..., vocab_start_index:vocab_start_index + 8].detach().clone().requires_grad_(True)
        labels = torch.tensor([[1, 8, -100, 15, 7], [0, 3, 9, -100, 12]],
                              dtype=torch.long,
                              device=get_accelerator().current_device())

        expected = torch.nn.functional.cross_entropy(reference_logits[:, :-1, :].contiguous().view(-1, 16),
                                                     labels[:, 1:].contiguous().view(-1))
        loss_fn = VocabParallelCausalLMLoss(tp_group=tp_group)
        actual = loss_fn(local_logits, labels=labels, vocab_size=16)
        torch.testing.assert_close(actual, expected)

        actual.backward()
        expected.backward()
        expected_local_grad = reference_logits.grad[..., vocab_start_index:vocab_start_index + 8]
        torch.testing.assert_close(local_logits.grad, expected_local_grad)

    def test_vocab_parallel_causal_lm_loss_ulysses_shift_labels_long_sequence(self):
        skip_on_device()
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        sp_group = dist.new_group(ranks=list(range(world_size)))
        device = get_accelerator().current_device()
        sequence_length = 1024
        assert sequence_length % world_size == 0

        torch.manual_seed(1618)
        full_logits = torch.randn(1, sequence_length, 16, device=device)
        shift_labels = torch.randint(0, 16, (1, sequence_length), dtype=torch.long, device=device)
        shift_labels[:, ::31] = -100

        local_sequence_length = sequence_length // world_size
        sequence_start = rank * local_sequence_length
        sequence_end = sequence_start + local_sequence_length
        local_logits = full_logits[:, sequence_start:sequence_end, :].detach().clone().requires_grad_(True)
        local_shift_labels = shift_labels[:, sequence_start:sequence_end]

        # Ulysses supplies rank-local logits and explicit shifted labels. Each
        # rank computes a mean over its valid tokens before weighted aggregation.
        loss_fn = VocabParallelCausalLMLoss(tp_group=None)
        local_loss = loss_fn(local_logits,
                             labels=None,
                             shift_labels=local_shift_labels,
                             vocab_size=16)
        local_count = (local_shift_labels != -100).sum().to(dtype=local_loss.dtype)
        global_loss_sum = (local_loss * local_count).detach().clone()
        global_token_count = local_count.detach().clone()
        dist.all_reduce(global_loss_sum, op=dist.ReduceOp.SUM, group=sp_group)
        dist.all_reduce(global_token_count, op=dist.ReduceOp.SUM, group=sp_group)
        actual = global_loss_sum / global_token_count.clamp_min(1)

        expected = torch.nn.functional.cross_entropy(full_logits.view(-1, 16), shift_labels.view(-1))
        torch.testing.assert_close(actual, expected)

    def test_liger_unavailable_is_actionable(self):
        skip_on_device()
        if importlib.util.find_spec("liger_kernel") is not None:
            pytest.skip("Liger Kernel is installed; optional backend path is covered separately")

        logits = torch.randn(1, 2, 8, requires_grad=True)
        target = torch.tensor([[0, 3]], dtype=torch.long)
        fallback_loss = vocab_parallel_cross_entropy(logits, target, use_liger=False)
        assert torch.isfinite(fallback_loss)
        with pytest.raises(ImportError, match="Liger Kernel is required"):
            vocab_parallel_cross_entropy(logits, target, use_liger=True)

    @pytest.mark.parametrize("zero_stage", [0, 1, 2])
    def test_vocab_parallel_lm_head_zero_optimizer_stages(self, zero_stage):
        skip_on_device()
        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-6
                }
            },
            "tensor_parallel": {
                "autotp_size": 2,
                "vocab_parallel_lm_head": True,
                "partition_config": {
                    "use_default_specs": False,
                    "layer_specs": [{
                        "patterns": [".*\\.weight$"],
                        "partition_type": "skip",
                    }],
                },
            },
            "zero_optimization": {
                "stage": zero_stage,
            },
        }

        torch.manual_seed(2024)
        model = UntiedCausalLM()
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config_dict)

        assert isinstance(engine.module.lm_head, VocabParallelLinear)
        assert engine.module.lm_head.weight.shape == (8, 8)
        assert engine.module.config.tie_word_embeddings is False
        assert engine.module.lm_head.weight is not engine.module.embed_tokens.weight
        original_weight = engine.module.lm_head.weight.detach().clone()

        input_ids = torch.randint(0, 16, (1, 4), device=engine.device)
        dist.broadcast(input_ids,
                       src=groups.get_tensor_model_parallel_src_rank(),
                       group=groups.get_tensor_model_parallel_group())
        outputs = engine(input_ids, labels=input_ids)
        assert outputs.logits.shape == (1, 4, 8)
        assert torch.isfinite(outputs.loss)
        engine.backward(outputs.loss)
        engine.step()

        updated_weight = engine.module.lm_head.weight.detach()
        assert torch.isfinite(updated_weight).all()
        assert not torch.equal(original_weight, updated_weight)


def test_invalid_custom_shape_rejected():
    bad_config = {
        "layer_specs": [{
            "patterns": [".*"],
            "partition_type": "column",
            "shape": [2, [1, 1]],
        }]
    }
    with pytest.raises(ValueError, match="nested tuple only allowed at partition_dim"):
        AutoTPConfig.from_dict(bad_config)


class TestAutoTPFusedWeights(DistributedTest):
    world_size = 2
    reuse_dist_env = False

    def test_gate_up_fused_weight_partition(self):
        skip_on_device()
        init_tp_engine(tp_size=2)

        hidden_dim = 8
        torch.manual_seed(42)
        linear = nn.Linear(hidden_dim,
                           hidden_dim * 2,
                           bias=True,
                           dtype=preferred_dtype(),
                           device=get_accelerator().current_device())
        full_weight = deepcopy(linear.weight.data)
        full_bias = deepcopy(linear.bias.data)

        layer = SubParamLinearLayer(deepcopy(linear),
                                    groups.get_tensor_model_parallel_group(),
                                    shape=(2, -1),
                                    partition_dim=0,
                                    name="mlp.gate_up_proj")
        assert layer._subparam_sizes == (hidden_dim, hidden_dim)
        assert layer.weight.shape == (hidden_dim, hidden_dim)

        layer.gather_params([layer.weight, layer.bias])
        torch.testing.assert_close(layer.weight.data, full_weight)
        torch.testing.assert_close(layer.bias.data, full_bias)

    def test_gqa_uneven_qkv_fused_weight_partition(self):
        skip_on_device()
        init_tp_engine(tp_size=2)

        hidden_dim = 8
        q_size, k_size, v_size = 8, 4, 4
        torch.manual_seed(123)
        linear = nn.Linear(hidden_dim,
                           q_size + k_size + v_size,
                           bias=True,
                           dtype=preferred_dtype(),
                           device=get_accelerator().current_device())
        full_weight = deepcopy(linear.weight.data)
        full_bias = deepcopy(linear.bias.data)

        layer = SubParamLinearLayer(deepcopy(linear),
                                    groups.get_tensor_model_parallel_group(),
                                    shape=((q_size, k_size, v_size), -1),
                                    partition_dim=0,
                                    name="self_attn.qkv_proj")
        assert layer._subparam_sizes == (q_size, k_size, v_size)
        assert layer.weight.shape == ((q_size + k_size + v_size) // 2, hidden_dim)

        layer.gather_params([layer.weight, layer.bias])
        torch.testing.assert_close(layer.weight.data, full_weight)
        torch.testing.assert_close(layer.bias.data, full_bias)

    def test_gqa_uneven_qkv_fused_forward(self):
        skip_on_device()
        groups._init_tp_mesh_device(tensor_model_parallel_size=2)

        hidden_dim = 8
        q_size, k_size, v_size = 8, 4, 4
        torch.manual_seed(321)
        linear = nn.Linear(hidden_dim,
                           q_size + k_size + v_size,
                           bias=True,
                           dtype=preferred_dtype(),
                           device=get_accelerator().current_device())
        layer = SubParamLinearLayer(deepcopy(linear),
                                    groups.get_tensor_model_parallel_group(),
                                    shape=((q_size, k_size, v_size), -1),
                                    partition_dim=0,
                                    name="self_attn.qkv_proj")

        torch.manual_seed(42)
        inputs = torch.randn(2, hidden_dim, dtype=preferred_dtype(), device=get_accelerator().current_device())
        full_output = linear(inputs)
        tp_output = layer(inputs)

        gathered_output = gather_subparam_output(tp_output, (q_size, k_size, v_size),
                                                 groups.get_tensor_model_parallel_group())
        assert_close_for_preferred_dtype(gathered_output, full_output)
