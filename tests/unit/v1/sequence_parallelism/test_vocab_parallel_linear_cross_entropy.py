# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import pytest
import torch
import torch.nn.functional as F

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.module_inject.tp_shard import AutoTPMeta, get_shard_size_list
from deepspeed.sequence.linear_cross_entropy import vocab_parallel_linear_cross_entropy
from unit.common import DistributedTest


def _inputs(vocab_size=17, hidden_size=8, device="cpu", dtype=torch.float32, bias=True):
    torch.manual_seed(7)
    hidden = torch.randn(2, 5, hidden_size, device=device, dtype=dtype)
    weight = torch.randn(vocab_size, hidden_size, device=device, dtype=dtype)
    bias_tensor = torch.randn(vocab_size, device=device, dtype=dtype) if bias else None
    target = torch.randint(0, vocab_size, (2, 5), device=device)
    target[0, 1] = -100
    target[1, 4] = -100
    return hidden, weight, bias_tensor, target


def _leaf(tensor):
    return None if tensor is None else tensor.detach().clone().requires_grad_(True)


def _reference(hidden, weight, bias, target, reduction):
    logits = F.linear(hidden.float(), weight.float(), None if bias is None else bias.float())
    return F.cross_entropy(logits.view(-1, weight.shape[0]), target.view(-1), reduction=reduction)


@pytest.mark.parametrize("reduction", ["sum", "mean"])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("chunk_size", [None, 1, 3])
def test_matches_linear_then_cross_entropy(reduction, use_bias, chunk_size):
    hidden, weight, bias, target = _inputs(bias=use_bias)
    ref_args = [_leaf(hidden), _leaf(weight), _leaf(bias)]
    args = [_leaf(hidden), _leaf(weight), _leaf(bias)]

    expected = _reference(*ref_args, target, reduction)
    actual = vocab_parallel_linear_cross_entropy(args[0],
                                                 args[1],
                                                 target,
                                                 bias=args[2],
                                                 reduction=reduction,
                                                 chunk_size=chunk_size)
    torch.testing.assert_close(actual, expected)

    # A non-unit upstream gradient must rescale the gradients precomputed in forward.
    (actual * 3).backward()
    (expected * 3).backward()
    for got, want in zip(args, ref_args):
        if want is not None:
            torch.testing.assert_close(got.grad, want.grad)


def test_all_ignored_is_zero():
    hidden, weight, _, target = _inputs(bias=False)
    hidden, weight = _leaf(hidden), _leaf(weight)
    loss = vocab_parallel_linear_cross_entropy(hidden, weight, torch.full_like(target, -100))

    torch.testing.assert_close(loss, torch.zeros_like(loss))
    loss.backward()
    torch.testing.assert_close(hidden.grad, torch.zeros_like(hidden))
    torch.testing.assert_close(weight.grad, torch.zeros_like(weight))


def test_frozen_weight_only_produces_hidden_grad():
    hidden, weight, _, target = _inputs(bias=False)
    hidden = _leaf(hidden)
    vocab_parallel_linear_cross_entropy(hidden, weight, target).backward()

    reference_hidden = _leaf(hidden)
    _reference(reference_hidden, weight, None, target, "mean").backward()
    torch.testing.assert_close(hidden.grad, reference_hidden.grad)
    assert weight.grad is None


def test_validates_inputs():
    hidden, weight, _, target = _inputs(bias=False)
    with pytest.raises(ValueError, match="Unsupported reduction"):
        vocab_parallel_linear_cross_entropy(hidden, weight, target, reduction="none")
    with pytest.raises(ValueError, match="matching non-hidden dimensions"):
        vocab_parallel_linear_cross_entropy(hidden, weight, target[:, :-1])
    with pytest.raises(ValueError, match="local_vocab, hidden"):
        vocab_parallel_linear_cross_entropy(hidden, weight[:, :-1], target)
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        vocab_parallel_linear_cross_entropy(hidden, weight, target, chunk_size=0)
    with pytest.raises(ValueError, match="out of range"):
        vocab_parallel_linear_cross_entropy(hidden, weight, torch.full_like(target, 17))


class TestVocabParallelLinearCrossEntropyTP(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_uneven_vocab_matches_full_vocab(self, dtype):
        device = torch.device(get_accelerator().current_device_name())
        rank = dist.get_rank()
        vocab_size = 17
        partition_sizes = get_shard_size_list(vocab_size, self.world_size, AutoTPMeta(), "lm_head")
        start = sum(partition_sizes[:rank])
        end = start + partition_sizes[rank]

        hidden, weight, bias, target = _inputs(vocab_size, device=device, dtype=dtype)
        ref_args = [_leaf(hidden), _leaf(weight), _leaf(bias)]
        args = [_leaf(hidden), _leaf(weight[start:end]), _leaf(bias[start:end])]

        expected = _reference(*ref_args, target, "mean")
        actual = vocab_parallel_linear_cross_entropy(args[0],
                                                     args[1],
                                                     target,
                                                     bias=args[2],
                                                     tp_group=dist.get_world_group(),
                                                     vocab_start_index=start,
                                                     vocab_end_index=end,
                                                     chunk_size=3)
        actual.backward()
        expected.backward()

        tolerance = {} if dtype == torch.float32 else dict(atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(actual, expected, **tolerance)
        torch.testing.assert_close(args[0].grad, ref_args[0].grad, **tolerance)
        torch.testing.assert_close(args[1].grad, ref_args[1].grad[start:end], **tolerance)
        torch.testing.assert_close(args[2].grad, ref_args[2].grad[start:end], **tolerance)
