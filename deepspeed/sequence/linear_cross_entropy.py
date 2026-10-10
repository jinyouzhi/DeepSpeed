# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Fused LM-head projection and cross entropy over a vocabulary-parallel weight.

A vocabulary-parallel LM head still materializes ``[tokens, vocab / tp]`` logits per rank.
This module fuses the projection into the loss: tokens are processed in row chunks, so only
one ``[chunk, vocab / tp]`` logits block is alive at a time, and the input and weight
gradients are accumulated during forward.
"""

import math

import torch
from torch.autograd.function import once_differentiable

import deepspeed.comm as dist
from deepspeed.sequence.cross_entropy import _global_sp_sum, _resolve_vocab_metadata


def _tp_active(tp_group):
    return tp_group is not None and dist.get_world_size(tp_group) > 1


# Every chunk reads and writes the whole fp32 weight-gradient buffer. Below about this many
# rows per chunk, that traffic outweighs the chunk's GEMMs and the loss becomes bandwidth-bound.
_MIN_CHUNK_ROWS = 1024


def _default_chunk_rows(num_tokens, global_vocab_size, tp_world_size, hidden_size):
    # Keep each fp32 logits block about the size of the hidden states, which bounds the
    # extra peak memory independently of the vocabulary size. Derived from the global shard
    # layout rather than the local shard so that every TP rank issues the same collectives
    # even when the vocabulary is split unevenly.
    max_local_vocab_size = math.ceil(global_vocab_size / tp_world_size)
    num_chunks = max(1, math.ceil(max_local_vocab_size / hidden_size))
    chunk_rows = max(math.ceil(num_tokens / num_chunks), _MIN_CHUNK_ROWS)
    return max(1, min(chunk_rows, num_tokens))


_MIXED_ADDMM_UNSUPPORTED = set()


def _accumulate_matmul(accumulator, mat1, mat2):
    """``accumulator += mat1 @ mat2`` with ``accumulator`` kept in its own (fp32) dtype.

    Accumulating chunk gradients in a low-precision buffer drops later contributions once the
    sum grows, so the gradient would depend on the chunk size.
    """
    if mat1.dtype == accumulator.dtype:
        accumulator.addmm_(mat1, mat2)
        return
    device_type = accumulator.device.type
    if device_type not in _MIXED_ADDMM_UNSUPPORTED:
        try:
            torch.addmm(accumulator, mat1, mat2, out_dtype=accumulator.dtype, out=accumulator)
            return
        except (RuntimeError, TypeError, NotImplementedError):
            # Older PyTorch or backends without a mixed-precision GEMM (e.g. CPU).
            _MIXED_ADDMM_UNSUPPORTED.add(device_type)
    accumulator.add_(torch.mm(mat1, mat2))


class _VocabParallelLinearCrossEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, hidden, weight, bias, target, tp_group, vocab_start_index, vocab_end_index, ignore_index,
                grad_scale, chunk_rows, compute_dtype):
        num_tokens = hidden.shape[0]
        local_vocab_size = weight.shape[0]
        tp_active = _tp_active(tp_group)
        ctx.input_dtypes = (hidden.dtype, weight.dtype, None if bias is None else bias.dtype)

        # Mirror what autocast would do for ``F.linear``: run the GEMMs in one compute dtype even
        # when activations and parameters differ (e.g. bf16 activations with fp32 weights).
        weight_requires_grad = weight.requires_grad
        bias_requires_grad = bias is not None and bias.requires_grad
        hidden = hidden.to(compute_dtype)
        weight = weight.to(compute_dtype)
        bias = None if bias is None else bias.to(compute_dtype)

        loss = torch.zeros(num_tokens, dtype=torch.float32, device=hidden.device)
        grad_hidden = torch.empty_like(hidden)
        # Parameter gradients sum over every chunk, so they are accumulated in fp32.
        grad_weight = torch.zeros_like(weight, dtype=torch.float32) if weight_requires_grad else None
        grad_bias = torch.zeros_like(bias, dtype=torch.float32) if bias_requires_grad else None

        for start in range(0, num_tokens, chunk_rows):
            end = min(start + chunk_rows, num_tokens)
            hidden_chunk = hidden[start:end]
            target_chunk = target[start:end]

            logits = hidden_chunk @ weight.t()
            if bias is not None:
                logits = logits + bias
            logits = logits.float()

            valid = target_chunk != ignore_index
            in_shard = valid & (target_chunk >= vocab_start_index) & (target_chunk < vocab_end_index)
            local_target = (target_chunk - vocab_start_index).clamp(0, local_vocab_size - 1)

            row_max = logits.amax(dim=-1)
            if tp_active:
                dist.all_reduce(row_max, op=dist.ReduceOp.MAX, group=tp_group)
            logits.sub_(row_max.unsqueeze(-1))
            target_logit = logits.gather(-1, local_target.unsqueeze(-1)).squeeze(-1)
            target_logit = torch.where(in_shard, target_logit, torch.zeros_like(target_logit))
            # Reuse the logits buffer for the softmax so only one fp32 block is alive per chunk.
            probs = logits.exp_()
            stats = torch.stack([probs.sum(dim=-1), target_logit], dim=-1)
            if tp_active:
                dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=tp_group)
            sum_exp, target_logit = stats.unbind(-1)

            loss[start:end] = torch.where(valid, sum_exp.log() - target_logit, torch.zeros_like(sum_exp))

            # d(loss) / d(logits) = softmax - onehot, zeroed for ignored tokens and pre-scaled by
            # the reduction factor so backward only has to multiply by the incoming gradient.
            row_scale = valid.to(probs.dtype) * grad_scale / sum_exp
            grad_logits = probs.mul_(row_scale.unsqueeze(-1))
            grad_logits.scatter_add_(-1, local_target.unsqueeze(-1),
                                     -(in_shard.to(probs.dtype) * grad_scale).unsqueeze(-1))
            if grad_bias is not None:
                grad_bias.add_(grad_logits.sum(dim=0))
            grad_logits = grad_logits.to(compute_dtype)

            torch.matmul(grad_logits, weight, out=grad_hidden[start:end])
            if grad_weight is not None:
                _accumulate_matmul(grad_weight, grad_logits.t(), hidden_chunk)

        ctx.save_for_backward(grad_hidden, grad_weight, grad_bias)
        ctx.tp_group = tp_group
        return loss.sum() * grad_scale

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grad_hidden, grad_weight, grad_bias = ctx.saved_tensors
        hidden_dtype, weight_dtype, bias_dtype = ctx.input_dtypes
        # The precomputed gradients assume a unit upstream gradient. Rescale in fp32 so that a
        # large loss scale (e.g. 65536 with fp16) does not overflow before it is applied.
        scale = grad_output.float()
        unit_scale = torch.equal(scale, torch.ones_like(scale))

        def finalize(grad, dtype):
            if grad is None:
                return None
            if unit_scale:
                return grad.to(dtype)
            return (grad.float() * scale).to(dtype)

        grad_hidden = finalize(grad_hidden, hidden_dtype)
        grad_weight = finalize(grad_weight, weight_dtype)
        grad_bias = finalize(grad_bias, bias_dtype)
        # Each rank only saw its vocabulary shard, so the input gradient is a partial sum, the
        # same reduction the column-parallel LM head performs in its own backward.
        if _tp_active(ctx.tp_group):
            dist.all_reduce(grad_hidden, op=dist.ReduceOp.SUM, group=ctx.tp_group)
        return grad_hidden, grad_weight, grad_bias, None, None, None, None, None, None, None, None


def vocab_parallel_linear_cross_entropy(hidden,
                                        weight,
                                        target,
                                        bias=None,
                                        tp_group=None,
                                        sp_group=None,
                                        vocab_start_index=None,
                                        vocab_end_index=None,
                                        ignore_index=-100,
                                        reduction="mean",
                                        chunk_size=None):
    """Cross entropy of ``hidden @ weight.T (+ bias)`` without materializing the logits.

    ``weight`` (and ``bias``) hold this rank's rows of a vocabulary-parallel LM head, for
    example ``VocabParallelLinear.weight``. ``hidden`` must be identical on every TP rank, as
    for a column-parallel layer; its gradient is all-reduced over ``tp_group`` in backward.

    Gradients are computed during forward, so only a single first-order backward is
    supported. ``chunk_size`` sets the number of tokens per chunk; by default each fp32
    logits chunk is about as large as ``hidden``, with at least 1024 tokens per chunk.
    """
    if reduction not in ("sum", "mean"):
        raise ValueError(f"Unsupported reduction: {reduction}")
    if hidden.shape[:-1] != target.shape:
        raise ValueError("hidden and target must have matching non-hidden dimensions")
    if weight.dim() != 2 or weight.shape[1] != hidden.shape[-1]:
        raise ValueError("weight must have shape [local_vocab, hidden]")
    if (vocab_start_index is None) != (vocab_end_index is None):
        raise ValueError("vocab_start_index and vocab_end_index must be provided together")
    if chunk_size is not None and chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    vocab_start_index, vocab_end_index, global_vocab_size = _resolve_vocab_metadata(
        weight.shape[0], vocab_start_index, vocab_end_index, tp_group, hidden.device)
    target = target.reshape(-1).to(dtype=torch.long)
    invalid_target = (target != ignore_index) & ((target < 0) | (target >= global_vocab_size))
    if invalid_target.any().item():
        raise ValueError(f"Target is out of range for vocabulary size {global_vocab_size}")

    hidden = hidden.reshape(-1, hidden.shape[-1]).contiguous()
    weight = weight.contiguous()
    num_tokens = hidden.shape[0]

    if reduction == "mean":
        valid_tokens = (target != ignore_index).sum().to(torch.float32)
        if sp_group is not None and dist.get_world_size(sp_group) > 1:
            dist.all_reduce(valid_tokens, op=dist.ReduceOp.SUM, group=sp_group)
        grad_scale = 1.0 / max(valid_tokens.item(), 1.0)
    else:
        grad_scale = 1.0
    if chunk_size is None:
        tp_world_size = dist.get_world_size(tp_group) if tp_group is not None else 1
        chunk_size = _default_chunk_rows(num_tokens, global_vocab_size, tp_world_size, hidden.shape[1])
    device_type = hidden.device.type
    if torch.is_autocast_enabled(device_type):
        compute_dtype = torch.get_autocast_dtype(device_type)
    else:
        compute_dtype = torch.promote_types(hidden.dtype, weight.dtype)

    with torch.autocast(device_type, enabled=False):
        loss = _VocabParallelLinearCrossEntropy.apply(hidden, weight, bias, target, tp_group, vocab_start_index,
                                                      vocab_end_index, ignore_index, grad_scale, chunk_size,
                                                      compute_dtype)
    return _global_sp_sum(loss, sp_group)
