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


def _default_chunk_rows(num_tokens, local_vocab_size, hidden_size):
    # Keep each fp32 logits block about the size of the hidden states, which bounds the
    # extra peak memory independently of the vocabulary size.
    num_chunks = max(1, math.ceil(local_vocab_size / hidden_size))
    return max(1, math.ceil(num_tokens / num_chunks))


class _VocabParallelLinearCrossEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, hidden, weight, bias, target, tp_group, vocab_start_index, vocab_end_index, ignore_index,
                grad_scale, chunk_rows):
        num_tokens = hidden.shape[0]
        local_vocab_size = weight.shape[0]
        tp_active = _tp_active(tp_group)

        loss = torch.zeros(num_tokens, dtype=torch.float32, device=hidden.device)
        grad_hidden = torch.empty_like(hidden)
        grad_weight = torch.zeros_like(weight) if weight.requires_grad else None
        grad_bias = torch.zeros_like(bias) if bias is not None and bias.requires_grad else None

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
            grad_logits = grad_logits.to(hidden.dtype)

            torch.matmul(grad_logits, weight, out=grad_hidden[start:end])
            if grad_weight is not None:
                grad_weight.addmm_(grad_logits.t(), hidden_chunk)
            if grad_bias is not None:
                grad_bias.add_(grad_logits.sum(dim=0).to(grad_bias.dtype))

        ctx.save_for_backward(grad_hidden, grad_weight, grad_bias)
        ctx.tp_group = tp_group
        return loss.sum() * grad_scale

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grad_hidden, grad_weight, grad_bias = ctx.saved_tensors
        # The precomputed gradients assume a unit upstream gradient; rescale only when needed
        # so the common ``loss.backward()`` path does not allocate another copy.
        if not torch.equal(grad_output, torch.ones_like(grad_output)):
            grad_hidden = grad_hidden * grad_output.to(grad_hidden.dtype)
            grad_weight = grad_weight * grad_output.to(grad_weight.dtype) if grad_weight is not None else None
            grad_bias = grad_bias * grad_output.to(grad_bias.dtype) if grad_bias is not None else None
        # Each rank only saw its vocabulary shard, so the input gradient is a partial sum, the
        # same reduction the column-parallel LM head performs in its own backward.
        if _tp_active(ctx.tp_group):
            dist.all_reduce(grad_hidden, op=dist.ReduceOp.SUM, group=ctx.tp_group)
        return grad_hidden, grad_weight, grad_bias, None, None, None, None, None, None, None


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
    logits chunk is about as large as ``hidden``.
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
        chunk_size = _default_chunk_rows(num_tokens, weight.shape[0], hidden.shape[1])

    loss = _VocabParallelLinearCrossEntropy.apply(hidden, weight, bias, target, tp_group, vocab_start_index,
                                                  vocab_end_index, ignore_index, grad_scale, chunk_size)
    return _global_sp_sum(loss, sp_group)
