# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from torch import nn

import deepspeed.comm as dist


class _VocabSequenceParallelCrossEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, vocab_seq_parallel_logits, target, sp_group):
        # vocab_seq_parallel_logits: [S/P, B, V]
        # target: [S/P, B]
        # return: [S, B]

        # Need softmax for backward
        softmax = torch.nn.functional.softmax(vocab_seq_parallel_logits, dim=-1)
        ctx.vocab_size = vocab_seq_parallel_logits.size(2)
        loss = torch.nn.functional.nll_loss(softmax.log().view(-1, ctx.vocab_size), target.view(-1), reduction='none')

        sp_world_size = dist.get_world_size(sp_group)
        sp_rank = dist.get_rank(sp_group)
        ctx.sp_world_size = sp_world_size
        ctx.sp_rank = sp_rank
        ctx.seqlen = vocab_seq_parallel_logits.size(0) * sp_world_size
        batch_size = vocab_seq_parallel_logits.size(1)

        loss_all = torch.empty(ctx.seqlen,
                               batch_size,
                               dtype=vocab_seq_parallel_logits.dtype,
                               device=vocab_seq_parallel_logits.device)
        dist.all_gather_into_tensor(loss_all, loss, group=sp_group)

        ctx.save_for_backward(softmax, target)

        return loss_all

    @staticmethod
    def backward(ctx, grad_output):
        softmax, target = ctx.saved_tensors

        step_seqlen = ctx.seqlen // ctx.sp_world_size
        sp_rank = ctx.sp_rank
        grad_output_part = grad_output[step_seqlen * sp_rank:step_seqlen * (sp_rank + 1), :]

        grad_input = softmax
        grad_2d = grad_input.view(-1, ctx.vocab_size)
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)

        grad_2d[arange_1d, target.view(-1)] -= 1
        grad_input.mul_(grad_output_part.unsqueeze(dim=-1))

        return grad_input, None, None, None


def vocab_sequence_parallel_cross_entropy(vocab_parallel_logits, target, sp_group):
    return _VocabSequenceParallelCrossEntropy.apply(vocab_parallel_logits, target, sp_group)


class _VocabParallelCrossEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, vocab_parallel_logits, target, tp_group, vocab_start_index, ignore_index):
        if vocab_parallel_logits.shape[:-1] != target.shape:
            raise ValueError("vocab_parallel_logits and target must have matching non-vocabulary dimensions")

        vocab_size = vocab_parallel_logits.shape[-1]
        tp_world_size = dist.get_world_size(tp_group) if tp_group is not None else 1
        tp_rank = dist.get_rank(tp_group) if tp_group is not None else 0
        if vocab_start_index is None:
            vocab_start_index = tp_rank * vocab_size

        target = target.to(dtype=torch.long)
        logits = vocab_parallel_logits.float()
        local_max = logits.detach().amax(dim=-1)
        global_max = local_max.clone()
        if tp_world_size > 1:
            dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=tp_group)

        exp_logits = torch.exp(logits - global_max.unsqueeze(-1))
        local_sum_exp = exp_logits.sum(dim=-1)
        global_sum_exp = local_sum_exp.detach().clone()
        if tp_world_size > 1:
            dist.all_reduce(global_sum_exp, op=dist.ReduceOp.SUM, group=tp_group)

        vocab_end_index = vocab_start_index + vocab_size
        valid_target = target != ignore_index
        target_in_partition = valid_target & (target >= vocab_start_index) & (target < vocab_end_index)
        local_target = (target - vocab_start_index).clamp(min=0, max=vocab_size - 1)
        target_logits = logits.gather(-1, local_target.unsqueeze(-1)).squeeze(-1)
        target_logits = torch.where(target_in_partition, target_logits, torch.zeros_like(target_logits))
        if tp_world_size > 1:
            dist.all_reduce(target_logits, op=dist.ReduceOp.SUM, group=tp_group)

        loss = torch.log(global_sum_exp) + global_max - target_logits
        loss = torch.where(valid_target, loss, torch.zeros_like(loss))

        ctx.save_for_backward(exp_logits, global_sum_exp, local_target, target_in_partition, valid_target)
        ctx.logits_dtype = vocab_parallel_logits.dtype
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        exp_logits, global_sum_exp, local_target, target_in_partition, valid_target = ctx.saved_tensors

        grad_logits = exp_logits / global_sum_exp.unsqueeze(-1)
        grad_logits.scatter_add_(-1, local_target.unsqueeze(-1),
                                 -target_in_partition.unsqueeze(-1).to(dtype=grad_logits.dtype))
        grad_logits *= grad_output.to(dtype=grad_logits.dtype).unsqueeze(-1)
        grad_logits *= valid_target.unsqueeze(-1).to(dtype=grad_logits.dtype)

        return grad_logits.to(dtype=ctx.logits_dtype), None, None, None, None


def _liger_vocab_parallel_cross_entropy(vocab_parallel_logits, target, tp_group, ignore_index, label_smoothing):
    try:
        from liger_kernel.transformers import LigerVocabParallelCrossEntropy
    except ImportError:
        try:
            from liger_kernel.transformers.vocab_parallel_cross_entropy import LigerVocabParallelCrossEntropy
        except ImportError as exc:
            raise ImportError("Liger Kernel is required for use_liger=True. "
                              "Install it with `pip install liger-kernel`.") from exc

    if vocab_parallel_logits.dim() != 3 or target.dim() != 2:
        raise ValueError("Liger vocab-parallel cross entropy expects logits [batch, seq, vocab] "
                         "and targets [batch, seq]")

    loss_fn = LigerVocabParallelCrossEntropy(ignore_index=ignore_index,
                                              label_smoothing=label_smoothing,
                                              reduction="none")
    loss = loss_fn(vocab_parallel_logits.transpose(0, 1).contiguous(), target.transpose(0, 1).contiguous(), tp_group)
    return loss.transpose(0, 1).contiguous()


def vocab_parallel_cross_entropy(vocab_parallel_logits,
                                 target,
                                 tp_group=None,
                                 vocab_start_index=None,
                                 ignore_index=-100,
                                 reduction="mean",
                                 use_liger=False,
                                 label_smoothing=0.0):
    """Compute cross entropy over vocabulary-sharded logits.

    ``vocab_parallel_logits`` contains only the local vocabulary shard while
    ``target`` contains global vocabulary indices.  The implementation reduces
    the per-token max, sum-exp, and target logit across ``tp_group`` without
    materializing a full-vocabulary logits tensor.
    """
    if reduction not in ("none", "sum", "mean"):
        raise ValueError(f"Unsupported reduction: {reduction}")
    if label_smoothing and not use_liger:
        raise ValueError("label_smoothing requires use_liger=True")

    if use_liger:
        loss = _liger_vocab_parallel_cross_entropy(vocab_parallel_logits, target, tp_group, ignore_index,
                                                   label_smoothing)
    else:
        loss = _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target, tp_group, vocab_start_index,
                                                ignore_index)

    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()

    valid_tokens = (target != ignore_index).sum().clamp_min(1)
    return loss.sum() / valid_tokens.to(dtype=loss.dtype)


class VocabParallelCrossEntropyLoss(nn.Module):
    """Module wrapper for ``vocab_parallel_cross_entropy``."""

    def __init__(self,
                 tp_group=None,
                 vocab_start_index=None,
                 ignore_index=-100,
                 reduction="mean",
                 use_liger=False,
                 label_smoothing=0.0):
        super().__init__()
        self.tp_group = tp_group
        self.vocab_start_index = vocab_start_index
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.use_liger = use_liger
        self.label_smoothing = label_smoothing

    def forward(self, vocab_parallel_logits, target):
        return vocab_parallel_cross_entropy(vocab_parallel_logits,
                                            target,
                                            tp_group=self.tp_group,
                                            vocab_start_index=self.vocab_start_index,
                                            ignore_index=self.ignore_index,
                                            reduction=self.reduction,
                                            use_liger=self.use_liger,
                                            label_smoothing=self.label_smoothing)


class VocabParallelCausalLMLoss(nn.Module):
    """Causal-LM loss adapter compatible with Hugging Face ``loss_function``."""

    def __init__(self, tp_group=None, use_liger=False, ignore_index=-100):
        super().__init__()
        self.tp_group = tp_group
        self.use_liger = use_liger
        self.ignore_index = ignore_index

    def forward(self,
                logits,
                labels=None,
                vocab_size=None,
                shift_labels=None,
                num_items_in_batch=None,
                **kwargs):
        if shift_labels is None:
            if labels is None:
                raise ValueError("labels or shift_labels must be provided")
            shift_labels = labels[..., 1:].contiguous()
            logits = logits[..., :-1, :].contiguous()
        else:
            shift_labels = shift_labels.contiguous()

        if vocab_size is not None:
            tp_world_size = dist.get_world_size(self.tp_group) if self.tp_group is not None else 1
            expected_vocab_size = logits.shape[-1] * tp_world_size
            if expected_vocab_size != vocab_size:
                raise ValueError(f"Vocab-parallel logits have local vocab size {logits.shape[-1]} with TP size "
                                 f"{tp_world_size}, but vocab_size={vocab_size}")

        reduction = "sum" if num_items_in_batch is not None else "mean"
        loss = vocab_parallel_cross_entropy(logits,
                                            shift_labels,
                                            tp_group=self.tp_group,
                                            ignore_index=self.ignore_index,
                                            reduction=reduction,
                                            use_liger=self.use_liger)
        if num_items_in_batch is not None:
            loss = loss / torch.as_tensor(num_items_in_batch, device=loss.device, dtype=loss.dtype).clamp_min(1)
        return loss


def configure_vocab_parallel_loss(model, tp_group=None, use_liger=False, ignore_index=-100):
    """Install a vocab-parallel causal-LM loss on models exposing ``loss_function``."""
    loss_fn = VocabParallelCausalLMLoss(tp_group=tp_group, use_liger=use_liger, ignore_index=ignore_index)

    if not hasattr(model, "loss_function"):
        raise ValueError("vocab_parallel_lm_head requires a model exposing a loss_function hook")

    original_loss_fn = model.loss_function
    if not hasattr(model, "_deepspeed_original_loss_function"):
        model._deepspeed_original_loss_function = original_loss_fn
    model._loss_function = loss_fn
    try:
        if model.loss_function is not loss_fn:
            model.loss_function = loss_fn
    except AttributeError:
        pass

    if model.loss_function is not loss_fn:
        raise ValueError("Unable to install the vocab-parallel loss_function hook on this model")
    return model
