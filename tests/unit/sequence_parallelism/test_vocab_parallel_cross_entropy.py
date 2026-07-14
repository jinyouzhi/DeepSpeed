# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import torch.nn.functional as F

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.module_inject.tp_shard import get_shard_size_list
from deepspeed.sequence.cross_entropy import vocab_parallel_cross_entropy
from unit.common import DistributedTest


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_vocab_parallel_cross_entropy_tp1_matches_torch(reduction):
    torch.manual_seed(42)
    logits = torch.randn(2, 3, 17, requires_grad=True)
    reference_logits = logits.detach().clone().requires_grad_(True)
    target = torch.tensor([[0, 8, -100], [16, 7, 9]])

    expected = F.cross_entropy(reference_logits.view(-1, 17), target.view(-1), reduction=reduction, ignore_index=-100)
    if reduction == "none":
        expected = expected.view_as(target)
    actual = vocab_parallel_cross_entropy(logits, target, reduction=reduction)

    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad)


def test_vocab_parallel_cross_entropy_all_ignored_is_zero():
    logits = torch.randn(2, 3, 11, requires_grad=True)
    target = torch.full((2, 3), -100)

    loss = vocab_parallel_cross_entropy(logits, target)

    torch.testing.assert_close(loss, torch.zeros_like(loss))
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


def test_vocab_parallel_cross_entropy_validates_inputs():
    logits = torch.randn(2, 3, 11)
    target = torch.zeros(2, 3, dtype=torch.long)

    with pytest.raises(ValueError, match="matching non-vocabulary dimensions"):
        vocab_parallel_cross_entropy(logits, target[:, :-1])
    with pytest.raises(ValueError, match="Unsupported reduction"):
        vocab_parallel_cross_entropy(logits, target, reduction="batchmean")
    with pytest.raises(ValueError, match="only supported with reduction='none'"):
        vocab_parallel_cross_entropy(logits, target, reduction="mean", gather_sequence_loss=True)
    with pytest.raises(ValueError, match="Vocabulary shard bounds"):
        vocab_parallel_cross_entropy(logits, target, vocab_start_index=0, vocab_end_index=10)
    target[0, 0] = 11
    with pytest.raises(ValueError, match="out of range"):
        vocab_parallel_cross_entropy(logits, target)


class TestVocabParallelCrossEntropyTP(DistributedTest):
    world_size = 2

    def test_uneven_vocab_matches_torch(self):
        device = get_accelerator().current_device()
        rank = dist.get_rank()
        partition_sizes = get_shard_size_list(17, self.world_size, "lm_head")
        vocab_start_index = sum(partition_sizes[:rank])
        vocab_end_index = vocab_start_index + partition_sizes[rank]

        torch.manual_seed(123)
        full_logits = torch.randn(2, 3, 17, device=device)
        reference_logits = full_logits.detach().clone().requires_grad_(True)
        local_logits = full_logits[..., vocab_start_index:vocab_end_index].detach().clone().requires_grad_(True)
        target = torch.tensor([[0, 8, -100], [16, 7, 9]], device=device)

        expected = F.cross_entropy(reference_logits.view(-1, 17), target.view(-1))
        actual = vocab_parallel_cross_entropy(local_logits,
                                              target,
                                              tp_group=dist.get_world_group(),
                                              vocab_start_index=vocab_start_index,
                                              vocab_end_index=vocab_end_index)

        torch.testing.assert_close(actual, expected)
        actual.backward()
        expected.backward()
        torch.testing.assert_close(local_logits.grad, reference_logits.grad[..., vocab_start_index:vocab_end_index])

    def test_rejects_non_contiguous_vocab_shards(self):
        device = get_accelerator().current_device()
        rank = dist.get_rank()
        target = torch.zeros(2, 3, dtype=torch.long, device=device)

        gap_start, gap_end = ((0, 8), (9, 17))[rank]
        gap_logits = torch.randn(2, 3, gap_end - gap_start, device=device)
        with pytest.raises(ValueError, match="contiguous, non-overlapping"):
            vocab_parallel_cross_entropy(gap_logits,
                                         target,
                                         tp_group=dist.get_world_group(),
                                         vocab_start_index=gap_start,
                                         vocab_end_index=gap_end)

        overlap_start, overlap_end = ((0, 9), (8, 17))[rank]
        overlap_logits = torch.randn(2, 3, overlap_end - overlap_start, device=device)
        with pytest.raises(ValueError, match="contiguous, non-overlapping"):
            vocab_parallel_cross_entropy(overlap_logits,
                                         target,
                                         tp_group=dist.get_world_group(),
                                         vocab_start_index=overlap_start,
                                         vocab_end_index=overlap_end)


class TestVocabParallelCrossEntropySP(DistributedTest):
    world_size = 2

    def test_sequence_loss_local_gathered_and_mean(self):
        device = get_accelerator().current_device()
        rank = dist.get_rank()
        local_sequence_size = 2

        torch.manual_seed(456)
        full_logits = torch.randn(4, 2, 13, device=device)
        target = torch.tensor([[0, 1], [2, -100], [12, 3], [4, 5]], device=device)
        sequence_start = rank * local_sequence_size
        sequence_end = sequence_start + local_sequence_size
        local_logits = full_logits[sequence_start:sequence_end].detach().clone().requires_grad_(True)
        local_target = target[sequence_start:sequence_end]

        expected_none = F.cross_entropy(full_logits.view(-1, 13), target.view(-1), reduction="none").view_as(target)
        actual_local = vocab_parallel_cross_entropy(local_logits, local_target, reduction="none")
        actual_gathered = vocab_parallel_cross_entropy(local_logits,
                                                       local_target,
                                                       sp_group=dist.get_world_group(),
                                                       reduction="none",
                                                       gather_sequence_loss=True)
        actual_mean = vocab_parallel_cross_entropy(local_logits,
                                                   local_target,
                                                   sp_group=dist.get_world_group(),
                                                   reduction="mean")

        torch.testing.assert_close(actual_local, expected_none[sequence_start:sequence_end])
        torch.testing.assert_close(actual_gathered, expected_none)
        torch.testing.assert_close(actual_mean, expected_none.sum() / (target != -100).sum())

    def test_gathered_sequence_loss_backward_accumulates_all_ranks(self):
        device = get_accelerator().current_device()
        rank = dist.get_rank()

        torch.manual_seed(654)
        local_logits = torch.randn(2, 3, 13, device=device, requires_grad=True)
        reference_logits = local_logits.detach().clone().requires_grad_(True)
        local_target = torch.tensor([[0, 1, 2], [3, 4, 5]], device=device)

        gathered_loss = vocab_parallel_cross_entropy(local_logits,
                                                     local_target,
                                                     sp_group=dist.get_world_group(),
                                                     reduction="none",
                                                     gather_sequence_loss=True)
        rank_weight = rank + 1
        (gathered_loss * rank_weight).sum().backward()

        total_weight = sum(range(1, self.world_size + 1))
        reference_loss = F.cross_entropy(reference_logits.view(-1, 13), local_target.view(-1), reduction="sum")
        (reference_loss * total_weight).backward()
        torch.testing.assert_close(local_logits.grad, reference_logits.grad)


class TestVocabParallelCrossEntropyTPAndSP(DistributedTest):
    world_size = 4

    def test_orthogonal_groups_match_torch(self):
        device = get_accelerator().current_device()
        rank = dist.get_rank()
        tp_groups = [dist.new_group(ranks=[0, 1]), dist.new_group(ranks=[2, 3])]
        sp_groups = [dist.new_group(ranks=[0, 2]), dist.new_group(ranks=[1, 3])]
        tp_group = tp_groups[rank // 2]
        sp_group = sp_groups[rank % 2]
        tp_rank = rank % 2
        sp_rank = rank // 2

        partition_sizes = get_shard_size_list(17, 2, "lm_head")
        vocab_start_index = sum(partition_sizes[:tp_rank])
        vocab_end_index = vocab_start_index + partition_sizes[tp_rank]
        sequence_start = sp_rank * 2
        sequence_end = sequence_start + 2

        torch.manual_seed(789)
        full_logits = torch.randn(4, 2, 17, device=device)
        reference_logits = full_logits.detach().clone().requires_grad_(True)
        local_logits = full_logits[sequence_start:sequence_end, ...,
                                   vocab_start_index:vocab_end_index].detach().clone().requires_grad_(True)
        target = torch.tensor([[0, 1], [8, -100], [16, 3], [4, 9]], device=device)
        local_target = target[sequence_start:sequence_end]

        expected = F.cross_entropy(reference_logits.view(-1, 17), target.view(-1))
        actual = vocab_parallel_cross_entropy(local_logits,
                                              local_target,
                                              tp_group=tp_group,
                                              sp_group=sp_group,
                                              vocab_start_index=vocab_start_index,
                                              vocab_end_index=vocab_end_index)

        torch.testing.assert_close(actual, expected)
        actual.backward()
        expected.backward()
        expected_grad = reference_logits.grad[sequence_start:sequence_end, ..., vocab_start_index:vocab_end_index]
        torch.testing.assert_close(local_logits.grad, expected_grad)
