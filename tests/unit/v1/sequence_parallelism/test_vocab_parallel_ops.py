# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import torch.nn.functional as F

import deepspeed.comm as dist
import deepspeed.sequence.vocab_parallel_ops as vocab_parallel_ops
from deepspeed.accelerator import get_accelerator
from deepspeed.module_inject.tp_shard import AutoTPMeta, get_shard_size_list
from deepspeed.sequence.vocab_parallel_ops import vocab_parallel_entropy, vocab_parallel_kl_div, vocab_parallel_logprobs
from unit.common import DistributedTest


def reference_logprobs(logits, index, ignore_index=-100):
    log_probs = F.log_softmax(logits.float(), dim=-1)
    valid = index != ignore_index
    gathered = log_probs.gather(-1, index.clamp(min=0))
    return torch.where(valid, gathered, torch.zeros_like(gathered))


def reference_entropy(logits):
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return -(log_probs.exp() * log_probs).sum(dim=-1)


def reference_kl(student_logits, teacher_logits, reverse):
    log_q = F.log_softmax(student_logits.float(), dim=-1)
    log_p = F.log_softmax(teacher_logits.float(), dim=-1)
    if reverse:
        return F.kl_div(log_p, log_q, reduction="none", log_target=True).sum(dim=-1)
    return F.kl_div(log_q, log_p, reduction="none", log_target=True).sum(dim=-1)


def leaf(tensor):
    return tensor.detach().clone().requires_grad_(True)


@pytest.mark.parametrize("top_k", [None, 3])
def test_logprobs_matches_log_softmax(top_k):
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 19)
    index_shape = (2, 5) if top_k is None else (2, 5, top_k)
    index = torch.randint(0, 19, index_shape)
    index[0, 1] = -100
    actual_logits, expected_logits = leaf(logits), leaf(logits)
    weight = torch.randn(index_shape)

    actual = vocab_parallel_logprobs(actual_logits, index)
    expected_index = index if top_k is not None else index.unsqueeze(-1)
    expected = reference_logprobs(expected_logits, expected_index)
    if top_k is None:
        expected = expected.squeeze(-1)

    torch.testing.assert_close(actual, expected)
    (actual * weight).sum().backward()
    (expected * weight).sum().backward()
    torch.testing.assert_close(actual_logits.grad, expected_logits.grad)


def test_entropy_matches_reference():
    torch.manual_seed(1)
    logits = torch.randn(3, 4, 23) * 3
    actual_logits, expected_logits = leaf(logits), leaf(logits)
    weight = torch.randn(3, 4)

    actual = vocab_parallel_entropy(actual_logits)
    expected = reference_entropy(expected_logits)

    torch.testing.assert_close(actual, expected)
    (actual * weight).sum().backward()
    (expected * weight).sum().backward()
    torch.testing.assert_close(actual_logits.grad, expected_logits.grad)


@pytest.mark.parametrize("reverse", [False, True])
def test_kl_div_matches_reference_for_student_and_teacher(reverse):
    torch.manual_seed(2)
    student = torch.randn(2, 6, 29) * 2
    teacher = torch.randn(2, 6, 29) * 2
    actual_student, actual_teacher = leaf(student), leaf(teacher)
    expected_student, expected_teacher = leaf(student), leaf(teacher)
    weight = torch.randn(2, 6)

    actual = vocab_parallel_kl_div(actual_student, actual_teacher, reverse=reverse)
    expected = reference_kl(expected_student, expected_teacher, reverse)

    torch.testing.assert_close(actual, expected)
    (actual * weight).sum().backward()
    (expected * weight).sum().backward()
    torch.testing.assert_close(actual_student.grad, expected_student.grad)
    torch.testing.assert_close(actual_teacher.grad, expected_teacher.grad)


def test_top_k_teacher_forward_kl():
    # Distillation from a teacher that only exposes top-k (id, logprob) pairs: the student
    # gradient must match the same loss computed on full-vocabulary student log-probabilities.
    torch.manual_seed(3)
    student = torch.randn(2, 4, 31)
    teacher_topk_logprobs, teacher_topk_ids = F.log_softmax(torch.randn(2, 4, 31), dim=-1).topk(5, dim=-1)
    actual_student, expected_student = leaf(student), leaf(student)

    def top_k_forward_kl(student_topk_logprobs):
        return (teacher_topk_logprobs.exp() * (teacher_topk_logprobs - student_topk_logprobs)).sum(dim=-1)

    actual = top_k_forward_kl(vocab_parallel_logprobs(actual_student, teacher_topk_ids))
    expected = top_k_forward_kl(reference_logprobs(expected_student, teacher_topk_ids))

    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(actual_student.grad, expected_student.grad)


def test_row_chunking_matches_reference(monkeypatch):
    # One row per chunk, so every per-token statistic is assembled from several chunks.
    monkeypatch.setattr(vocab_parallel_ops, "_CHUNK_NUMEL", 1)
    torch.manual_seed(6)
    student = torch.randn(3, 4, 13)
    teacher = torch.randn(3, 4, 13)
    index = torch.randint(0, 13, (3, 4, 2))
    actual_student, actual_teacher = leaf(student), leaf(teacher)
    expected_student, expected_teacher = leaf(student), leaf(teacher)

    actual = (vocab_parallel_logprobs(actual_student, index).sum(dim=-1) + vocab_parallel_entropy(actual_student) +
              vocab_parallel_kl_div(actual_student, actual_teacher, reverse=True))
    expected = (reference_logprobs(expected_student, index).sum(dim=-1) + reference_entropy(expected_student) +
                reference_kl(expected_student, expected_teacher, True))

    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(actual_student.grad, expected_student.grad)
    torch.testing.assert_close(actual_teacher.grad, expected_teacher.grad)


def test_empty_token_batch():
    logits = torch.randn(0, 11, requires_grad=True)

    outputs = [
        vocab_parallel_logprobs(logits, torch.zeros(0, dtype=torch.long)),
        vocab_parallel_logprobs(logits, torch.zeros(0, 3, dtype=torch.long)),
        vocab_parallel_entropy(logits),
        vocab_parallel_kl_div(logits, torch.randn(0, 11)),
    ]
    sum(output.sum() for output in outputs).backward()

    assert [tuple(output.shape) for output in outputs] == [(0, ), (0, 3), (0, ), (0, )]
    assert logits.grad.shape == logits.shape


def test_outputs_support_in_place_masking():
    torch.manual_seed(7)
    student = torch.randn(2, 3, 11, requires_grad=True)
    teacher = torch.randn(2, 3, 11)
    mask = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])

    entropy = vocab_parallel_entropy(student)
    kl = vocab_parallel_kl_div(student, teacher, reverse=True)
    (entropy.mul_(mask).sum() + kl.mul_(mask).sum()).backward()

    expected_student = leaf(student)
    ((reference_entropy(expected_student) + reference_kl(expected_student, teacher, True)) * mask).sum().backward()
    torch.testing.assert_close(student.grad, expected_student.grad)


def test_double_backward_raises_instead_of_returning_wrong_values():
    logits = torch.randn(2, 7, requires_grad=True)

    (grad, ) = torch.autograd.grad(vocab_parallel_entropy(logits).sum(), logits, create_graph=True)

    with pytest.raises(RuntimeError):
        grad.sum().backward()


def test_detached_teacher_gets_no_gradient():
    student = torch.randn(2, 3, 11, requires_grad=True)
    teacher = torch.randn(2, 3, 11)

    vocab_parallel_kl_div(student, teacher, reverse=True).sum().backward()

    assert student.grad is not None
    assert teacher.grad is None


def test_bf16_inputs_return_bf16_gradients():
    torch.manual_seed(4)
    student = torch.randn(2, 3, 13, dtype=torch.bfloat16, requires_grad=True)
    teacher = torch.randn(2, 3, 13, dtype=torch.bfloat16)

    loss = vocab_parallel_kl_div(student, teacher)
    loss.sum().backward()

    assert loss.dtype == torch.float32
    assert student.grad.dtype == torch.bfloat16


def test_validates_inputs():
    logits = torch.randn(2, 3, 11)
    with pytest.raises(ValueError, match="non-vocabulary dimensions"):
        vocab_parallel_logprobs(logits, torch.zeros(2, 4, dtype=torch.long))
    with pytest.raises(ValueError, match="out of range"):
        vocab_parallel_logprobs(logits, torch.full((2, 3), 11))
    with pytest.raises(ValueError, match="provided together"):
        vocab_parallel_logprobs(logits, torch.zeros(2, 3, dtype=torch.long), vocab_start_index=0)
    with pytest.raises(ValueError, match="same shape"):
        vocab_parallel_kl_div(logits, torch.randn(2, 3, 12))


class TestVocabParallelOpsTP(DistributedTest):
    world_size = 2

    def test_uneven_shards_match_full_vocabulary(self):
        device = torch.device(get_accelerator().current_device_name())
        rank = dist.get_rank()
        tp_group = dist.get_world_group()
        vocab_size = 17
        partition_sizes = get_shard_size_list(vocab_size, self.world_size, AutoTPMeta(), "lm_head")
        start = sum(partition_sizes[:rank])
        end = start + partition_sizes[rank]

        torch.manual_seed(5)
        student = torch.randn(2, 4, vocab_size, device=device) * 2
        teacher = torch.randn(2, 4, vocab_size, device=device) * 2
        index = torch.randint(0, vocab_size, (2, 4, 3), device=device)
        index[1, 2, 0] = -100
        weight = torch.randn(2, 4, device=device)

        full_student, full_teacher = leaf(student), leaf(teacher)
        local_student = leaf(student[..., start:end])
        local_teacher = leaf(teacher[..., start:end])

        actual = (vocab_parallel_logprobs(
            local_student, index, tp_group=tp_group, vocab_start_index=start, vocab_end_index=end).sum(dim=-1) +
                  vocab_parallel_entropy(local_student, tp_group=tp_group) +
                  vocab_parallel_kl_div(local_student, local_teacher, tp_group=tp_group) +
                  vocab_parallel_kl_div(local_student, local_teacher, reverse=True, tp_group=tp_group))
        expected = (reference_logprobs(full_student, index).sum(dim=-1) + reference_entropy(full_student) +
                    reference_kl(full_student, full_teacher, False) + reference_kl(full_student, full_teacher, True))

        torch.testing.assert_close(actual, expected)
        (actual * weight).sum().backward()
        (expected * weight).sum().backward()
        torch.testing.assert_close(local_student.grad, full_student.grad[..., start:end])
        torch.testing.assert_close(local_teacher.grad, full_teacher.grad[..., start:end])
