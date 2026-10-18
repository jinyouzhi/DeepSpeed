# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Transfer analysis is checked against hand-derived counts and against the executor's own stats."""

import math
import os
from dataclasses import asdict

import pytest
import torch

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.checkpoint.affine import contiguous_split_map, replicated_map
from deepspeed.checkpoint.affine_transfer import plan_transfer
from deepspeed.checkpoint.affine_transfer_analysis import (INTER_NODE, INTRA_NODE, LOCAL, PEER, PathCost,
                                                           ReplicaRecorder, analyze_transfers, estimate_seconds,
                                                           expected_stats, fit_path_cost, node_path_of, replica_report,
                                                           report_to_dict)
from deepspeed.checkpoint.affine_transfer_executor import (StagingBudget, TransferEndpoints, TransferRequest,
                                                           execute_transfers)
from unit.common import DistributedTest

FP32 = 4


def _uneven_handoff():
    """8 elements, sources [4, 4] on processes 0-1, targets [2, 6] on processes 2-3.

    By hand: target 0 takes source 0[0:2]; target 1 takes source 0[2:4] and source 1[0:4].
    With 2 elements per round, 1->3 needs two rounds and every other pair one.
    """
    source = contiguous_split_map((8, ), [4, 4], 0)
    target = contiguous_split_map((8, ), [2, 6], 0)
    endpoints = TransferEndpoints(source={0: 0, 1: 1}, target={0: 2, 1: 3})
    return plan_transfer(target, source), endpoints


def test_volume_rounds_and_messages_match_hand_count():
    plan, endpoints = _uneven_handoff()
    report = analyze_transfers([plan], endpoints, FP32, StagingBudget(2 * FP32))

    assert (report.segments, report.chunks, report.rounds) == (3, 4, 2)
    assert (report.moved_bytes, report.remote_bytes, report.local_copy_bytes) == (32, 32, 0)
    by_process = {entry.process: entry for entry in report.processes}
    assert (by_process[0].sent_bytes, by_process[0].send_messages, by_process[0].rounds) == (16, 2, 1)
    assert (by_process[1].sent_bytes, by_process[1].send_messages, by_process[1].rounds) == (16, 2, 2)
    assert (by_process[2].received_bytes, by_process[2].rounds) == (8, 1)
    assert (by_process[3].received_bytes, by_process[3].receive_messages, by_process[3].receive_peers) == (24, 3, 2)
    # One-dimensional slices are packed on both sides, so nothing is staged.
    assert sum(entry.pack_bytes + entry.unpack_bytes + entry.staging_bytes for entry in report.processes) == 0


def test_strided_source_is_packed_before_sending():
    """Rows to columns on two processes: each sends a 2x2 block that is strided in its source."""
    source = contiguous_split_map((4, 4), [2, 2], 0)
    target = contiguous_split_map((4, 4), [2, 2], 1)
    endpoints = TransferEndpoints(source={0: 0, 1: 1}, target={0: 0, 1: 1})
    report = analyze_transfers([plan_transfer(target, source)], endpoints, FP32)

    assert (report.local_copy_bytes, report.remote_bytes) == (32, 32)
    for entry in report.processes:
        assert (entry.pack_bytes, entry.unpack_bytes, entry.staging_bytes) == (16, 0, 16)
    assert report.alias_candidate_bytes == 0


def test_same_layout_on_same_processes_could_alias():
    map_ = contiguous_split_map((4, 4), [2, 2], 0)
    endpoints = TransferEndpoints(source={0: 0, 1: 1}, target={0: 0, 1: 1})
    report = analyze_transfers([plan_transfer(map_, map_)], endpoints, FP32)
    assert report.alias_candidate_bytes == 64
    assert report.remote_bytes == 0


def test_shared_plan_counts_once_per_request():
    plan, endpoints = _uneven_handoff()
    single = analyze_transfers([plan], endpoints, FP32, StagingBudget(2 * FP32))
    double = analyze_transfers([plan, plan], endpoints, FP32, StagingBudget(2 * FP32))
    assert (double.segments, double.chunks, double.remote_bytes) == (6, 8, 64)
    # A larger batch packs into the same per-peer arenas, so it cannot need fewer rounds.
    assert double.rounds >= single.rounds


def test_node_path_classes():
    path_of = node_path_of(4)
    assert [path_of(1, 1), path_of(1, 3), path_of(3, 4)] == [LOCAL, INTRA_NODE, INTER_NODE]


def test_round_serial_and_path_parallel_estimates():
    """With 1 s latency and 8 B/s, process 3 receives 16 B then 8 B (3 s + 2 s) but streams 24 B in 4 s."""
    plan, endpoints = _uneven_handoff()
    report = analyze_transfers([plan], endpoints, FP32, StagingBudget(2 * FP32))
    estimate = estimate_seconds(report, {PEER: PathCost(latency_seconds=1.0, bytes_per_second=8.0)})
    assert estimate['round_serial_seconds'] == pytest.approx(5.0)
    assert estimate['path_parallel_bound_seconds'] == pytest.approx(4.0)
    with pytest.raises(KeyError):
        estimate_seconds(report, {})


def test_fit_recovers_latency_and_bandwidth():
    sizes = [1 << 10, 1 << 20, 1 << 24, 1 << 28]
    seconds = [5e-6 + size / 2e10 for size in sizes]
    cost = fit_path_cost(sizes, seconds)
    assert cost.latency_seconds == pytest.approx(5e-6, rel=1e-6)
    assert cost.bytes_per_second == pytest.approx(2e10, rel=1e-6)
    with pytest.raises(ValueError):
        fit_path_cost([8, 8], [1.0, 2.0])


def _replicated_source(target_processes):
    source = replicated_map((4, ), 2)
    target = contiguous_split_map((4, ), [2, 2], 0)
    endpoints = TransferEndpoints(source={0: 0, 1: 1}, target=dict(enumerate(target_processes)))
    return source, target, endpoints


def test_recording_choices_does_not_change_the_plan():
    source, target, endpoints = _replicated_source((2, 3))
    assert plan_transfer(target, source, replica_selector=ReplicaRecorder()) == plan_transfer(target, source)


def test_replica_report_on_remote_targets():
    """Both holders have everything; the default reads 16 B from process 0, an even split reads 8 B each."""
    source, target, endpoints = _replicated_source((2, 3))
    recorder = ReplicaRecorder()
    plan = plan_transfer(target, source, replica_selector=recorder)
    replicas = replica_report(recorder, [plan], endpoints, FP32)
    assert replicas.replicated_bytes == 16
    assert replicas.read_bytes_by_process == {0: 16}
    assert replicas.balanced_read_bound_bytes == pytest.approx(8.0)
    assert (replicas.colocated_available_bytes, replicas.colocated_chosen_bytes) == (0, 0)


def test_replica_report_counts_colocated_reads():
    source, target, endpoints = _replicated_source((0, 1))
    default = ReplicaRecorder()
    replica_report_default = replica_report(default, [plan_transfer(target, source, replica_selector=default)],
                                            endpoints, FP32)
    local_first = ReplicaRecorder(lambda request: request.target_rank)
    replica_report_local = replica_report(local_first, [plan_transfer(target, source, replica_selector=local_first)],
                                          endpoints, FP32)
    assert replica_report_default.colocated_available_bytes == 16
    assert replica_report_default.colocated_chosen_bytes == 8
    assert replica_report_local.colocated_chosen_bytes == 16


def test_report_is_json_ready():
    source, target, endpoints = _replicated_source((2, 3))
    recorder = ReplicaRecorder()
    plan = plan_transfer(target, source, replica_selector=recorder)
    report = analyze_transfers([plan], endpoints, FP32)
    summary = report_to_dict(report,
                             replicas=replica_report(recorder, [plan], endpoints, FP32),
                             costs={PEER: PathCost(1e-5, 1e10)},
                             include_messages=True)
    assert summary['volume']['remote_bytes'] == 16
    assert summary['potential']['balanced_read_bound_bytes'] == pytest.approx(8.0)
    assert len(summary['messages']) == len(report.messages)


def _join_transfer_group():
    if not dist.is_initialized():
        dist.init_distributed(dist_backend=get_accelerator().communication_backend_name(), init_method='env://')
    local_rank = int(os.environ.get('LOCAL_RANK', dist.get_rank()))
    get_accelerator().set_device(local_rank)
    return torch.device(get_accelerator().current_device_name())


@pytest.mark.skipif(get_accelerator().device_count() < 4, reason='requires four visible accelerator devices')
class TestPredictionMatchesExecutor(DistributedTest):
    world_size = 4

    @pytest.mark.parametrize('peer_elements', [3, 64])
    def test_expected_stats_equal_executor_stats(self, peer_elements):
        """A batch of a strided and a replicated parameter, so staging, chunking and local copies all occur."""
        device = _join_transfer_group()
        rank = dist.get_rank()
        dtype = torch.float32
        endpoints = TransferEndpoints(source={0: 0, 1: 1, 2: 2, 3: 3}, target={0: 3, 1: 2, 2: 1, 3: 0})
        cases = [
            (contiguous_split_map((8, 8), [2, 2, 2, 2], 0), contiguous_split_map((8, 8), [2, 2, 2, 2], 1)),
            (replicated_map((8, 4), 4), contiguous_split_map((8, 4), [1, 3, 2, 2], 0)),
        ]
        requests, plans = [], []
        for index, (source, target) in enumerate(cases):
            numel = math.prod(source.logical_shape)
            full = torch.arange(numel, dtype=dtype, device=device).reshape(source.logical_shape)
            sources = {
                location: source.extract(full, location).contiguous()
                for location, process in endpoints.source.items() if process == rank
            }
            targets = {
                location: torch.full(shape, float('nan'), dtype=dtype, device=device)
                for location, shape in target.shard_shapes.items() if endpoints.target[location] == rank
            }
            plan = plan_transfer(target, source)
            plans.append(plan)
            requests.append(TransferRequest(f'param-{index}', plan, sources, targets))

        budget = StagingBudget(peer_elements * 4)
        stats = execute_transfers(requests, endpoints, dtype, budget)
        report = analyze_transfers(plans, endpoints, 4, budget)
        assert asdict(stats) == expected_stats(report, rank)
