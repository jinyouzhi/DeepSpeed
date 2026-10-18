# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Describe what a transfer batch will do before it runs, without deciding anything.

The planner chooses segments and replicas, and the executor chooses chunks, rounds and staging.
This module only reads those outputs. It reports volume, granularity, communication structure,
the executor's schedule and staging, path classes, and how much room a few capabilities the
planner does not have yet (aliasing, multi-source reads, overlap across paths) would have.

Nothing here feeds back into a plan, and nothing here picks a replica or a schedule: those
decisions, and the cost model behind them, belong to the planner's policy layer. Time estimates
appear only when the caller supplies measured ``PathCost`` values; they exist to check how well a
simple latency/bandwidth model explains measured transfers, which is the evidence a cost model needs.
"""

import json
import math
from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .affine_transfer import ReplicaSelectionRequest, TransferPlan, TransferSegment
from .affine_transfer_executor import StagingBudget, TransferEndpoints, _packed_rounds, build_operations

__all__ = [
    'LOCAL', 'PEER', 'INTRA_NODE', 'INTER_NODE', 'PathCost', 'TransferMessage', 'ProcessReport', 'PlanReport',
    'ReplicaRecorder', 'ReplicaReport', 'analyze_transfers', 'default_path_of', 'node_path_of', 'expected_stats',
    'estimate_seconds', 'fit_path_cost', 'replica_report', 'report_to_dict', 'distribution'
]

REPORT_SCHEMA = 1

LOCAL = 'local'
PEER = 'peer'
INTRA_NODE = 'intra_node'
INTER_NODE = 'inter_node'

# Subset enumeration is exact but exponential; past this many source processes the bound is skipped.
MAX_PROCESSES_FOR_EXACT_BOUND = 16


def default_path_of(from_process: int, to_process: int) -> str:
    return LOCAL if from_process == to_process else PEER


def node_path_of(processes_per_node: int) -> Callable[[int, int], str]:
    """Classify by node, for processes numbered node by node."""
    if processes_per_node <= 0:
        raise ValueError('processes_per_node must be positive')

    def path_of(from_process: int, to_process: int) -> str:
        if from_process == to_process:
            return LOCAL
        if from_process // processes_per_node == to_process // processes_per_node:
            return INTRA_NODE
        return INTER_NODE

    return path_of


@dataclass(frozen=True)
class PathCost:
    """Startup latency and bandwidth of one path class, measured rather than assumed."""
    latency_seconds: float
    bytes_per_second: float

    def seconds(self, nbytes: int) -> float:
        return self.latency_seconds + nbytes / self.bytes_per_second


@dataclass(frozen=True)
class TransferMessage:
    """One point-to-point payload the executor posts in one round."""
    round: int
    from_process: int
    to_process: int
    nbytes: int
    entries: int
    packed_on_send: bool
    unpacked_on_receive: bool
    path: str


@dataclass(frozen=True)
class ProcessReport:
    process: int
    sent_bytes: int = 0
    received_bytes: int = 0
    local_copy_bytes: int = 0
    send_messages: int = 0
    receive_messages: int = 0
    rounds: int = 0
    send_peers: int = 0
    receive_peers: int = 0
    pack_bytes: int = 0
    unpack_bytes: int = 0
    staging_bytes: int = 0


@dataclass(frozen=True)
class PlanReport:
    element_size: int
    bytes_per_peer: int
    parameters: int
    segments: int
    chunks: int
    logical_bytes: int
    moved_bytes: int
    local_copy_bytes: int
    remote_bytes: int
    rounds: int
    alias_candidate_bytes: int
    processes: Tuple[ProcessReport, ...]
    messages: Tuple[TransferMessage, ...]
    segment_bytes: Tuple[int, ...]
    chunk_bytes: Tuple[int, ...]
    planner: Dict[str, int] = field(default_factory=dict)

    def process(self, process: int) -> ProcessReport:
        for entry in self.processes:
            if entry.process == process:
                return entry
        return ProcessReport(process)


def _is_packed(shape: Sequence[int], strides: Sequence[int]) -> bool:
    """Whether a strided region is one run of elements in row-major order, as torch judges it."""
    expected = 1
    for extent, stride in zip(reversed(tuple(shape)), reversed(tuple(strides))):
        if extent == 1:
            continue
        if stride != expected:
            return False
        expected *= extent
    return True


def _chunk_is_packed(operation, source_side: bool) -> bool:
    strides = operation.segment.source_strides if source_side else operation.segment.target_strides
    return _is_packed(operation.chunk_shape, strides)


def _alias_candidate_bytes(plan: TransferPlan, endpoints: TransferEndpoints, element_size: int) -> int:
    """Bytes of target shards that a single packed, co-located source region fills completely.

    Such a target could be a view of the source instead of a copy, if lifetimes allowed it. The
    executor always copies; this only measures how much a zero-copy path could save.
    """
    by_target: Dict[int, List[TransferSegment]] = {}
    for segment in plan.segments:
        by_target.setdefault(segment.target_rank, []).append(segment)
    total = 0
    for target_rank, segments in by_target.items():
        if len(segments) != 1:
            continue
        segment = segments[0]
        same_process = endpoints.source[segment.source_rank] == endpoints.target[target_rank]
        packed = _is_packed(segment.shape, segment.source_strides) and _is_packed(segment.shape,
                                                                                  segment.target_strides)
        if same_process and packed:
            total += segment.numel * element_size
    return total


def analyze_transfers(plans: Sequence[TransferPlan],
                      endpoints: TransferEndpoints,
                      element_size: int,
                      budget: StagingBudget = StagingBudget(),
                      path_of: Optional[Callable[[int, int], str]] = None) -> PlanReport:
    """Report one ``execute_transfers`` batch, given its plans in request order.

    The schedule comes from the executor's own ``build_operations`` and round packing, so the
    counts match what each process will return in ``TransferStats``.
    """
    if path_of is None:
        path_of = default_path_of

    indexed = []
    schedules = {}
    for item, plan in enumerate(plans):
        identity = id(plan)
        if identity not in schedules:
            schedules[identity] = build_operations(plan, endpoints, budget, element_size)
        indexed.extend((item, operation) for operation in schedules[identity])

    local_copy = {}
    by_pair: Dict[Tuple[int, int], list] = {}
    for item, operation in indexed:
        if operation.from_process == operation.to_process:
            process = operation.from_process
            local_copy[process] = local_copy.get(process, 0) + operation.payload_bytes
        else:
            pair = (operation.from_process, operation.to_process)
            by_pair.setdefault(pair, []).append((item, operation))

    max_elements = budget.bytes_per_peer // element_size
    messages = []
    sent, received, send_count, receive_count = {}, {}, {}, {}
    pack, unpack, send_peers, receive_peers = {}, {}, {}, {}
    rounds_of, staging = {}, {}
    for pair in sorted(by_pair):
        from_process, to_process = pair
        rounds = _packed_rounds(by_pair[pair], max_elements)
        largest_packed_send = 0
        largest_unpacked_receive = 0
        for round_index, (entries, count) in enumerate(rounds):
            nbytes = count * element_size
            # The executor sends a view only when the round holds one packed region; otherwise it stages.
            single = len(entries) == 1
            send_direct = single and _chunk_is_packed(entries[0][1], source_side=True)
            receive_direct = single and _chunk_is_packed(entries[0][1], source_side=False)
            messages.append(
                TransferMessage(round=round_index,
                                from_process=from_process,
                                to_process=to_process,
                                nbytes=nbytes,
                                entries=len(entries),
                                packed_on_send=not send_direct,
                                unpacked_on_receive=not receive_direct,
                                path=path_of(from_process, to_process)))
            if not send_direct:
                pack[from_process] = pack.get(from_process, 0) + nbytes
                largest_packed_send = max(largest_packed_send, nbytes)
            if not receive_direct:
                unpack[to_process] = unpack.get(to_process, 0) + nbytes
                largest_unpacked_receive = max(largest_unpacked_receive, nbytes)
            sent[from_process] = sent.get(from_process, 0) + nbytes
            received[to_process] = received.get(to_process, 0) + nbytes
        send_count[from_process] = send_count.get(from_process, 0) + len(rounds)
        receive_count[to_process] = receive_count.get(to_process, 0) + len(rounds)
        send_peers[from_process] = send_peers.get(from_process, 0) + 1
        receive_peers[to_process] = receive_peers.get(to_process, 0) + 1
        # A process runs as many rounds as its busiest peer queue, in either direction.
        rounds_of[from_process] = max(rounds_of.get(from_process, 0), len(rounds))
        rounds_of[to_process] = max(rounds_of.get(to_process, 0), len(rounds))
        # One arena per peer and direction, sized by its largest staged round.
        staging[from_process] = staging.get(from_process, 0) + largest_packed_send
        staging[to_process] = staging.get(to_process, 0) + largest_unpacked_receive

    processes = set(endpoints.source.values()) | set(endpoints.target.values())
    process_reports = tuple(
        ProcessReport(process=process,
                      sent_bytes=sent.get(process, 0),
                      received_bytes=received.get(process, 0),
                      local_copy_bytes=local_copy.get(process, 0),
                      send_messages=send_count.get(process, 0),
                      receive_messages=receive_count.get(process, 0),
                      rounds=rounds_of.get(process, 0),
                      send_peers=send_peers.get(process, 0),
                      receive_peers=receive_peers.get(process, 0),
                      pack_bytes=pack.get(process, 0),
                      unpack_bytes=unpack.get(process, 0),
                      staging_bytes=staging.get(process, 0)) for process in sorted(processes))

    planner = {}
    for plan in plans:
        for name, value in asdict(plan.statistics).items():
            planner[name] = planner.get(name, 0) + value

    segment_bytes = tuple(segment.numel * element_size for plan in plans for segment in plan.segments)
    remote_bytes = sum(sent.values())
    local_bytes = sum(local_copy.values())
    return PlanReport(element_size=element_size,
                      bytes_per_peer=budget.bytes_per_peer,
                      parameters=len(plans),
                      segments=len(segment_bytes),
                      chunks=len(indexed),
                      logical_bytes=sum(math.prod(plan.logical_shape) * element_size for plan in plans),
                      moved_bytes=sum(segment_bytes),
                      local_copy_bytes=local_bytes,
                      remote_bytes=remote_bytes,
                      rounds=max(rounds_of.values(), default=0),
                      alias_candidate_bytes=sum(
                          _alias_candidate_bytes(plan, endpoints, element_size) for plan in plans),
                      processes=process_reports,
                      messages=tuple(messages),
                      segment_bytes=segment_bytes,
                      chunk_bytes=tuple(operation.payload_bytes for _, operation in indexed),
                      planner=planner)


def expected_stats(report: PlanReport, process: int) -> Dict[str, int]:
    """The ``TransferStats`` fields ``process`` should return for the analyzed batch."""
    entry = report.process(process)
    return {
        'segments': report.segments,
        'chunks': report.chunks,
        'rounds': entry.rounds,
        'local_copy_bytes': entry.local_copy_bytes,
        'sent_bytes': entry.sent_bytes,
        'received_bytes': entry.received_bytes,
    }


def distribution(values: Sequence[float]) -> Dict[str, float]:
    """Count, sum and order statistics, with nearest-rank percentiles."""
    ordered = sorted(values)
    if not ordered:
        return {'count': 0, 'sum': 0}

    def percentile(fraction):
        index = max(0, math.ceil(fraction * len(ordered)) - 1)
        return ordered[index]

    return {
        'count': len(ordered),
        'sum': sum(ordered),
        'min': ordered[0],
        'p50': percentile(0.5),
        'p90': percentile(0.9),
        'max': ordered[-1],
        'mean': sum(ordered) / len(ordered),
    }


def _imbalance(values: Sequence[int]) -> Optional[float]:
    """Busiest over mean; 1.0 is perfectly even. None when nothing moves."""
    total = sum(values)
    if total == 0:
        return None
    return max(values) / (total / len(values))


def estimate_seconds(report: PlanReport, costs: Mapping[str, PathCost]) -> Dict[str, object]:
    """Two model estimates from measured path costs, for comparison against measured time.

    This is a model check, not a cost model: no planner or selector calls it.

    Each process is assumed to have one link per direction and path class, shared by every message
    on it. ``round_serial`` follows the executor: rounds run one after another, and a round lasts
    as long as its busiest link. ``path_parallel`` drops the round boundaries, so each link streams
    all its bytes after one startup latency; it is a lower bound on ``round_serial``, and the gap is
    the room left for scheduling across rounds and paths.
    """
    missing = sorted({message.path for message in report.messages} - set(costs))
    if missing:
        raise KeyError(f'no PathCost for path classes {missing}')

    round_bytes: Dict[Tuple[int, int, str, str], int] = {}
    total_bytes: Dict[Tuple[int, str, str], int] = {}
    for message in report.messages:
        for process, direction in ((message.from_process, 'send'), (message.to_process, 'receive')):
            round_key = (process, message.round, direction, message.path)
            round_bytes[round_key] = round_bytes.get(round_key, 0) + message.nbytes
            total_key = (process, direction, message.path)
            total_bytes[total_key] = total_bytes.get(total_key, 0) + message.nbytes

    busiest_link_in_round: Dict[Tuple[int, int], float] = {}
    for (process, round_index, _, path), nbytes in round_bytes.items():
        key = (process, round_index)
        seconds = costs[path].seconds(nbytes)
        busiest_link_in_round[key] = max(busiest_link_in_round.get(key, 0.0), seconds)
    round_serial: Dict[int, float] = {}
    for (process, _), seconds in busiest_link_in_round.items():
        round_serial[process] = round_serial.get(process, 0.0) + seconds

    path_parallel: Dict[int, float] = {}
    for (process, _, path), nbytes in total_bytes.items():
        seconds = costs[path].seconds(nbytes)
        path_parallel[process] = max(path_parallel.get(process, 0.0), seconds)

    # Local copies run after the last round in the executor, so they add to the serial estimate.
    local_cost = costs.get(LOCAL)
    for entry in report.processes:
        if local_cost is not None and entry.local_copy_bytes:
            local_seconds = local_cost.seconds(entry.local_copy_bytes)
            round_serial[entry.process] = round_serial.get(entry.process, 0.0) + local_seconds
            path_parallel[entry.process] = max(path_parallel.get(entry.process, 0.0), local_seconds)

    return {
        'model': 'per-link latency + bytes / bandwidth from measured PathCost; not a measurement',
        'round_serial_seconds': max(round_serial.values(), default=0.0),
        'path_parallel_bound_seconds': max(path_parallel.values(), default=0.0),
        'round_serial_by_process': dict(sorted(round_serial.items())),
    }


def fit_path_cost(sizes: Sequence[int], seconds: Sequence[float]) -> PathCost:
    """Least-squares ``seconds = latency + size / bandwidth`` over measured message sizes."""
    if len(sizes) != len(seconds) or len(set(sizes)) < 2:
        raise ValueError('need matching samples at two or more distinct sizes')
    count = len(sizes)
    mean_size = sum(sizes) / count
    mean_seconds = sum(seconds) / count
    covariance = sum((size - mean_size) * (time - mean_seconds) for size, time in zip(sizes, seconds))
    variance = sum((size - mean_size)**2 for size in sizes)
    slope = covariance / variance
    if slope <= 0:
        raise ValueError('time does not grow with size; these samples cannot give a bandwidth')
    latency = max(0.0, mean_seconds - slope * mean_size)
    return PathCost(latency_seconds=latency, bytes_per_second=1.0 / slope)


class ReplicaRecorder:
    """A replica selector that records every choice the planner offers and keeps the given policy.

    Wrapping ``selector`` (or the planner default, the first holder) changes no plan; it only
    keeps the candidate sets, which a finished plan no longer shows.
    """

    def __init__(self, selector: Optional[Callable[[ReplicaSelectionRequest], int]] = None):
        self.selector = selector
        # (candidate ranks, chosen rank, target rank, elements) per replicated region
        self.choices: List[Tuple[Tuple[int, ...], int, int, int]] = []

    def __call__(self, request: ReplicaSelectionRequest) -> int:
        if self.selector is None:
            chosen = request.candidates[0].source_rank
        else:
            chosen = self.selector(request)
        candidates = tuple(candidate.source_rank for candidate in request.candidates)
        self.choices.append((candidates, chosen, request.target_rank, math.prod(request.shape)))
        return chosen


@dataclass(frozen=True)
class ReplicaReport:
    replicated_bytes: int
    colocated_available_bytes: int
    colocated_chosen_bytes: int
    read_bytes_by_process: Dict[int, int]
    balanced_read_bound_bytes: Optional[float]


def _balanced_bound(fixed: Mapping[int, int], flexible: Mapping[Tuple[int, ...], int]) -> Optional[float]:
    """Smallest possible busiest-reader load if replicated bytes could be split across their holders.

    A fractional assignment exists with maximum load T exactly when every set of processes S can
    absorb its fixed load plus all replicated bytes that only S holds: T >= that total / |S|.
    """
    processes = sorted(set(fixed) | {process for holders in flexible for process in holders})
    if len(processes) > MAX_PROCESSES_FOR_EXACT_BOUND:
        return None
    best = 0.0
    for size in range(1, len(processes) + 1):
        for subset in combinations(processes, size):
            members = set(subset)
            load = sum(fixed.get(process, 0) for process in subset)
            load += sum(nbytes for holders, nbytes in flexible.items() if set(holders) <= members)
            best = max(best, load / size)
    return best


def replica_report(recorder: ReplicaRecorder, plans: Sequence[TransferPlan], endpoints: TransferEndpoints,
                   element_size: int) -> ReplicaReport:
    """How much replicated data there was, where it was read from, and how even reads could be.

    ``plans`` must be exactly the plans built with ``recorder`` as their selector.
    """
    read: Dict[int, int] = {}
    for plan in plans:
        for segment in plan.segments:
            process = endpoints.source[segment.source_rank]
            read[process] = read.get(process, 0) + segment.numel * element_size

    flexible: Dict[Tuple[int, ...], int] = {}
    fixed = dict(read)
    replicated = available = chosen_local = 0
    for candidates, chosen, target_rank, elements in recorder.choices:
        nbytes = elements * element_size
        holders = tuple(sorted({endpoints.source[rank] for rank in candidates}))
        reader = endpoints.source[chosen]
        target_process = endpoints.target[target_rank]
        replicated += nbytes
        fixed[reader] -= nbytes
        flexible[holders] = flexible.get(holders, 0) + nbytes
        if target_process in holders:
            available += nbytes
        if reader == target_process:
            chosen_local += nbytes

    return ReplicaReport(replicated_bytes=replicated,
                         colocated_available_bytes=available,
                         colocated_chosen_bytes=chosen_local,
                         read_bytes_by_process=dict(sorted(read.items())),
                         balanced_read_bound_bytes=_balanced_bound(fixed, flexible))


def report_to_dict(report: PlanReport,
                   replicas: Optional[ReplicaReport] = None,
                   costs: Optional[Mapping[str, PathCost]] = None,
                   include_messages: bool = False) -> Dict[str, object]:
    """A JSON-ready summary; ``include_messages`` adds the full per-round message list."""
    pairs: Dict[str, Dict[str, int]] = {}
    paths: Dict[str, Dict[str, int]] = {}
    for message in report.messages:
        pair = pairs.setdefault(f'{message.from_process}->{message.to_process}', {'messages': 0, 'bytes': 0})
        pair['messages'] += 1
        pair['bytes'] += message.nbytes
        path = paths.setdefault(message.path, {'messages': 0, 'bytes': 0})
        path['messages'] += 1
        path['bytes'] += message.nbytes
    local_bytes = report.local_copy_bytes
    if local_bytes:
        local = paths.setdefault(LOCAL, {'messages': 0, 'bytes': 0})
        local['bytes'] += local_bytes

    sent = [entry.sent_bytes for entry in report.processes]
    received = [entry.received_bytes for entry in report.processes]
    staged_messages = sum(1 for message in report.messages if message.packed_on_send or message.unpacked_on_receive)
    summary = {
        'schema': REPORT_SCHEMA,
        'element_size': report.element_size,
        'bytes_per_peer': report.bytes_per_peer,
        'volume': {
            'parameters': report.parameters,
            'logical_bytes': report.logical_bytes,
            'moved_bytes': report.moved_bytes,
            'local_copy_bytes': report.local_copy_bytes,
            'remote_bytes': report.remote_bytes,
        },
        'granularity': {
            'segments': distribution(report.segment_bytes),
            'chunks': distribution(report.chunk_bytes),
            'messages': distribution([message.nbytes for message in report.messages]),
            'staged_messages': staged_messages,
            'pack_bytes': sum(entry.pack_bytes for entry in report.processes),
            'unpack_bytes': sum(entry.unpack_bytes for entry in report.processes),
        },
        'schedule': {
            'rounds': report.rounds,
            'staging_bytes_max': max((entry.staging_bytes for entry in report.processes), default=0),
        },
        'balance': {
            'sent_imbalance': _imbalance(sent),
            'received_imbalance': _imbalance(received),
            'active_pairs': len(pairs),
        },
        'processes': [asdict(entry) for entry in report.processes],
        'pairs': pairs,
        'paths': paths,
        'potential': {
            'alias_candidate_bytes': report.alias_candidate_bytes,
        },
        'planner': report.planner,
    }
    if replicas is not None:
        summary['replicas'] = asdict(replicas)
        busiest = max(replicas.read_bytes_by_process.values(), default=0)
        summary['potential']['busiest_reader_bytes'] = busiest
        summary['potential']['balanced_read_bound_bytes'] = replicas.balanced_read_bound_bytes
    if costs is not None:
        summary['estimate'] = estimate_seconds(report, costs)
        summary['path_costs'] = {name: asdict(cost) for name, cost in costs.items()}
    if include_messages:
        summary['messages'] = [asdict(message) for message in report.messages]
    # Round-trip once so callers get exactly what json.dump would write (string keys, plain types).
    return json.loads(json.dumps(summary))
