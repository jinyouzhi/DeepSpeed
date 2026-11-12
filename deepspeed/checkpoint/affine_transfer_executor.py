# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Move a parameter between two shardings by executing a plan from ``affine_transfer``.

The plan names, for each copy, a region of one source shard and the region of one target shard that
should receive it. This module turns those names into data movement: pack out of the source buffer,
across a transport group, through a staging buffer, and into the target buffer. The full logical
parameter is never formed anywhere, which was the point of planning instead of converting.

Three things are deliberately not decided here.

*Endpoints.* Which process actually holds a plan location is supplied by the caller. The plan's two
rank namespaces are not one world size: a source rank 0 has no claim to be the process that is target
rank 0, and folding the two together is how a transfer reads one rank's shard as another's.

*Budget.* How many bytes may be in flight is the caller's, because what fits depends on what else is
resident on the device. A budget that is exhausted is an error, never a silently larger transfer.

*Dtype.* Passed in rather than read from the buffers, because a process that holds no buffer at all
still has to reach the same conclusion as the ones that do -- see ``plan_digest``.

The protocol is: agree on a digest of the schedule before any communication, then work in rounds. In
each round every rank arms all of its receives first and only then posts its sends, so two ranks that
each have something for the other cannot both be waiting on the other's send. That ordering is the
entire deadlock argument, and it only holds because the digest made every rank compute the same
rounds. Returns only once every payload it posted has landed, so a caller can read the target buffers
as soon as it is given control.

Staging is an artifact of the current transport, not of affine transfer. Torch point-to-point
operations move one contiguous tensor per peer per message, so a round whose regions are strided or come
from several segments is packed through a per-peer arena, and ``StagingBudget`` bounds that arena; a
round that is one region already contiguous in its buffer is sent or received in place.
A zero-copy transport such as GPUDirect RDMA would instead take each round's ``(offset, length)`` pairs
as a scatter/gather list over registered memory: that removes the arena copy, but needs a registration
lifetime this API has no place for. Expect the wire contract -- one packed message per peer per round --
to change when such a path lands.

``TransferWindow`` is that registration for peers that can address each other's memory (one node over
PCIe, NVLink or Xe Link). Buffers carved from a window are published to the group, and an operation whose
source or target lies in a window is copied in place by one process -- the target pulls, or the source
pushes -- with no staging, no peer message and no budget. Everything else keeps the protocol above.

Registration is only worth it when it is reused, and so is the rest of the setup: validation, the digest
agreement, the exchange of window offsets, arenas and peer views. ``prepare_transfers`` does all of it once
and returns a ``PreparedTransfer`` whose ``run`` moves only data, as often as the caller's weights change.
``execute_transfers`` is one prepare followed by one run.

One device per process, with explicit transport-group rank bindings, synchronous on return. No CUDA graphs, no overlap with compute,
no custom kernels: those all change the schedule, which is the last thing a first version of a data
mover should be perturbing.
"""

import math
import sys
import zlib
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from deepspeed.accelerator import get_accelerator

from .affine_transfer import TransferPlan, TransferSegment

__all__ = [
    'AffineTransferExecutionError', 'TransferEndpoints', 'StagingBudget', 'TransferStats', 'TransferRequest',
    'plan_digest', 'execute_transfer', 'execute_transfers', 'TransferWindow', 'prepare_transfers', 'PreparedTransfer'
]


class AffineTransferExecutionError(ValueError):
    """Wrong enough that no data should have moved."""


@dataclass(frozen=True)
class TransferEndpoints:
    """Which process holds each location named by the plan. ``source`` covers source ranks, ``target``
    covers target ranks, and the two are separate mappings for the reason given above."""
    source: Mapping[int, int]
    target: Mapping[int, int]


@dataclass(frozen=True)
class StagingBudget:
    """Bytes a rank may stage per peer, in each direction, at one time.

    A property of the current torch point-to-point transport rather than of the plan; see the module
    docstring."""
    bytes_per_peer: int = 8 << 20


@dataclass(frozen=True)
class TransferStats:
    segments: int
    chunks: int
    rounds: int
    local_copy_bytes: int
    sent_bytes: int
    received_bytes: int
    mapped_bytes: int = 0


class TransferWindow:
    """Device memory that every process of the transfer group can address.

    Construction is collective over ``group``. Carve long-lived shards with ``tensor`` and pass the window
    to every call: registration is paid once, not per transfer. ``direction`` decides who copies when both
    the source and the target of an operation lie in a window; which one is faster depends on the
    interconnect, so it is the caller's.
    """
    ALIGNMENT = 512
    READY, DONE = 0, 1

    def __init__(self, nbytes: int, group: Optional[object] = None, direction: str = 'pull'):
        import deepspeed.comm as dist
        if direction not in ('pull', 'push'):
            raise AffineTransferExecutionError("window direction must be 'pull' or 'push'")
        self.group = group
        self.direction = direction
        self.nbytes = -(-nbytes // self.ALIGNMENT) * self.ALIGNMENT
        self.pool, self._handle = dist.symm_mem_rendezvous(self.nbytes, torch.uint8, group)
        self._used = 0
        self._peers: Dict[Tuple[int, torch.dtype], torch.Tensor] = {}
        self._stream = None
        # Stream-ordered peer signals are verified on CUDA and XPU; elsewhere a run fences with group barriers.
        self.signals = self.pool.device.type in ('cuda', 'xpu')

    def tensor(self, shape: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
        """A new shard inside the window; never freed before the window."""
        nbytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        start = self._used
        if start + nbytes > self.nbytes:
            raise AffineTransferExecutionError(f'window holds {self.nbytes - start} more bytes, {nbytes} requested')
        self._used = -(-(start + nbytes) // self.ALIGNMENT) * self.ALIGNMENT
        return self.pool[start:start + nbytes].view(dtype).view(tuple(shape))

    def offset_of(self, buffer: torch.Tensor) -> Optional[int]:
        """Element offset of ``buffer`` in this process's window, or None if it lies elsewhere."""
        if buffer.device != self.pool.device:
            return None
        start = buffer.data_ptr() - self.pool.data_ptr()
        size = buffer.element_size()
        if start < 0 or start % size or start + buffer.numel() * size > self.nbytes:
            return None
        return start // size

    def peer(self, process: int, dtype: torch.dtype) -> torch.Tensor:
        """``process``'s window as flat elements of ``dtype``, mapped into this process."""
        key = (process, dtype)
        if key not in self._peers:
            import deepspeed.comm as dist
            numel = self.nbytes // torch.empty((), dtype=dtype).element_size()
            self._peers[key] = dist.symm_mem_peer_tensor(self._handle, process, numel, dtype)
        return self._peers[key]

    @property
    def stream(self):
        """The stream window copies run on, so they overlap staged messages on the caller's stream.

        One stream for all peers, not one each: copiers that each read one peer at a time in staggered
        order keep every link busy, while concurrent per-peer streams make the reads contend for them.
        """
        if self._stream is None:
            self._stream = get_accelerator().Stream()
        return self._stream

    def signal(self, process: int, channel: int) -> None:
        """Raise ``channel`` at ``process``, ordered after this process's current stream."""
        self._handle.put_signal(process, channel)

    def wait(self, process: int, channel: int) -> None:
        """Hold this process's current stream until ``process`` raises ``channel``, then lower it."""
        self._handle.wait_signal(process, channel)


def _window_offsets(requests: Sequence['TransferRequest'], window: TransferWindow) -> Dict[Tuple, int]:
    """This process's shards that lie in the window, keyed so that every process can look them up."""
    offsets = {}
    for request in requests:
        for side, buffers in (('source', request.source_buffers), ('target', request.target_buffers)):
            for location, buffer in buffers.items():
                if isinstance(buffer, torch.Tensor) and buffer.is_contiguous():
                    offset = window.offset_of(buffer)
                    if offset is not None:
                        offsets[(request.key, side, location)] = offset
    return offsets


def _mapped_direction(request_key: str, operation: '_Operation', offsets: Mapping[Tuple, int],
                      window: TransferWindow) -> Optional[str]:
    """``'pull'``, ``'push'`` or None, identically on every process of the group."""
    segment = operation.segment
    pull = (request_key, 'source', segment.source_rank) in offsets
    push = (request_key, 'target', segment.target_rank) in offsets
    if pull and push:
        return window.direction
    if pull:
        return 'pull'
    return 'push' if push else None


@dataclass(frozen=True)
class TransferRequest:
    """One parameter; ``key`` must identify it identically on every process."""
    key: str
    plan: Optional[TransferPlan]
    source_buffers: Mapping[int, torch.Tensor]
    target_buffers: Mapping[int, torch.Tensor]
    preflight_error: Optional[str] = None


@dataclass(frozen=True)
class _Operation:
    """One chunk of one segment, resolved to a pair of processes."""
    segment: TransferSegment
    chunk: int
    chunk_shape: Tuple[int, ...]
    source_offset: int
    target_offset: int
    from_process: int
    to_process: int
    elements: int

    @property
    def key(self) -> Tuple:
        return (self.segment.target_rank, self.segment.target_offset, self.segment.source_rank,
                self.segment.source_offset, self.chunk)

    @property
    def payload_bytes(self) -> int:
        return self.elements * self._element_size

    _element_size: int = 0


def _chunks(segment: TransferSegment, max_elements: int):
    """Tile a strided rectangle by its packed payload, retaining both address spaces."""
    if segment.numel <= max_elements:
        yield segment
        return
    tile = [1] * len(segment.shape)
    remaining = max_elements
    for axis in range(len(tile) - 1, -1, -1):
        tile[axis] = min(segment.shape[axis], remaining)
        remaining //= tile[axis]
    origins = [range(0, extent, step) for extent, step in zip(segment.shape, tile)]
    for origin in product(*origins):
        shape = tuple(min(step, extent - low) for step, extent, low in zip(tile, segment.shape, origin))
        yield TransferSegment(
            source_rank=segment.source_rank,
            source_offset=segment.source_offset + sum(low * stride
                                                      for low, stride in zip(origin, segment.source_strides)),
            source_strides=segment.source_strides,
            target_rank=segment.target_rank,
            target_offset=segment.target_offset + sum(low * stride
                                                      for low, stride in zip(origin, segment.target_strides)),
            target_strides=segment.target_strides,
            shape=shape)


def build_operations(plan: TransferPlan, endpoints: TransferEndpoints, budget: StagingBudget,
                     element_size: int) -> List[_Operation]:
    """The complete schedule, as a list every rank derives identically."""
    if element_size <= 0 or budget.bytes_per_peer < element_size:
        raise AffineTransferExecutionError('the staging budget must hold at least one element')
    operations: List[_Operation] = []
    for segment in plan.segments:
        _validate_segment(segment, len(plan.logical_shape))
        for index, chunk in enumerate(_chunks(segment, budget.bytes_per_peer // element_size)):
            operation = _Operation(segment=segment,
                                   chunk=index,
                                   chunk_shape=chunk.shape,
                                   source_offset=chunk.source_offset,
                                   target_offset=chunk.target_offset,
                                   from_process=endpoints.source[segment.source_rank],
                                   to_process=endpoints.target[segment.target_rank],
                                   elements=chunk.numel,
                                   _element_size=element_size)
            operations.append(operation)
    operations.sort(key=lambda operation: operation.key)
    return operations


def plan_digest(plan: TransferPlan, endpoints: TransferEndpoints, budget: StagingBudget, element_size: int) -> int:
    """A checksum over the schedule: geometry, addressing, endpoints and budget.

    Deliberately not over buffer contents -- those differ by design. This is the thing every rank must
    be able to compute before it owns any data, which is why dtype is an argument and not an
    inspection.
    """
    payload = [(segment.source_rank, segment.target_rank, segment.source_offset, segment.target_offset, segment.shape,
                segment.source_strides, segment.target_strides) for segment in plan.segments]
    framed = {
        'schema': 3,
        'segments': payload,
        'logical_shape': plan.logical_shape,
        'source_endpoints': dict(sorted(endpoints.source.items())),
        'target_endpoints': dict(sorted(endpoints.target.items())),
        'bytes_per_peer': budget.bytes_per_peer,
        'element_size': element_size,
    }
    return zlib.crc32(repr(sorted(framed.items(), key=lambda item: item[0])).encode())


def _validate_segment(segment: TransferSegment, ndim: int) -> None:
    if len(segment.shape) != ndim:
        raise AffineTransferExecutionError('segment dimensionality does not match its logical shape')
    for size in segment.shape:
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise AffineTransferExecutionError('segment extents must be positive integers')
    for offset, strides in ((segment.source_offset, segment.source_strides), (segment.target_offset,
                                                                              segment.target_strides)):
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise AffineTransferExecutionError('segment offsets must be non-negative integers')
        if len(strides) != ndim:
            raise AffineTransferExecutionError('segment stride dimensionality does not match its shape')
        for size, stride in zip(segment.shape, strides):
            if not isinstance(stride, int) or isinstance(stride, bool) or stride < 0 or (size > 1 and stride == 0):
                raise AffineTransferExecutionError('segment strides must be positive on non-singleton axes')


def _validate_buffers(plan: TransferPlan, source_buffers: Mapping[int, torch.Tensor],
                      target_buffers: Mapping[int, torch.Tensor], dtype: torch.dtype, device: torch.device,
                      endpoints: TransferEndpoints, rank: int) -> None:
    for which, buffers, bindings in (('source', source_buffers, endpoints.source), ('target', target_buffers,
                                                                                    endpoints.target)):
        for location, buffer in buffers.items():
            if bindings.get(location) != rank:
                raise AffineTransferExecutionError(f'{which} location {location} is not owned by this process')
            _flat(location, buffers, dtype)
            if buffer.device != device:
                raise AffineTransferExecutionError(
                    f'{which} location {location}: buffer is on {buffer.device}, transport is on {device}')
    for segment in plan.segments:
        _validate_segment(segment, len(plan.logical_shape))
        for which, location, offset, strides, buffers, bindings in (('source', segment.source_rank,
                                                                     segment.source_offset, segment.source_strides,
                                                                     source_buffers, endpoints.source),
                                                                    ('target', segment.target_rank,
                                                                     segment.target_offset, segment.target_strides,
                                                                     target_buffers, endpoints.target)):
            if location not in bindings:
                raise AffineTransferExecutionError(f'{which} location {location} has no endpoint binding')
            if bindings[location] != rank:
                continue
            if location not in buffers:
                raise AffineTransferExecutionError(f'{which} location {location} is owned here but holds no buffer')
            reach = offset + sum((size - 1) * stride for size, stride in zip(segment.shape, strides))
            if reach >= buffers[location].numel():
                raise AffineTransferExecutionError(
                    f'{which} of {segment!r} reaches element {reach} of a {buffers[location].numel()}-element buffer')


def _validate_aliases(requests: Sequence[TransferRequest]) -> None:
    sources = [
        _data_range(buffer) for request in requests for buffer in request.source_buffers.values() if buffer.numel()
    ]
    targets = [
        _data_range(buffer) for request in requests for buffer in request.target_buffers.values() if buffer.numel()
    ]
    for low, high in targets:
        if any(low < source_high and source_low < high for source_low, source_high in sources):
            raise AffineTransferExecutionError('a source and a target overlap within the transfer batch')
    for index, (low, high) in enumerate(targets):
        if any(low < other_high and other_low < high for other_low, other_high in targets[:index]):
            raise AffineTransferExecutionError('two targets overlap within the transfer batch')


def execute_transfer(plan: TransferPlan,
                     source_buffers: Mapping[int, torch.Tensor],
                     target_buffers: Mapping[int, torch.Tensor],
                     endpoints: TransferEndpoints,
                     dtype: torch.dtype,
                     budget: StagingBudget = StagingBudget(),
                     group: Optional[object] = None,
                     window: Optional[TransferWindow] = None) -> TransferStats:
    """Execute one plan with the same preflight and completion contract as a batch.

    Bindings use transport-group ranks. Buffers must be packed views on the owning process;
    their storage offsets are retained. Rejection on any participant precedes all target writes.
    Transport failures after payload begins may leave partially written targets.
    """
    request = TransferRequest('parameter', plan, source_buffers, target_buffers)
    return execute_transfers([request], endpoints, dtype, budget, group, window)


def _packed_rounds(indexed_operations: Sequence[Tuple[int, _Operation]], max_elements: int):
    rounds = []
    for item, operation in indexed_operations:
        if not rounds or rounds[-1][1] + operation.elements > max_elements:
            rounds.append(([], 0))
        entries, used = rounds[-1]
        entries.append((item, operation, used))
        rounds[-1] = (entries, used + operation.elements)
    return rounds


def _direct_payload(entries, requests, dtype, source):
    if len(entries) != 1:
        return None
    item, operation, _ = entries[0]
    request, segment = requests[item], operation.segment
    if source:
        flat = _flat(segment.source_rank, request.source_buffers, dtype)
        offset, strides = operation.source_offset, segment.source_strides
    else:
        flat = _flat(segment.target_rank, request.target_buffers, dtype)
        offset, strides = operation.target_offset, segment.target_strides
    region = _region(flat, offset, operation.chunk_shape, strides)
    # A view can replace staging only when the transport sees the same packed element order.
    return region.view(-1) if region.is_contiguous() else None


def _arena_views(rounds, direct, dtype, device, element_size):
    """Stage every peer of one direction in slices of a single allocation.

    Transports that map peer buffers through IPC handles (oneCCL on XPU) pay per allocation when a
    released arena is allocated again, so one allocation per direction pays it once instead of per peer.
    """
    # 256-byte slice starts keep each peer's messages as aligned as a separate allocation would be.
    alignment = max(1, 256 // element_size)
    sizes = {}
    for peer, peer_rounds in sorted(rounds.items()):
        staged = [count for index, (_, count) in enumerate(peer_rounds) if direct[peer][index] is None]
        if staged:
            sizes[peer] = (max(staged) + alignment - 1) // alignment * alignment
    if not sizes:
        return {}
    flat = torch.empty(sum(sizes.values()), dtype=dtype, device=device)
    views, offset = {}, 0
    for peer, size in sizes.items():
        views[peer] = flat[offset:offset + size]
        offset += size
    return views


def execute_transfers(requests: Sequence[TransferRequest],
                      endpoints: TransferEndpoints,
                      dtype: torch.dtype,
                      budget: StagingBudget = StagingBudget(),
                      group: Optional[object] = None,
                      window: Optional[TransferWindow] = None) -> TransferStats:
    """Move a frozen parameter batch with collective preflight and bounded peer messages.

    Each direction stages at most ``bytes_per_peer`` per active peer. All requests and staging
    allocations are checked before payload, including read/write aliases between parameters.
    The caller stream's final unpack and local copies complete before this function returns.
    With a ``window`` (the same one on every process), operations touching window shards are copied
    in place, and no peer reads or writes them after return. A batch moved repeatedly should be
    prepared once with ``prepare_transfers`` instead.
    """
    return prepare_transfers(requests, endpoints, dtype, budget, group, window).run()


@torch.no_grad()
def prepare_transfers(requests: Sequence[TransferRequest],
                      endpoints: TransferEndpoints,
                      dtype: torch.dtype,
                      budget: StagingBudget = StagingBudget(),
                      group: Optional[object] = None,
                      window: Optional[TransferWindow] = None) -> 'PreparedTransfer':
    """Everything ``execute_transfers`` does before data moves, done once for repeated runs.

    Collective, with the same preflight and rejection contract as ``execute_transfers``. The result
    keeps the buffers, staging arenas and window offsets it was prepared with: buffers must stay where
    they are, and every process must call ``run`` the same number of times. ``run(requests)`` checks
    that they did.
    """
    import deepspeed.comm as dist

    initialized = dist.is_initialized()
    if initialized and group is not None and callable(getattr(group, 'rank', None)):
        # Ask the group itself: a weight-update group built outside the default world (as trainer <-> engine
        # groups are) maps every member's default-world rank 0 to group rank 0 in torch's bookkeeping.
        rank, world_size = group.rank(), group.size()
    else:
        rank = dist.get_rank(group=group) if initialized else 0
        world_size = dist.get_world_size(group=group) if initialized else 1
    device = torch.device(get_accelerator().current_device_name()) if initialized else torch.device('cpu')
    # Without a process group every buffer is on the host, whatever the accelerator is.
    stream_ordered = initialized and not get_accelerator().is_synchronized_device()
    indexed, local_copies = [], []
    recv_rounds, send_rounds, recv_arenas, send_arenas = {}, {}, {}, {}
    recv_direct, send_direct = {}, {}
    digest, rejected = 0, 0
    error = None
    mapped_copies, mapped_total, copiers = [], 0, set()
    offsets: Dict[Tuple, int] = {}
    if window is not None and world_size > 1:
        # Collective, so it runs before anything that may reject; a bad buffer simply publishes nothing.
        try:
            local_offsets = _window_offsets(requests, window)
        except (AttributeError, RuntimeError, TypeError):
            local_offsets = {}
        published = [None] * world_size
        dist.all_gather_object(published, local_offsets, group=group)
        for table in published:
            offsets.update(table)
    try:
        element_size = torch.empty((), dtype=dtype).element_size()
        if (not isinstance(budget.bytes_per_peer, int) or isinstance(budget.bytes_per_peer, bool)
                or budget.bytes_per_peer < element_size):
            raise AffineTransferExecutionError('the staging budget must hold at least one element')
        for bindings in (endpoints.source, endpoints.target):
            for location, process in bindings.items():
                if (not isinstance(location, int) or isinstance(location, bool) or location < 0
                        or not isinstance(process, int) or isinstance(process, bool) or not 0 <= process < world_size):
                    raise AffineTransferExecutionError('endpoint bindings must name valid transport-group ranks')
        keys = [request.key for request in requests]
        if any(not isinstance(key, str) or not key for key in keys) or len(set(keys)) != len(keys):
            raise AffineTransferExecutionError('transfer batch keys must be distinct non-empty strings')
        schedules = {}
        for item, request in enumerate(requests):
            if request.preflight_error is not None:
                raise AffineTransferExecutionError(request.preflight_error)
            if request.plan is None:
                raise AffineTransferExecutionError('a transfer request has no plan')
            _validate_buffers(request.plan, request.source_buffers, request.target_buffers, dtype, device, endpoints,
                              rank)
            # Many parameters share the same plan. Reuse only geometry within this call;
            # every request still validates its own buffers and participates in the digest.
            identity = id(request.plan)
            if identity not in schedules:
                schedules[identity] = (build_operations(request.plan, endpoints, budget, element_size),
                                       plan_digest(request.plan, endpoints, budget, element_size))
            indexed.extend((item, operation) for operation in schedules[identity][0])
        _validate_aliases(requests)
        fingerprint = (3, str(dtype), budget.bytes_per_peer, tuple(sorted(endpoints.source.items())),
                       tuple(sorted(endpoints.target.items())),
                       tuple((request.key, schedules[id(request.plan)][1]) for request in requests))
        digest = zlib.crc32(repr(fingerprint).encode())
        incoming: Dict[int, List[Tuple[int, _Operation]]] = {}
        outgoing: Dict[int, List[Tuple[int, _Operation]]] = {}
        for item, operation in indexed:
            if operation.from_process == operation.to_process == rank:
                local_copies.append((item, operation))
                continue
            direction = None
            if offsets and operation.from_process != operation.to_process:
                direction = _mapped_direction(requests[item].key, operation, offsets, window)
            if direction is not None:
                mapped_total += 1
                copier = operation.to_process if direction == 'pull' else operation.from_process
                if copier == rank:
                    mapped_copies.append((item, operation, direction))
                elif operation.from_process == rank or operation.to_process == rank:
                    copiers.add(copier)
            elif operation.to_process == rank:
                incoming.setdefault(operation.from_process, []).append((item, operation))
            elif operation.from_process == rank:
                outgoing.setdefault(operation.to_process, []).append((item, operation))
        max_elements = budget.bytes_per_peer // element_size
        recv_rounds = {peer: _packed_rounds(ops, max_elements) for peer, ops in incoming.items()}
        send_rounds = {peer: _packed_rounds(ops, max_elements) for peer, ops in outgoing.items()}
        recv_direct = {
            peer: [_direct_payload(entries, requests, dtype, False) for entries, _ in rounds]
            for peer, rounds in recv_rounds.items()
        }
        send_direct = {
            peer: [_direct_payload(entries, requests, dtype, True) for entries, _ in rounds]
            for peer, rounds in send_rounds.items()
        }
        recv_arenas = _arena_views(recv_rounds, recv_direct, dtype, device, element_size)
        send_arenas = _arena_views(send_rounds, send_direct, dtype, device, element_size)
    except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, OverflowError, MemoryError) as caught:
        error = caught
        rejected = 1
        recv_arenas, send_arenas = {}, {}
        print(f'[transfer] rank {rank} rejected the batch: {caught}', file=sys.stderr, flush=True)

    # Invalid metadata must not be inspected again while constructing the agreement payload.
    if world_size > 1:
        local = torch.tensor([digest, rejected], dtype=torch.int64, device=device)
        seen = torch.empty((world_size, 2), dtype=torch.int64, device=device)
        dist.all_gather(list(seen.unbind(0)), local, group=group)
        # Read the complete decision once instead of synchronizing each rank's scalar separately.
        decisions = seen.cpu().tolist()
        if any(pair[1] for pair in decisions):
            raise AffineTransferExecutionError(
                'a rank rejected the transfer batch before it began; no data moved') from error
        if len({pair[0] for pair in decisions}) != 1:
            raise AffineTransferExecutionError('ranks disagree about the transfer schedule (batch schedule)')
    elif rejected:
        raise AffineTransferExecutionError(f'the transfer batch was rejected before data moved: {error}') from error

    # Agreement reached: offsets are now trusted, so resolve each window copy to a pair of views.
    mapped: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = {}
    mapped_bytes = 0
    for item, operation, direction in _staggered(mapped_copies, rank):
        request, segment = requests[item], operation.segment
        if direction == 'pull':
            peer_process = operation.from_process
            peer = window.peer(peer_process, dtype)
            base = offsets[(request.key, 'source', segment.source_rank)]
            source = _region(peer, base + operation.source_offset, operation.chunk_shape, segment.source_strides)
            target = _region(_flat(segment.target_rank, request.target_buffers, dtype), operation.target_offset,
                             operation.chunk_shape, segment.target_strides)
        else:
            peer_process = operation.to_process
            peer = window.peer(peer_process, dtype)
            base = offsets[(request.key, 'target', segment.target_rank)]
            source = _region(_flat(segment.source_rank, request.source_buffers, dtype), operation.source_offset,
                             operation.chunk_shape, segment.source_strides)
            target = _region(peer, base + operation.target_offset, operation.chunk_shape, segment.target_strides)
        mapped.setdefault(peer_process, []).append((target, source))
        mapped_bytes += operation.payload_bytes

    prepared = PreparedTransfer()
    prepared._requests, prepared._dtype, prepared._group, prepared._window = requests, dtype, group, window
    prepared._element_size, prepared._stream_ordered = element_size, stream_ordered
    prepared._recv_rounds, prepared._send_rounds = recv_rounds, send_rounds
    prepared._recv_direct, prepared._send_direct = recv_direct, send_direct
    prepared._recv_arenas, prepared._send_arenas = recv_arenas, send_arenas
    prepared._local_copies, prepared._mapped, prepared._mapped_bytes = local_copies, mapped, mapped_bytes
    prepared._mapped_total, prepared._copiers = mapped_total, sorted(copiers)
    prepared._segments = sum(len(request.plan.segments) for request in requests)
    prepared._chunks = len(indexed)
    prepared._world_size, prepared._device = world_size, device
    prepared._layout = _buffer_layout(requests)
    return prepared


class PreparedTransfer:
    """A batch validated, agreed and laid out by ``prepare_transfers``; ``run`` only moves data."""
    _runs = 0

    def run(self, requests: Optional[Sequence[TransferRequest]] = None) -> TransferStats:
        """Move the batch again from the buffers' current contents. Collective; synchronous on return.

        The batch keeps the buffers and peer views it was prepared with, so a buffer moved to new storage
        since then (as ZeRO-3 ``offload_states`` / ``reload_states`` move every weight partition and an
        aliased master) would leave this writing into the old storage without error. Pass the batch's
        requests rebuilt from the live state, such as fresh ``zero_param_storage_shard`` views, to check
        first that every buffer is where it was prepared. If one moved on any process, every process
        raises before data moves, and the batch has to be prepared again. The check costs one small
        all-reduce.
        """
        if requests is not None:
            _check_layout(self, requests)
        return _run(self)


@torch.no_grad()
def _run(prepared: PreparedTransfer) -> TransferStats:
    import deepspeed.comm as dist
    requests, dtype, group, window = prepared._requests, prepared._dtype, prepared._group, prepared._window
    element_size, stream_ordered = prepared._element_size, prepared._stream_ordered
    recv_rounds, send_rounds = prepared._recv_rounds, prepared._send_rounds
    recv_direct, send_direct = prepared._recv_direct, prepared._send_direct
    recv_arenas, send_arenas = prepared._recv_arenas, prepared._send_arenas
    signals = prepared._mapped_total and window.signals
    if prepared._mapped_total and not signals and prepared._runs:
        # Owners may have rewritten their window shards since the last run; the preflight
        # agreement covered the first one.
        dist.barrier(group=group)
    if signals:
        for copier in prepared._copiers:
            window.signal(copier, window.READY)
    # Window copies go first, on their own stream, so they overlap the staged rounds below.
    if prepared._mapped:
        current = get_accelerator().current_stream()
        window.stream.wait_stream(current)
        with get_accelerator().stream(window.stream):
            for peer, copies in prepared._mapped.items():
                if signals:
                    window.wait(peer, window.READY)
                for target, source in copies:
                    target.copy_(source)
                if signals:
                    window.signal(peer, window.DONE)

    rounds = max((len(value) for value in list(recv_rounds.values()) + list(send_rounds.values())), default=0)
    sent = received = copied = 0
    pending = []

    for round_index in range(rounds):
        p2p = []
        for peer, peer_rounds in sorted(recv_rounds.items()):
            if round_index < len(peer_rounds):
                payload = recv_direct[peer][round_index]
                if payload is None:
                    payload = recv_arenas[peer][:peer_rounds[round_index][1]]
                p2p.append(('recv', payload, peer))
        for peer, peer_rounds in sorted(send_rounds.items()):
            if round_index >= len(peer_rounds):
                continue
            entries, count = peer_rounds[round_index]
            payload = send_direct[peer][round_index]
            if payload is None:
                arena = send_arenas[peer]
                for item, operation, offset in entries:
                    source = _flat(operation.segment.source_rank, requests[item].source_buffers, dtype)
                    arena[offset:offset + operation.elements].view(operation.chunk_shape).copy_(
                        _region(source, operation.source_offset, operation.chunk_shape,
                                operation.segment.source_strides))
                payload = arena[:count]
            sent += count * element_size
            p2p.append(('send', payload, peer))
        operations = [
            dist.P2POp(dist.irecv if direction == 'recv' else dist.isend,
                       buffer,
                       dist.get_global_rank(group, peer) if group is not None else peer,
                       group=group) for direction, buffer, peer in p2p
        ]
        if operations:
            for handle in dist.batch_isend_irecv(operations):
                block_stream = getattr(handle, 'block_current_stream', None)
                if stream_ordered and callable(block_stream):
                    # Order arena reuse on the caller stream; transport errors surface at the bounded drain below.
                    block_stream()
                    pending.append(handle)
                else:
                    handle.wait()
        # Stream ordering already protects arena reuse. Bound retained handles while avoiding
        # a host polling delay after every small group of rounds; all errors are observed on return.
        if (round_index + 1) % 32 == 0 or round_index + 1 == rounds:
            for handle in pending:
                handle.wait()
            pending.clear()
        for peer, peer_rounds in recv_rounds.items():
            if round_index >= len(peer_rounds):
                continue
            received += peer_rounds[round_index][1] * element_size
            if recv_direct[peer][round_index] is not None:
                continue
            for item, operation, offset in peer_rounds[round_index][0]:
                target = _flat(operation.segment.target_rank, requests[item].target_buffers, dtype)
                payload = recv_arenas[peer][offset:offset + operation.elements].view(operation.chunk_shape)
                _region(target, operation.target_offset, operation.chunk_shape,
                        operation.segment.target_strides).copy_(payload)

    for item, operation in prepared._local_copies:
        source = _flat(operation.segment.source_rank, requests[item].source_buffers, dtype)
        target = _flat(operation.segment.target_rank, requests[item].target_buffers, dtype)
        _region(target, operation.target_offset, operation.chunk_shape, operation.segment.target_strides).copy_(
            _region(source, operation.source_offset, operation.chunk_shape, operation.segment.source_strides))
        copied += operation.payload_bytes

    if prepared._mapped:
        current.wait_stream(window.stream)
    if signals:
        # Copiers are done with this process's shards before it may reuse them.
        for copier in prepared._copiers:
            window.wait(copier, window.DONE)
    if stream_ordered:
        get_accelerator().current_stream().synchronize()
    if prepared._mapped_total and not signals:
        # A pulled source or pushed target belongs to another process: neither may be reused until
        # every copier is done.
        dist.barrier(group=group)
    prepared._runs += 1
    return TransferStats(prepared._segments, prepared._chunks, rounds, copied, sent, received, prepared._mapped_bytes)


def _buffer_layout(requests: Sequence[TransferRequest]) -> Tuple:
    """Which memory each request's local buffers occupy, and under which plan."""
    layout = []
    for request in requests:
        for side, buffers in (('source', request.source_buffers), ('target', request.target_buffers)):
            for location in sorted(buffers):
                buffer = buffers[location]
                layout.append((request.key, request.plan, side, location, buffer.data_ptr(), buffer.numel(),
                               buffer.dtype, buffer.device))
    return tuple(layout)


def _check_layout(prepared: PreparedTransfer, requests: Sequence[TransferRequest]) -> None:
    import deepspeed.comm as dist
    try:
        moved = int(_buffer_layout(requests) != prepared._layout)
    except (AttributeError, TypeError):
        moved = 1
    if prepared._world_size > 1:
        # One process raising alone would leave its peers waiting in the transfer.
        flag = torch.tensor([moved], dtype=torch.int64, device=prepared._device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=prepared._group)
        moved = int(flag.item())
    if moved:
        raise AffineTransferExecutionError('a buffer of the prepared transfer moved or changed since '
                                           'prepare_transfers; prepare the batch again. No data moved.')


def _staggered(mapped_copies, rank: int):
    """This process's window copies, starting at a different peer than its neighbours.

    Every copier walking its peers in the same order makes them all read from (or write to) one peer at a
    time, which serializes on that peer's link while the others idle. Rotating the start by rank spreads
    the copiers over the peers; within a peer the plan order is kept.
    """

    def peer_of(entry):
        _, operation, direction = entry
        return operation.from_process if direction == 'pull' else operation.to_process

    peers = sorted({peer_of(entry) for entry in mapped_copies})
    if not peers:
        return []
    start = rank % len(peers)
    position = {peer: (index - start) % len(peers) for index, peer in enumerate(peers)}
    return sorted(mapped_copies, key=lambda entry: position[peer_of(entry)])


def _data_range(buffer: torch.Tensor) -> Tuple[int, int]:
    """Byte extent a tensor's elements occupy in storage."""
    start = buffer.data_ptr()
    span = buffer.numel() * buffer.element_size()
    return start, start + span


def _flat(location: int, buffers: Mapping[int, torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
    """The buffer for ``location`` as its own elements, without copying it."""
    try:
        buffer = buffers[location]
    except KeyError:
        raise AffineTransferExecutionError(f'location {location} is bound to this process but holds no buffer')
    if buffer.layout != torch.strided or not buffer.is_contiguous():
        raise AffineTransferExecutionError(f'location {location}: shard is not contiguous in its own storage')
    if buffer.dtype != dtype:
        raise AffineTransferExecutionError(f'location {location}: {buffer.dtype} is not {dtype}')
    return buffer.view(-1)


def _region(flat: torch.Tensor, offset: int, shape: Tuple[int, ...], strides: Tuple[int, ...]) -> torch.Tensor:
    """``flat`` viewed as one planned copy region.

    Plan offsets count elements from the first element of the shard; ``as_strided`` counts from the
    start of the storage. Adding the buffer's own storage offset is the explicit form of that, and the
    alternative -- making the buffer contiguous -- would copy the whole shard to fix arithmetic.
    """
    return torch.as_strided(flat, size=shape, stride=strides, storage_offset=flat.storage_offset() + offset)
