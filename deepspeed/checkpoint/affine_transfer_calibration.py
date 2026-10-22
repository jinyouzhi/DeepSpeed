# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Measure the links a transfer plan uses, as raw times by message size.

``affine_transfer_analysis`` reports what a plan will move and over which path classes; this
module measures how long those paths take on the machine at hand. Spec-sheet bandwidth is not a
usable substitute: what a transfer sees depends on the communication library, on how many links
share a switch or root port, and on fixed per-message costs, none of which a datasheet states.

``calibrate_links`` records raw samples; ``fit_links`` summarizes them per path as a fixed cost
plus a bandwidth, with the fit error alongside so a consumer can see where that form does not
hold. Modeling contention beyond the measured patterns, and choosing plans, belong to the
planner's policy layer.
"""

import statistics
import time
from typing import Callable, Dict, List, Optional, Sequence

import torch

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator

from .affine_transfer_analysis import LOCAL, default_path_of

__all__ = ['CALIBRATION_SCHEMA', 'DEFAULT_SIZES', 'calibrate_links', 'fit_links']

CALIBRATION_SCHEMA = 1

# 4 KiB to 64 MiB, so both the latency-bound and the bandwidth-bound ends of every path are sampled.
DEFAULT_SIZES = tuple(1 << shift for shift in range(12, 27, 2))


def _synchronize() -> None:
    get_accelerator().synchronize()


def _exchange(send: torch.Tensor, recv: torch.Tensor, to_process: Optional[int], from_process: Optional[int]) -> None:
    ops = []
    # oneCCL hangs on a group call that starts with a receive, so sends always go first.
    if to_process is not None:
        ops.append(dist.P2POp(dist.isend, send, to_process))
    if from_process is not None:
        ops.append(dist.P2POp(dist.irecv, recv, from_process))
    if ops:
        for handle in dist.batch_isend_irecv(ops):
            handle.wait()


def _timed_exchange(send, recv, to_process, from_process) -> float:
    dist.barrier()
    _synchronize()
    started = time.perf_counter()
    _exchange(send, recv, to_process, from_process)
    _synchronize()
    return time.perf_counter() - started


def _timed_copy(send: torch.Tensor, recv: torch.Tensor) -> float:
    _synchronize()
    started = time.perf_counter()
    recv.copy_(send)
    _synchronize()
    return time.perf_counter() - started


def calibrate_links(sizes: Sequence[int] = DEFAULT_SIZES,
                    repeats: int = 10,
                    warmup: int = 2,
                    path_of: Callable[[int, int], str] = default_path_of) -> Dict:
    """Time local copies, every ordered process pair alone, and all processes at once.

    Every process of the default group must call this. It returns the same dictionary on every
    process, with ``repeats`` samples per entry and no aggregation:

    - ``local``: a device-local copy on each process.
    - ``peer``: each ordered pair with every other process idle, so that links through different
      switches and sockets are each sampled. A sample is the slower of its two endpoints.
    - ``concurrent``: every process sends to ``process + shift`` and receives from
      ``process - shift`` at the same time, for every shift, so links are sampled while sharing.

    ``path`` on each sample is ``path_of(from_process, to_process)``, the same classification the
    analysis report uses, so measured times can be joined to a report's messages by path.

    The pair sweep makes ``world_size * (world_size - 1) * len(sizes) * (warmup + repeats)``
    barriers, so keep ``sizes`` short on large jobs.
    """
    if repeats <= 0 or warmup < 0:
        raise ValueError('repeats must be positive and warmup non-negative')
    if any(size <= 0 for size in sizes):
        raise ValueError('sizes must be positive byte counts')

    rank = dist.get_rank()
    world = dist.get_world_size()
    device = torch.device(get_accelerator().current_device_name())
    local_times: Dict[int, List[float]] = {}
    peer_times: Dict[tuple, List[float]] = {}
    concurrent_times: Dict[tuple, List[float]] = {}
    for size in sizes:
        send = torch.ones(size, dtype=torch.uint8, device=device)
        recv = torch.empty(size, dtype=torch.uint8, device=device)

        times = [_timed_copy(send, recv) for _ in range(warmup + repeats)]
        local_times[size] = times[warmup:]

        for source in range(world):
            for target in range(world):
                if source == target:
                    continue
                to_process = target if rank == source else None
                from_process = source if rank == target else None
                times = [_timed_exchange(send, recv, to_process, from_process) for _ in range(warmup + repeats)]
                if rank in (source, target):
                    peer_times[(size, source, target)] = times[warmup:]

        for shift in range(1, world):
            to_process = (rank + shift) % world
            from_process = (rank - shift) % world
            times = [_timed_exchange(send, recv, to_process, from_process) for _ in range(warmup + repeats)]
            concurrent_times[(size, shift)] = times[warmup:]

    # Samples stay on their own process until every timing is done, so no collective runs inside a timed region.
    gathered = [None] * world
    dist.all_gather_object(gathered, (local_times, peer_times, concurrent_times))

    local, peer, concurrent = [], [], []
    for size in sizes:
        for process in range(world):
            local.append({'bytes': size, 'process': process, 'path': LOCAL, 'seconds': gathered[process][0][size]})
        for source in range(world):
            for target in range(world):
                if source == target:
                    continue
                key = (size, source, target)
                seconds = [max(pair) for pair in zip(gathered[source][1][key], gathered[target][1][key])]
                peer.append({
                    'bytes': size,
                    'from_process': source,
                    'to_process': target,
                    'path': path_of(source, target),
                    'seconds': seconds
                })
        for shift in range(1, world):
            concurrent.append({
                'bytes': size,
                'shift': shift,
                'paths': [path_of(process, (process + shift) % world) for process in range(world)],
                'seconds_by_process': [gathered[process][2][(size, shift)] for process in range(world)],
            })
    return {
        'schema': CALIBRATION_SCHEMA,
        'world_size': world,
        'communication_backend': get_accelerator().communication_backend_name(),
        'repeats': repeats,
        'warmup': warmup,
        'local': local,
        'peer': peer,
        'concurrent': concurrent,
    }


def _fit(points: List[tuple]) -> Dict:
    """Fit ``seconds = alpha + bytes / bandwidth`` to (bytes, seconds) points.

    Residuals are weighted by 1/seconds, so the fit minimizes relative error. Unweighted least
    squares would let the largest messages decide everything and leave alpha meaningless.
    """
    weights = [1.0 / (seconds * seconds) for _, seconds in points]
    total = sum(weights)
    mean_x = sum(w * x for w, (x, _) in zip(weights, points)) / total
    mean_y = sum(w * y for w, (_, y) in zip(weights, points)) / total
    spread = sum(w * (x - mean_x)**2 for w, (x, _) in zip(weights, points))
    if spread == 0:
        raise ValueError('a fit needs at least two distinct message sizes')
    slope = sum(w * (x - mean_x) * (y - mean_y) for w, (x, y) in zip(weights, points)) / spread
    alpha = mean_y - slope * mean_x
    predicted = [alpha + slope * x for x, _ in points]
    return {
        'alpha_seconds': alpha,
        # A non-positive slope means time did not grow with size, so no bandwidth is implied.
        'bytes_per_second': 1.0 / slope if slope > 0 else None,
        'max_relative_error': max(abs(p - y) / y for p, (_, y) in zip(predicted, points)),
        'sizes': sorted({x
                         for x, _ in points}),
    }


def _fit_groups(groups: Dict[str, Dict[int, List[float]]]) -> Dict[str, Dict]:
    fits = {}
    for path, seconds_by_size in groups.items():
        points = [(size, statistics.median(seconds)) for size, seconds in sorted(seconds_by_size.items())]
        fits[path] = _fit(points)
    return fits


def fit_links(calibration: Dict) -> Dict:
    """Summarize a ``calibrate_links`` result as ``alpha + bytes / bandwidth`` per path.

    Samples are pooled by path label and by size, and the median at each size is fitted, so a
    single slow repeat does not move the line. Returns ``{'schema', 'local', 'peer',
    'concurrent'}``; each of the last three maps a path label to ``alpha_seconds``,
    ``bytes_per_second``, ``max_relative_error`` and the ``sizes`` fitted. ``concurrent`` is
    fitted from what each process saw while every process was sending, i.e. per-link cost under
    the all-to-all-shift sharing pattern, not in isolation.
    """
    if calibration.get('schema') != CALIBRATION_SCHEMA:
        raise ValueError(f'unsupported calibration schema {calibration.get("schema")!r}')

    local: Dict[str, Dict[int, List[float]]] = {}
    for sample in calibration['local']:
        local.setdefault(sample['path'], {}).setdefault(sample['bytes'], []).extend(sample['seconds'])

    peer: Dict[str, Dict[int, List[float]]] = {}
    for sample in calibration['peer']:
        peer.setdefault(sample['path'], {}).setdefault(sample['bytes'], []).extend(sample['seconds'])

    concurrent: Dict[str, Dict[int, List[float]]] = {}
    for sample in calibration['concurrent']:
        for path, seconds in zip(sample['paths'], sample['seconds_by_process']):
            concurrent.setdefault(path, {}).setdefault(sample['bytes'], []).extend(seconds)

    return {
        'schema': CALIBRATION_SCHEMA,
        'local': _fit_groups(local),
        'peer': _fit_groups(peer),
        'concurrent': _fit_groups(concurrent),
    }
