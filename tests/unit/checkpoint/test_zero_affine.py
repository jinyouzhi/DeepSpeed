# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import pytest
import torch

from deepspeed.checkpoint.affine import contiguous_split_map
from deepspeed.checkpoint.affine_transfer import PlanBudget, plan_transfer
from deepspeed.checkpoint.zero_affine import zero_storage_affine_map


def test_replicas_list_every_holder():
    full = torch.arange(6.)
    affine_map = zero_storage_affine_map(6, [4, 1, 6])

    assert set(affine_map.shard_shapes) == {1, 4, 6}
    for rank in (1, 4, 6):
        assert torch.equal(affine_map.extract(full, rank), full)

    asked = []

    def selector(request):
        asked.append(sorted(candidate.source_rank for candidate in request.candidates))
        return request.candidates[-1].source_rank

    plan_transfer(contiguous_split_map((6, ), [3, 3], 0), affine_map, PlanBudget(), replica_selector=selector)
    # Each target block may be read from any of the three replicas.
    assert asked == [[1, 4, 6], [1, 4, 6]]


def test_zero3_partitions_drop_padding():
    # 10 elements in partitions of 4: holders keep 4, 4 and 2 elements; the fourth is all padding.
    full = torch.arange(10.)
    affine_map = zero_storage_affine_map(10, [2, 0, 3, 1], partition_numel=4)

    assert affine_map.shard_shapes == {2: (4, ), 0: (4, ), 3: (2, ), 1: (0, )}
    assert torch.equal(affine_map.extract(full, 2), full[0:4])
    assert torch.equal(affine_map.extract(full, 0), full[4:8])
    assert torch.equal(affine_map.extract(full, 3), full[8:10])
    assert affine_map.extract(full, 1).numel() == 0


@pytest.mark.parametrize('numel, holders, partition', [(9, [0, 1], 4), (4, [0, 0], None), (4, [], None)])
def test_rejects_storage_that_cannot_hold_the_parameter(numel, holders, partition):
    with pytest.raises(ValueError):
        zero_storage_affine_map(numel, holders, partition)
