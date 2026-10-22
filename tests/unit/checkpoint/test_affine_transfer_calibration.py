# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import json
import os

import pytest

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.checkpoint.affine_transfer_analysis import INTER_NODE, INTRA_NODE, LOCAL, node_path_of
from deepspeed.checkpoint.affine_transfer_calibration import CALIBRATION_SCHEMA, calibrate_links
from unit.common import DistributedTest


def test_rejects_empty_samples():
    with pytest.raises(ValueError):
        calibrate_links(repeats=0)
    with pytest.raises(ValueError):
        calibrate_links(sizes=[0])


@pytest.mark.skipif(get_accelerator().device_count() < 4, reason='requires four visible accelerator devices')
class TestCalibrateLinks(DistributedTest):
    world_size = 4

    def test_every_path_is_sampled_and_labeled(self):
        get_accelerator().set_device(int(os.environ.get('LOCAL_RANK', dist.get_rank())))
        sizes = [4096, 1 << 20]
        # Pretend two processes per node so both peer path classes occur on one machine.
        result = calibrate_links(sizes=sizes, repeats=3, warmup=1, path_of=node_path_of(2))

        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, json.dumps(result, sort_keys=True))
        assert len(set(gathered)) == 1, 'every process must return the same result'

        world = 4
        assert result['schema'] == CALIBRATION_SCHEMA
        assert result['world_size'] == world
        assert sorted((s['bytes'], s['process']) for s in result['local']) == sorted(
            (size, process) for size in sizes for process in range(world))
        assert sorted((s['bytes'], s['from_process'], s['to_process']) for s in result['peer']) == sorted(
            (size, source, target) for size in sizes for source in range(world) for target in range(world)
            if source != target)
        assert sorted((s['bytes'], s['shift']) for s in result['concurrent']) == sorted(
            (size, shift) for size in sizes for shift in range(1, world))

        for sample in result['local']:
            assert sample['path'] == LOCAL
        for sample in result['peer']:
            expected = INTRA_NODE if sample['from_process'] // 2 == sample['to_process'] // 2 else INTER_NODE
            assert sample['path'] == expected
        for sample in result['concurrent']:
            assert sample['paths'] == [node_path_of(2)(p, (p + sample['shift']) % world) for p in range(world)]
            assert len(sample['seconds_by_process']) == world

        samples = [s['seconds'] for s in result['local'] + result['peer']]
        samples += [seconds for s in result['concurrent'] for seconds in s['seconds_by_process']]
        for seconds in samples:
            assert len(seconds) == 3
            assert all(value > 0 for value in seconds)
