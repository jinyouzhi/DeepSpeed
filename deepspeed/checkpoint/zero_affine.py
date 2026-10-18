# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Affine maps of the storage ZeRO keeps for a dense parameter.

``autoep_storage_affine_map`` describes expert parameters; this module describes the other
parameters of a ZeRO model, so a transfer planner sees every legal holder of every element.
Both describe the flattened parameter keyed by global rank, so their maps plan together.
"""

import deepspeed.comm as dist
from deepspeed.checkpoint.affine import AffinePiece, ParamAffineMap

__all__ = ['zero_storage_affine_map', 'zero_param_storage_affine_map']


def zero_storage_affine_map(numel, holder_ranks, partition_numel=None):
    """Describe a flattened parameter of ``numel`` elements stored by ``holder_ranks``.

    Without ``partition_numel`` every holder stores the whole parameter, as data-parallel
    replicas do under ZeRO-0/1/2, so the one piece lists every holder. With it, holder ``i``
    stores elements ``[i * n, (i + 1) * n)`` as ZeRO-3 does; the trailing padding holds no
    parameter elements and is not part of the shard, so a holder past the end has an empty shard.
    """
    numel = int(numel)
    holder_ranks = [int(rank) for rank in holder_ranks]
    if numel < 0:
        raise ValueError(f'Parameter numel cannot be negative: {numel}.')
    if not holder_ranks or len(set(holder_ranks)) != len(holder_ranks):
        raise ValueError(f'Holder ranks must be a non-empty list of distinct ranks: {holder_ranks}.')

    if partition_numel is None:
        piece = AffinePiece(shape=(numel, ),
                            source_offset=0,
                            source_strides=(1, ),
                            dest_offset=0,
                            dest_strides=(1, ),
                            locations=holder_ranks)
        affine_map = ParamAffineMap(logical_shape=(numel, ),
                                    shard_shapes={rank: (numel, )
                                                  for rank in holder_ranks},
                                    pieces_by_rank={rank: [piece]
                                                    for rank in holder_ranks})
        affine_map.validate()
        return affine_map

    partition_numel = int(partition_numel)
    if partition_numel < 0 or partition_numel * len(holder_ranks) < numel:
        raise ValueError(f'{len(holder_ranks)} partitions of {partition_numel} elements cannot hold {numel} elements.')
    shard_shapes = {}
    pieces_by_rank = {}
    for index, rank in enumerate(holder_ranks):
        start = min(index * partition_numel, numel)
        stop = min(start + partition_numel, numel)
        shard_shapes[rank] = (stop - start, )
        pieces_by_rank[rank] = []
        if stop > start:
            pieces_by_rank[rank].append(
                AffinePiece(shape=(stop - start, ),
                            source_offset=start,
                            source_strides=(1, ),
                            dest_offset=0,
                            dest_strides=(1, ),
                            locations=[rank]))
    affine_map = ParamAffineMap(logical_shape=(numel, ), shard_shapes=shard_shapes, pieces_by_rank=pieces_by_rank)
    affine_map.validate()
    return affine_map


def zero_param_storage_affine_map(param, replica_group=None):
    """Describe the storage of a live dense ``param`` across global ranks.

    A ZeRO-3 parameter is described by its own partition group and ``ds_tensor`` size.
    Otherwise every rank of ``replica_group`` holds a whole copy; it defaults to the
    data-parallel group. Pass the group whose ranks hold identical values: under tensor
    parallelism the data-parallel peers of a rank hold its TP shard, not the full parameter.
    """
    if hasattr(param, 'ds_id'):
        holders = dist.get_all_ranks_from_group(param.ds_process_group)
        return zero_storage_affine_map(param.ds_numel, holders, param.ds_tensor.ds_numel)
    if replica_group is None:
        from deepspeed.utils import groups
        replica_group = groups._get_data_parallel_group()
    return zero_storage_affine_map(param.numel(), dist.get_all_ranks_from_group(replica_group))
