# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Prototype: route segKI decode attention / GDN through vLLM's kernels.

Full attention: one Triton launch does q/k RMSNorm(1+w) + partial RoPE and
writes K/V into an NHD cache, then vLLM's FA3 varlen kernel attends with
per-row ``seqused_k``. GDN: vLLM's packed fused-recurrent decode kernel on a
[V, K]-major state (slot 0 is vLLM's null block).
"""

import triton
import triton.language as tl


def fa3_available():
    try:
        from vllm.vllm_flash_attn import flash_attn_varlen_func, is_fa_version_supported  # noqa: F401
        return is_fa_version_supported(3)
    except Exception:
        return False


def gdn_packed_available():
    try:
        from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (  # noqa: F401
            fused_recurrent_gated_delta_rule_packed_decode)
        return True
    except Exception:
        return False


@triton.jit
def _qk_norm_rope_store_kernel(q_raw, kv_raw, q_out, k_cache, v_cache, pos_ptr, qw_ptr, kw_ptr, cos_ptr, sin_ptr,
                               q_raw_row, q_raw_head, kv_raw_row, eps, L, HQ: tl.constexpr, KV: tl.constexpr,
                               D: tl.constexpr, ROT: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, D)
    half: tl.constexpr = ROT // 2
    pos = tl.load(pos_ptr + b)
    is_q = h < HQ
    if is_q:
        src = q_raw + b * q_raw_row + h * q_raw_head
        w_ptr = qw_ptr
    else:
        src = kv_raw + b * kv_raw_row + (h - HQ) * D
        w_ptr = kw_ptr
    x = tl.load(src + d).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, 0) / D + eps)
    w = tl.load(w_ptr + d)
    xn = x * rstd * w
    in_rot = d < ROT
    partner = tl.where(d < half, d + half, d - half)
    partner = tl.where(in_rot, partner, d)
    xp = tl.load(src + partner).to(tl.float32) * rstd * tl.load(w_ptr + partner)
    rd = tl.where(in_rot, d, 0)
    c = tl.load(cos_ptr + pos * ROT + rd, mask=in_rot, other=1.0)
    s = tl.load(sin_ptr + pos * ROT + rd, mask=in_rot, other=0.0)
    sign = tl.where(d < half, -1.0, 1.0)
    y = tl.where(in_rot, xn * c + sign * xp * s, xn)
    if is_q:
        tl.store(q_out + (b * HQ + h) * D + d, y.to(q_out.dtype.element_ty))
    else:
        kh = h - HQ
        dst = ((b * L + pos) * KV + kh) * D
        tl.store(k_cache + dst + d, y.to(k_cache.dtype.element_ty))
        v = tl.load(kv_raw + b * kv_raw_row + KV * D + kh * D + d)
        tl.store(v_cache + dst + d, v)


def qk_norm_rope_store(q_raw, kv_raw, q_out, k_cache, v_cache, pos, q_w, k_w, cos_t, sin_t, eps):
    """q_raw: [B, HQ, D] view (row/head strides arbitrary, last dim dense);
    kv_raw: [B, 2*KV*D] rows holding [k | v]; caches: [B, L, KV, D] dense."""
    B, HQ, D = q_raw.shape
    L, KV = k_cache.shape[1], k_cache.shape[2]
    _qk_norm_rope_store_kernel[(B, HQ + KV)](q_raw, kv_raw, q_out, k_cache, v_cache, pos, q_w, k_w, cos_t, sin_t,
                                             q_raw.stride(0), q_raw.stride(1), kv_raw.stride(0), eps, L, HQ, KV, D,
                                             cos_t.shape[-1])


def fa3_decode(q, k_cache, v_cache, seqused_k, cu_q, block_table, scale, out):
    # One page per row: block_table[b] = [b], page size = cache length.
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    L = k_cache.shape[1]
    return flash_attn_varlen_func(q,
                                  k_cache,
                                  v_cache,
                                  max_seqlen_q=1,
                                  cu_seqlens_q=cu_q,
                                  max_seqlen_k=L,
                                  seqused_k=seqused_k,
                                  block_table=block_table,
                                  softmax_scale=scale,
                                  causal=False,
                                  out=out,
                                  fa_version=3)


def gdn_packed_decode(mixed_qkv, a, b, A_log, dt_bias, scale, state, out, indices):
    from vllm.third_party.flash_linear_attention.ops.fused_recurrent import \
        fused_recurrent_gated_delta_rule_packed_decode
    fused_recurrent_gated_delta_rule_packed_decode(mixed_qkv=mixed_qkv,
                                                   a=a,
                                                   b=b,
                                                   A_log=A_log,
                                                   dt_bias=dt_bias,
                                                   scale=scale,
                                                   initial_state=state,
                                                   out=out,
                                                   ssm_state_indices=indices,
                                                   use_qk_l2norm_in_kernel=True)
    return out
