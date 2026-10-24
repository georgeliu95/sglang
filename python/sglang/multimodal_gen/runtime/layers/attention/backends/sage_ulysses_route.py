# SPDX-License-Identifier: Apache-2.0
"""Rank-invariant H3 geometry guards before prepared Sage QKV collectives."""

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_group,
    get_ulysses_parallel_world_size,
)


def check_v2g_route(
    q,
    k,
    v,
    *,
    cu_seqlens_host,
    max_seqlen,
    ulysses_active,
    ring_active,
    capability_check=None,
    group_agreement_check=None,
):
    def result(reason=None):
        return SimpleNamespace(enabled=reason is None, reason=reason)

    if os.environ.get("SGLANG_DIFFUSION_DISABLE_SM120_LOWP_A2A_V2G", "0").lower() in (
        "1",
        "true",
    ):
        return result("disabled")
    if not ulysses_active or ring_active:
        return result("not pure Ulysses")
    if torch.is_grad_enabled() or any(x.requires_grad for x in (q, k, v)):
        return result("training")
    if torch.compiler.is_compiling() or not q.is_cuda:
        return result("not eager CUDA")
    capability = torch.cuda.get_device_capability(q.device)
    gb10_delivery = (
        os.environ.get("SGLANG_H3_GB10_DELIVERY") == "1"
        and capability == (12, 1)
    )
    if (
        (capability != (12, 0) and not gb10_delivery)
        or torch.cuda.is_current_stream_capturing()
    ):
        return result("not eager SM120")
    if q.ndim != 3 or not (
        q.shape == k.shape == v.shape
        and q.dtype == k.dtype == v.dtype
        and q.device == k.device == v.device
    ):
        return result("QKV mismatch")
    if q.dtype not in (torch.bfloat16, torch.float16) or q.shape[-1] != 128:
        return result("unsupported dtype/head dimension")
    world = get_ulysses_parallel_world_size()
    if world not in ((2, 4) if gb10_delivery else (4, 8)) or q.shape[1] % world:
        return result("unsupported Ulysses geometry")
    total = q.shape[0] * world
    cu = cu_seqlens_host
    if (
        cu is None
        or len(cu) != 3
        or cu[0] != 0
        or cu[2] != total
        or not 0 < cu[1] <= total
        or cu[1] != max_seqlen
    ):
        return result("unsupported packed documents")
    group = get_sp_group().ulysses_group
    if group is None or dist.get_world_size(group) != world:
        return result("process group mismatch")
    return result()
