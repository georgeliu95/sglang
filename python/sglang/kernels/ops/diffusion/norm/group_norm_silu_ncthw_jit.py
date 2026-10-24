"""Fused fp32 GroupNorm + SiLU over NCTHW, with per-frame statistics on request.

``group_norm_silu_ncthw(x, weight, bias, num_groups=, eps=, time_isolated=)``
replaces ``F.silu(F.group_norm(x, ...))`` on a 5-D ``[B, C, T, H, W]`` (or 4-D
``[B, C, H, W]``) fp32 tensor and returns a contiguous fp32 tensor of the same
shape. ``time_isolated=True`` takes the statistics per (batch, frame, group);
the eager chain is ``permute -> reshape(B*T, C, H, W) -> group_norm``, which
is what a causal video VAE encoder needs so that no frame sees another.

Contract: close, not bit-exact. Statistics are exact two-pass fp32 central
moments and the epilogue is the direct ``((x - mean) * rstd) * gamma + beta``
with one rounding per op, so a constant group is exactly ``silu(beta)`` and
everything else differs from eager only by fp32 reduction order. Mount it
behind the quality gate, never on the ``lossless`` path.

The vectorized path (128-bit loads) is picked here for contiguous, 16-byte
aligned input with ``W % 4 == 0``; every other layout, including permuted
views and spatial slices, runs the strided path with the same arithmetic.

Verified shapes (SM100): the MiniMax-H3 encoder ladder
``[7, 128, 1, 256, 256]`` .. ``[7, 1024, 1, 16, 16]`` with 32 groups,
``[1, 256, 9, 64, 64]`` per frame, plus odd strided and 4-D cases in
``test/registered/kernels/ops/diffusion/test_group_norm_silu_ncthw.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.kernels.ops.diffusion.common.platform import is_cuda_sm_at_least
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_MIN_SM = (10, 0)
# Must match kVecWidth / kVecAlignment in csrc/diffusion/group_norm_silu_ncthw.cuh.
_VEC_WIDTH = 4
_VEC_ALIGNMENT = 16


@cache_once
def _jit_group_norm_silu_ncthw_module(vectorized: bool) -> Module:
    args = make_cpp_args(vectorized)
    return load_jit(
        "diffusion_group_norm_silu_ncthw",
        *args,
        cuda_files=["diffusion/group_norm_silu_ncthw.cuh"],
        cuda_wrappers=[
            (
                "group_norm_silu_ncthw",
                f"group_norm_silu_ncthw::GroupNormSiluNcthwKernel<{args}>::run",
            ),
        ],
    )


@register_custom_op(op_name="diffusion_group_norm_silu_ncthw", mutates_args=["out"])
def _group_norm_silu_ncthw_out(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    num_groups: int,
    time_isolated: bool,
    eps: float,
    vectorized: bool,
) -> None:
    module = _jit_group_norm_silu_ncthw_module(vectorized)
    module.group_norm_silu_ncthw(x, weight, bias, out, num_groups, time_isolated, eps)


def _vectorizable(x: torch.Tensor) -> bool:
    return (
        x.is_contiguous()
        and x.shape[-1] % _VEC_WIDTH == 0
        and x.data_ptr() % _VEC_ALIGNMENT == 0
    )


def group_norm_silu_ncthw(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    num_groups: int,
    eps: float,
    time_isolated: bool = False,
) -> torch.Tensor:
    """``F.silu(F.group_norm(x, num_groups, weight, bias, eps))`` as one launch.

    A 4-D input is treated as a single frame. The C++ launcher validates every
    tensor and raises on a contract violation.
    """
    value = x.unsqueeze(2) if x.ndim == 4 else x
    out = torch.empty(value.shape, dtype=torch.float32, device=value.device)
    _group_norm_silu_ncthw_out(
        value,
        weight,
        bias,
        out,
        num_groups,
        time_isolated,
        float(eps),
        _vectorizable(value),
    )
    return out.squeeze(2) if x.ndim == 4 else out


def can_use_group_norm_silu_ncthw(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    num_groups: int,
    eps: float,
) -> bool:
    return (
        isinstance(x, torch.Tensor)
        and x.is_cuda
        and x.dtype == torch.float32
        and x.ndim in (4, 5)
        and x.numel() > 0
        and num_groups > 0
        and x.shape[1] % num_groups == 0
        and eps > 0
        and all(
            isinstance(param, torch.Tensor)
            and param.device == x.device
            and param.dtype == torch.float32
            and param.shape == (x.shape[1],)
            and param.is_contiguous()
            for param in (weight, bias)
        )
        and is_cuda_sm_at_least(_MIN_SM, x.device)
        and not torch.compiler.is_compiling()
    )


__all__ = ["can_use_group_norm_silu_ncthw", "group_norm_silu_ncthw"]
