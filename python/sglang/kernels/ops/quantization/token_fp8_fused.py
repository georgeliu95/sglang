"""Producer-fused dynamic per-token E4M3 quantization for the DiT FP8 linears.

The unfused chain writes a BF16 activation, then ``sglang_per_token_quant_fp8`` (the JIT
``per_token_quant_fp8_warp_kernel``, jit/csrc/gemm/per_token_quant_fp8.cuh, built with precise math everywhere but
SM90) reads it back to produce ``(e4m3 rows, fp32 scale per row)``. These kernels do the quantization in the
producer's own pass, on the BF16-rounded values the producer would have written, with the same arithmetic:

* ``scale = amax / 448`` and ``inv = 1 / scale`` (0 for a zero row): IEEE divisions (``div.rn``).
* ``q = cvt.rn.satfinite.e4m3(max(min(x * inv, 448), -448))``.

so the outputs are byte-identical to the unfused chain (``test_token_fp8_fusion.py``). One exception, by design: an
all-zero row gives scale 0 and zero bytes, as the warp kernel does; below 2 * 8 * SMs rows the unfused chain runs
``per_token_quant_fp8_cta_kernel``, which has no scale == 0 guard and writes 448 (0x7e) bytes for such a row (they
dequantize to 0 all the same). H3's per-rank token counts (4728) are on the warp path.

* ``rmsnorm_indexed_scale_shift_fp8``: ``rmsnorm(x) * (1 + scale[idx]) + shift[idx]`` (the
  ``rmsnorm_indexed_scale_shift`` kernel's math), optionally also storing the BF16 rows.
* ``silu_mul_fp8``: SwiGLU with activation rounding (``silu_and_mul_with_activation_rounding``: the SiLU output is
  rounded to BF16 before the multiply), BF16 product rows quantized; the BF16 rows are never written.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

FP8_MAX = 448.0


@triton.jit
def _token_scale_inv(amax):
    """(scale, inverse) of a row amax exactly as the JIT per-token quantizer computes them (precise math)."""
    scale = tl.math.div_rn(amax, 448.0)
    inv = tl.where(scale == 0.0, 0.0, tl.math.div_rn(1.0, tl.where(scale == 0.0, 1.0, scale)))
    return scale, inv


@triton.jit
def _quantize_row(values_bf16, mask):
    """values_bf16: one row (BF16, masked lanes ignored) -> (e4m3 row, fp32 scale)."""
    v = values_bf16.to(tl.float32)
    amax = tl.max(tl.where(mask, tl.abs(v), 0.0), axis=0)
    scale, inv = _token_scale_inv(amax)
    q = tl.maximum(tl.minimum(v * inv, 448.0), -448.0).to(tl.float8e4nv)
    return q, scale


@triton.jit
def _rmsnorm_indexed_scale_shift_fp8_kernel(
    y_ptr,
    q_ptr,
    s_ptr,
    x_ptr,
    w_ptr,
    scale_ptr,
    shift_ptr,
    idx_ptr,
    mod_row_stride,
    idx_stride,
    DIM: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE_DIM: tl.constexpr,
    STORE_Y: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    # Contiguity capped at 8 (one 16-byte BF16 vector): left alone, Triton coalesces the 1-byte e4m3 store at 16 per
    # thread and lays the whole row out that way, which reorders the fp32 sum of squares against the unfused kernel
    # (8 per thread, set by its BF16 loads/stores) and moves rstd by an ulp on some rows (5376-wide rows: 177 of
    # 4728 BF16 rows differed). Same layout -> same reduction tree -> same bits.
    cols = tl.max_contiguous(tl.arange(0, BLOCK_SIZE_DIM), 8)
    mask = cols < DIM
    mod_ptr = tl.load(idx_ptr + row * idx_stride) * mod_row_stride + cols

    # Same expression as _rmsnorm_indexed_scale_shift_kernel (rmsnorm_scale_shift_triton.py).
    x = tl.load(x_ptr + row * DIM + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.math.rsqrt(tl.sum(x * x, axis=0) / DIM + EPS)
    w = tl.load(w_ptr + cols, mask=mask).to(tl.float32)
    scale = tl.load(scale_ptr + mod_ptr, mask=mask).to(tl.float32)
    shift = tl.load(shift_ptr + mod_ptr, mask=mask).to(tl.float32)
    y = (x * rstd * w) * (1.0 + scale) + shift
    yb = y.to(tl.bfloat16)
    if STORE_Y:
        tl.store(y_ptr + row * DIM + cols, yb, mask=mask)
    q, s = _quantize_row(yb, mask)
    tl.store(q_ptr + row * DIM + cols, q, mask=mask)
    tl.store(s_ptr + row, s)


@triton.jit
def _silu_mul_fp8_kernel(q_ptr, s_ptr, x_ptr, HIDDEN: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < HIDDEN
    gate = tl.load(x_ptr + row * (2 * HIDDEN) + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_ptr + row * (2 * HIDDEN) + HIDDEN + cols, mask=mask, other=0.0).to(tl.float32)
    # activation.cuh, kRoundActivation (fast_math=False): bf16(g / (1 + expf(-g))), then bf16(f32(act) * up).
    act = tl.math.div_rn(gate, 1.0 + libdevice.exp(-gate)).to(tl.bfloat16)
    prod = (act.to(tl.float32) * up).to(tl.bfloat16)
    q, s = _quantize_row(prod, mask)
    tl.store(q_ptr + row * HIDDEN + cols, q, mask=mask)
    tl.store(s_ptr + row, s)


def can_use_rmsnorm_indexed_scale_shift_fp8(x, weight, shift, scale, indices) -> bool:
    return (
        x.is_cuda
        and x.dim() == 2
        and x.is_contiguous()
        and x.dtype == torch.bfloat16
        and x.numel() > 0
        and weight.is_cuda
        and weight.dim() == 1
        and weight.numel() == x.shape[1]
        and shift.dim() == scale.dim() == 2
        and shift.shape == scale.shape
        and shift.shape[1] == x.shape[1]
        and shift.stride(0) == scale.stride(0)
        and shift.stride(1) == scale.stride(1) == 1
        and indices.dim() == 1
        and indices.numel() == x.shape[0]
        and not indices.dtype.is_floating_point
    )


def rmsnorm_indexed_scale_shift_fp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
    *,
    keep_bf16: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """(bf16 rows or None, e4m3 rows [T, D], fp32 per-row scales [T, 1]) of rmsnorm_indexed_scale_shift."""
    rows, dim = x.shape
    y = torch.empty_like(x) if keep_bf16 else None
    q = torch.empty((rows, dim), dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty((rows, 1), dtype=torch.float32, device=x.device)
    with torch.get_device_module().device(x.device):
        _rmsnorm_indexed_scale_shift_fp8_kernel[(rows,)](
            y if keep_bf16 else q,
            q,
            s,
            x,
            weight,
            scale,
            shift,
            indices,
            scale.stride(0),
            indices.stride(0),
            DIM=dim,
            EPS=eps,
            BLOCK_SIZE_DIM=triton.next_power_of_2(dim),
            STORE_Y=keep_bf16,
            num_warps=8,
        )
    return y, q, s


def can_use_silu_mul_fp8(hidden: torch.Tensor) -> bool:
    return (
        hidden.is_cuda
        and hidden.dim() == 2
        and hidden.dtype == torch.bfloat16
        and hidden.is_contiguous()
        and hidden.shape[-1] % 2 == 0
        and hidden.numel() > 0
    )


def silu_mul_fp8(hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """hidden [T, 2F] BF16 ([gate | up]) -> (e4m3 [T, F], fp32 scales [T, 1]) of the rounded SwiGLU rows."""
    rows, width = hidden.shape
    half = width // 2
    q = torch.empty((rows, half), dtype=torch.float8_e4m3fn, device=hidden.device)
    s = torch.empty((rows, 1), dtype=torch.float32, device=hidden.device)
    block = triton.next_power_of_2(half)
    with torch.get_device_module().device(hidden.device):
        _silu_mul_fp8_kernel[(rows,)](q, s, hidden, HIDDEN=half, BLOCK=block, num_warps=16 if block >= 8192 else 8)
    return q, s


__all__ = [
    "can_use_rmsnorm_indexed_scale_shift_fp8",
    "can_use_silu_mul_fp8",
    "rmsnorm_indexed_scale_shift_fp8",
    "silu_mul_fp8",
]
