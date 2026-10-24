"""Row/column-scaled FP8 GEMM on GB10 (SM121), scales fused into a CuTe DSL kernel.

Bit-identical to the AOT fp8_scaled_mm: FP32 accumulation over K, then
acc * scale_b * scale_a and one BF16 rounding. A persistent 256x192x64-tile kernel
is compiled per (M, N, K, device, variant) on first use (tens of seconds).
"""

from __future__ import annotations

from typing import Final

import msgspec
import torch

from sglang.srt.environ import envs


class _Variant(msgspec.Struct, frozen=True):
    """Output staging buffers, SMEM-staged scales, and the 8-tile band direction."""

    output_buffers: int
    stage_scales: bool
    along_m: bool


# (N, K) of the MiniMax-H3 DiT projections; picked on GB10 from 16 measured
# shapes (768p on 1/2/4 GPUs and 512p).
_VARIANTS: Final = {
    (21504, 5376): _Variant(3, True, False),  # qkv
    (5376, 7168): _Variant(3, True, False),  # attention out
    (28672, 5376): _Variant(3, True, True),  # fc1 (gate+up)
    (5376, 14336): _Variant(3, True, False),  # fc2
}
# Below this M the persistent grid is mostly idle; such calls (e.g. the M=537
# text refiner) keep the AOT path.
_MIN_M: Final = 1024
_COMPILED: Final[dict[tuple[int, int, int, int, _Variant], object]] = {}


def _compile(a, b, c, raw, sa, sb, variant: _Variant, sm_count: int, stream) -> object:
    import cutlass
    import cutlass.cute as cute

    from sglang.kernels.ops.gemm.gb10_cute_fp8_kernel import Sm120RowwiseGemmKernel

    kernel = Sm120RowwiseGemmKernel(
        cutlass.Float32, (256, 192, 64), variant.output_buffers,
        False, False, False, True, 8, 8, True, True, 1, 4, True, True,
        mma_register_requirement=240, load_register_requirement=24,
        dedup_scale_loads=True, epilogue_tile_n=32, defer_epilogue_drain=False,
        output_warp=True, physical_a_copy=False, b_cache_evict_last=False,
        reuse_column_scales=False, specialize_output_state=True,
        swizzle_along_m=variant.along_m, prefetch_scales=False,
        stage_scales=variant.stage_scales,
    )  # fmt: skip
    return cute.compile(kernel, a, b, c, raw, sa, sb, sm_count, stream)


def fp8_cute_mm(
    a: torch.Tensor, b: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor
) -> torch.Tensor:
    """BF16(scale_a * (FP32(a @ b) * scale_b)); a [M, K] row-major, b [K, N] col."""
    import cuda.bindings.driver as cuda
    from cutlass.cute.runtime import from_dlpack

    m, k = a.shape
    n = b.shape[1]
    variant = _VARIANTS[(n, k)]
    c = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    cute_a = from_dlpack(a.unsqueeze(-1), assumed_align=16)
    cute_b = from_dlpack(b.T.unsqueeze(-1), assumed_align=16)
    cute_c = from_dlpack(c.unsqueeze(-1), assumed_align=16)
    cute_sa = from_dlpack(sa.reshape(m), assumed_align=4)
    cute_sb = from_dlpack(sb.reshape(n), assumed_align=4)
    # The kernel ABI carries an unused raw-accumulator operand (write_raw=False).
    raw = from_dlpack(torch.empty((1, 1), device=a.device), assumed_align=4)
    stream = cuda.CUstream(torch.cuda.current_stream(a.device).cuda_stream)
    key = (m, n, k, a.device.index or 0, variant)
    compiled = _COMPILED.get(key)
    if compiled is None:
        sm_count = torch.cuda.get_device_properties(a.device).multi_processor_count
        compiled = _COMPILED[key] = _compile(
            cute_a, cute_b, cute_c, raw, cute_sa, cute_sb, variant, sm_count, stream
        )
    compiled(cute_a, cute_b, cute_c, raw, cute_sa, cute_sb, stream)
    return c


def eligible(
    a: torch.Tensor,
    b: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> bool:
    """Opt-in; H3 projection shapes with per-row/per-column FP32 scales, no bias."""
    if not envs.SGLANG_ENABLE_GB10_CUTE_FP8_GEMM.get():
        return False
    if a.ndim != 2 or b.ndim != 2 or out_dtype != torch.bfloat16 or bias is not None:
        return False
    m, k = a.shape
    n = b.shape[1]
    return (
        (n, k) in _VARIANTS
        and m >= _MIN_M
        and b.shape[0] == k
        and a.dtype == torch.float8_e4m3fn
        and b.dtype == a.dtype
        and a.device.type == "cuda"
        and b.device == a.device
        and a.is_contiguous()
        and b.stride() == (1, k)
        and sa.dtype == torch.float32
        and sb.dtype == torch.float32
        and sa.device == a.device
        and sb.device == a.device
        and sa.is_contiguous()
        and sb.is_contiguous()
        and sa.numel() == m
        and sb.numel() == n
    )
