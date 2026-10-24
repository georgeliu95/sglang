"""MiniMax-H3 VAE output ops: tile assembly, temporal window write, RGB de-normalization.

Three bit-exact (``torch.equal``) replacements for the eager chain in
``minimax_h3_video_vae/klvae.py`` and ``processor.py``:

- ``minimax_h3_vae_assemble_tiles(tiles, grid=, y_overlap=, x_overlap=)``:
  ``tiles`` is ``[N, B, C, T, H, W]`` fp32 in row-major (row, col) tile order,
  dim-0 stride arbitrary (an ``as_strided`` view over one buffer or a
  ``torch.stack``). Each tile is blended over the vertical overlap with the
  raw tile above it, then over the horizontal overlap with the raw tile to
  its left, cropped, and placed: the contiguous ``[B, C, T, Hout, Wout]``
  frame ``AutoencoderKL._assemble_tiles`` builds. Overlaps are one ``int``
  per axis or one value per gap.
- ``minimax_h3_vae_temporal_blend_write(part, overlap, blend_extent=, out=, start=, count=)``:
  ``out[:, :, start:start+count]`` receives ``part`` with its first
  ``blend_extent`` frames blended against the trailing frames of ``overlap``
  (``None`` disables the blend); the rest of ``out`` is untouched. ``out`` must
  not alias what is read: the predicate rejects shared storage, the launcher
  rejects any overlap between the byte range of the written frames and the
  byte ranges of the frames it reads (a bounds check, so interleaved views of
  one buffer are rejected too).
- ``minimax_h3_vae_denorm_clamp(value, mean=, std=)``:
  ``clamp((value - mean[c]) / std[c], 0, 1)`` on ``[B, 3, T, H, W]``, NaN preserved.

The blend is klvae's linear ramp ``a * (1 - w) + b * w`` with the weight
``w = k * (1 / n)`` rounded the way aten evaluates ``positions / extent`` for
a Python-int divisor (reciprocal first), and every op is one fp32 rounding,
so the kernels match eager bit for bit on both the strided and the 128-bit
vectorized path. The interface carries no H3
constants, but only the MiniMax-H3 VAE is wired to it today. SM100+.

Verified shapes (SM100): 2x2 tiles of ``[1, 3, 17, 256, 256]`` with 64-pixel
overlaps, temporal windows of ``[1, 3, 17, 448, 448]`` with a 4-frame blend,
plus the 1-tile, 3x3, odd-overlap and strided cases in
``test/registered/kernels/ops/diffusion/test_minimax_h3_vae_output.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.kernels.ops.diffusion.common.platform import is_cuda_sm_at_least
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_MIN_SM = (10, 0)
# Must match kMaxTiles / kVecWidth / kVecAlignment in csrc/diffusion/minimax_h3_vae_output.cuh.
_MAX_TILES = 64
_VEC_WIDTH = 4
_VEC_ALIGNMENT = 16


@cache_once
def _jit_minimax_h3_vae_output_module(vectorized: bool) -> Module:
    args = make_cpp_args(vectorized)
    return load_jit(
        "diffusion_minimax_h3_vae_output",
        *args,
        cuda_files=["diffusion/minimax_h3_vae_output.cuh"],
        cuda_wrappers=[
            (
                "assemble_tiles",
                f"minimax_h3_vae_output::AssembleTilesKernel<{args}>::run",
            ),
            (
                "temporal_blend_write",
                f"minimax_h3_vae_output::TemporalBlendWriteKernel<{args}>::run",
            ),
            (
                "denorm_clamp",
                f"minimax_h3_vae_output::DenormClampKernel<{args}>::run",
            ),
        ],
    )


def _vectorizable(*tensors: torch.Tensor) -> bool:
    # Mirror of the launcher's 128-bit precondition: innermost stride 1,
    # `W % 4 == 0`, every other stride and the data pointer 16-byte aligned.
    return all(
        t.stride(-1) == 1
        and t.shape[-1] % _VEC_WIDTH == 0
        and all(s % _VEC_WIDTH == 0 for s in t.stride()[:-1])
        and t.data_ptr() % _VEC_ALIGNMENT == 0
        for t in tensors
    )


@register_custom_op(
    op_name="diffusion_minimax_h3_vae_assemble_tiles", mutates_args=["out"]
)
def _assemble_tiles_out(
    tiles: torch.Tensor,
    tile_table: torch.Tensor,
    out: torch.Tensor,
    grid_cols: int,
    vectorized: bool,
) -> None:
    module = _jit_minimax_h3_vae_output_module(vectorized)
    module.assemble_tiles(tiles, tile_table, out, grid_cols)


@register_custom_op(
    op_name="diffusion_minimax_h3_vae_temporal_blend_write", mutates_args=["out"]
)
def _temporal_blend_write_out(
    part: torch.Tensor,
    overlap: torch.Tensor,
    out: torch.Tensor,
    start: int,
    count: int,
    blend_extent: int,
    vectorized: bool,
) -> None:
    module = _jit_minimax_h3_vae_output_module(vectorized)
    module.temporal_blend_write(part, overlap, out, start, count, blend_extent)


@register_custom_op(
    op_name="diffusion_minimax_h3_vae_denorm_clamp", mutates_args=["out"]
)
def _denorm_clamp_out(
    value: torch.Tensor,
    out: torch.Tensor,
    mean0: float,
    mean1: float,
    mean2: float,
    std0: float,
    std1: float,
    std2: float,
    vectorized: bool,
) -> None:
    module = _jit_minimax_h3_vae_output_module(vectorized)
    module.denorm_clamp(value, out, mean0, mean1, mean2, std0, std1, std2)


def _per_gap(overlap: int | Sequence[int], gaps: int) -> list[int] | None:
    """One overlap per gap between tiles, or ``None`` when the count is off."""
    if isinstance(overlap, int):
        return [overlap] * gaps
    values = [int(v) for v in overlap]
    return values if len(values) == gaps else None


def _tile_table(
    grid: tuple[int, int],
    y_gaps: list[int],
    x_gaps: list[int],
    tile_h: int,
    tile_w: int,
) -> tuple[list[list[int]], int, int]:
    rows, cols = grid
    heights = [tile_h - (y_gaps[i] if i < rows - 1 else 0) for i in range(rows)]
    widths = [tile_w - (x_gaps[j] if j < cols - 1 else 0) for j in range(cols)]
    # Rows of (oy, ox, h, w, hy, wx); the blend extents follow klvae.blend's
    # min(a.shape, b.shape, extent).
    table = []
    oy = 0
    for i in range(rows):
        ox = 0
        for j in range(cols):
            table.append(
                [
                    oy,
                    ox,
                    heights[i],
                    widths[j],
                    min(y_gaps[i - 1], tile_h) if i else 0,
                    min(x_gaps[j - 1], tile_w) if j else 0,
                ]
            )
            ox += widths[j]
        oy += heights[i]
    return table, sum(heights), sum(widths)


def minimax_h3_vae_assemble_tiles(
    tiles: torch.Tensor,
    *,
    grid: tuple[int, int],
    y_overlap: int | Sequence[int],
    x_overlap: int | Sequence[int],
) -> torch.Tensor:
    """Blend, crop and place ``grid[0] x grid[1]`` tiles into one contiguous frame."""
    rows, cols = grid
    if tiles.ndim != 6 or tiles.shape[0] != rows * cols:
        raise ValueError(
            f"expected a [rows*cols, B, C, T, H, W] tile stack for grid {grid}, "
            f"got shape {tuple(tiles.shape)}"
        )
    y_gaps = _per_gap(y_overlap, rows - 1)
    x_gaps = _per_gap(x_overlap, cols - 1)
    if y_gaps is None or x_gaps is None:
        raise ValueError(f"overlaps must be one int per axis or one per gap of {grid}")
    tile_h, tile_w = tiles.shape[-2:]
    table, out_h, out_w = _tile_table(grid, y_gaps, x_gaps, tile_h, tile_w)
    out = torch.empty(
        (*tiles.shape[1:4], out_h, out_w), dtype=torch.float32, device=tiles.device
    )
    vectorized = _vectorizable(tiles, out) and all(
        rect[1] % _VEC_WIDTH == 0
        and rect[3] % _VEC_WIDTH == 0
        and rect[5] % _VEC_WIDTH == 0
        for rect in table
    )
    tile_table = torch.tensor(table, dtype=torch.int32)
    _assemble_tiles_out(tiles, tile_table, out, cols, vectorized)
    return out


def can_use_minimax_h3_vae_assemble_tiles(
    tiles: torch.Tensor,
    *,
    grid: tuple[int, int],
    y_overlap: int | Sequence[int],
    x_overlap: int | Sequence[int],
) -> bool:
    rows, cols = grid
    if not (
        isinstance(tiles, torch.Tensor)
        and tiles.is_cuda
        and tiles.dtype == torch.float32
        and tiles.ndim == 6
        and rows > 0
        and cols > 0
        and rows * cols <= _MAX_TILES
        and tiles.shape[0] == rows * cols
        and tiles.numel() > 0
    ):
        return False
    y_gaps = _per_gap(y_overlap, rows - 1)
    x_gaps = _per_gap(x_overlap, cols - 1)
    tile_h, tile_w = tiles.shape[-2:]
    return (
        y_gaps is not None
        and x_gaps is not None
        and all(0 <= gap < tile_h for gap in y_gaps)
        and all(0 <= gap < tile_w for gap in x_gaps)
        and is_cuda_sm_at_least(_MIN_SM, tiles.device)
        and not torch.compiler.is_compiling()
    )


def minimax_h3_vae_temporal_blend_write(
    part: torch.Tensor,
    overlap: torch.Tensor | None,
    *,
    blend_extent: int,
    out: torch.Tensor,
    start: int,
    count: int,
) -> None:
    """Write ``part[:, :, :count]`` into ``out[:, :, start:start+count]``, blending
    its leading frames against ``overlap`` when one is given."""
    if overlap is None:
        # The launcher never reads `overlap` at extent 0; pass `part` for the shape check.
        overlap, extent = part, 0
    else:
        extent = min(int(blend_extent), overlap.shape[2], part.shape[2])
    _temporal_blend_write_out(
        part,
        overlap,
        out,
        int(start),
        int(count),
        extent,
        _vectorizable(part, overlap, out),
    )


def can_use_minimax_h3_vae_temporal_blend_write(
    part: torch.Tensor,
    overlap: torch.Tensor | None,
    *,
    blend_extent: int,
    out: torch.Tensor,
    start: int,
    count: int,
) -> bool:
    if not isinstance(part, torch.Tensor):
        return False
    tensors = (part, out) if overlap is None else (part, overlap, out)
    if not all(
        isinstance(t, torch.Tensor)
        and t.is_cuda
        and t.dtype == torch.float32
        and t.ndim == 5
        and t.device == part.device
        and t.shape[:2] == part.shape[:2]
        and t.shape[3:] == part.shape[3:]
        for t in tensors
    ):
        return False
    if not (
        blend_extent >= 0
        and 0 <= count <= part.shape[2]
        and start >= 0
        and start + count <= out.shape[2]
    ):
        return False
    out_storage = out.untyped_storage().data_ptr()
    if any(t.untyped_storage().data_ptr() == out_storage for t in tensors[:-1]):
        return False
    return (
        is_cuda_sm_at_least(_MIN_SM, part.device) and not torch.compiler.is_compiling()
    )


def minimax_h3_vae_denorm_clamp(
    value: torch.Tensor, *, mean: Sequence[float], std: Sequence[float]
) -> torch.Tensor:
    """``clamp((value - mean[c]) / std[c], 0, 1)`` as one launch; NaN preserved."""
    mean = [float(v) for v in mean]
    std = [float(v) for v in std]
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("mean and std must hold one value per RGB channel")
    out = torch.empty(value.shape, dtype=torch.float32, device=value.device)
    _denorm_clamp_out(value, out, *mean, *std, _vectorizable(value, out))
    return out


def can_use_minimax_h3_vae_denorm_clamp(
    value: torch.Tensor, *, mean: Sequence[float], std: Sequence[float]
) -> bool:
    return (
        isinstance(value, torch.Tensor)
        and value.is_cuda
        and value.dtype == torch.float32
        and value.ndim == 5
        and value.shape[1] == 3
        and value.numel() > 0
        and len(mean) == 3
        and len(std) == 3
        and is_cuda_sm_at_least(_MIN_SM, value.device)
        and not torch.compiler.is_compiling()
    )


__all__ = [
    "can_use_minimax_h3_vae_assemble_tiles",
    "can_use_minimax_h3_vae_denorm_clamp",
    "can_use_minimax_h3_vae_temporal_blend_write",
    "minimax_h3_vae_assemble_tiles",
    "minimax_h3_vae_denorm_clamp",
    "minimax_h3_vae_temporal_blend_write",
]
