"""MiniMax-H3 VAE output ops (JIT CUDA) against the eager klvae chain, bitwise.

``blend`` and ``assemble_reference`` below re-implement ``AutoencoderKL.blend``
and ``_assemble_tiles`` with plain aten ops, ``revert_reference`` the
``VAEProcessor.revert_tensor`` Normalize + clamp. Every kernel result must be
``torch.equal`` to its reference (NaN-aware only where the input carries NaN);
a tolerance here would hide a real bug.
"""

import sys

import pytest
import torch

from sglang.kernels.ops.diffusion import (
    can_use_minimax_h3_vae_assemble_tiles,
    can_use_minimax_h3_vae_denorm_clamp,
    can_use_minimax_h3_vae_temporal_blend_write,
    minimax_h3_vae_assemble_tiles,
    minimax_h3_vae_denorm_clamp,
    minimax_h3_vae_temporal_blend_write,
)
from sglang.kernels.ops.diffusion.common.platform import is_cuda_sm_at_least
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not is_cuda_sm_at_least((10, 0)),
    reason="MiniMax-H3 VAE output kernels are gated to SM100+",
)

IMAGENET_MEAN = (-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225)
IMAGENET_STD = (1 / 0.229, 1 / 0.224, 1 / 0.225)


@pytest.fixture(autouse=True)
def inference():
    torch.manual_seed(20260919)
    with torch.inference_mode():
        yield


def blend(a, b, extent, dim):
    # AutoencoderKL.blend: fp32 ramp weights, a * (1 - w) then += b * w.
    extent = min(a.shape[dim], b.shape[dim], extent)
    positions = torch.arange(extent, device=b.device, dtype=b.dtype)
    shape = [1] * a.ndim
    shape[dim] = extent
    weight_a = (1 - positions / extent).view(shape)
    weight_b = (positions / extent).view(shape)
    blended = a.narrow(dim, a.shape[dim] - extent, extent) * weight_a
    blended.add_(b.narrow(dim, 0, extent) * weight_b)
    return torch.cat((blended, b.narrow(dim, extent, b.shape[dim] - extent)), dim=dim)


def assemble_reference(tiles, grid, y_gaps, x_gaps):
    # AutoencoderKL._assemble_tiles: vertical blend with the raw tile above,
    # horizontal blend with the raw tile to the left, crop, place.
    rows, cols = grid
    bands = []
    for i in range(rows):
        pieces = []
        for j in range(cols):
            tile = tiles[i * cols + j]
            if i:
                tile = blend(tiles[(i - 1) * cols + j], tile, y_gaps[i - 1], -2)
            if j:
                tile = blend(tiles[i * cols + j - 1], tile, x_gaps[j - 1], -1)
            if i < rows - 1:
                tile = tile[..., : tile.shape[-2] - y_gaps[i], :]
            if j < cols - 1:
                tile = tile[..., : tile.shape[-1] - x_gaps[j]]
            pieces.append(tile)
        bands.append(torch.cat(pieces, dim=-1))
    return torch.cat(bands, dim=-2)


def revert_reference(value, mean, std):
    mean = value.new_tensor(mean).view(1, 3, 1, 1, 1)
    std = value.new_tensor(std).view(1, 3, 1, 1, 1)
    return (value - mean).div_(std).clamp_(0, 1)


def assert_bitwise(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


def make_tiles(grid, tile_shape, layout):
    """[N, B, C, T, H, W] tiles with distinct per-tile offsets, in one of three layouts:
    a fresh contiguous stack, a stride-2 W slice (scalar path), or an
    as_strided view whose dim-0 stride is not the tile size (vectorized path)."""
    rows, cols = grid
    n = rows * cols
    b, c, t, h, w = tile_shape
    offsets = torch.arange(n, device="cuda", dtype=torch.float32).view(n, 1, 1, 1, 1, 1)
    if layout == "contiguous":
        return torch.randn(n, b, c, t, h, w, device="cuda") + 3 * offsets
    if layout == "strided_w":
        wide = torch.randn(n, b, c, t, h, 2 * w, device="cuda") + 3 * offsets
        tiles = wide[..., 1::2]
        assert not tiles.is_contiguous()
        return tiles
    # Tiles interleaved along a middle axis of one buffer: [B, C, T, N, H, W].
    buffer = torch.randn(b, c, t, n, h, w, device="cuda")
    buffer += 3 * offsets.view(1, 1, 1, n, 1, 1)
    tiles = buffer.permute(3, 0, 1, 2, 4, 5)
    assert tiles.stride(0) == h * w
    return tiles


ASSEMBLE_CASES = [
    pytest.param((1, 1), (1, 3, 2, 7, 11), 0, 0, id="one-tile"),
    pytest.param((2, 2), (2, 3, 5, 8, 12), 4, 4, id="2x2-aligned"),
    pytest.param((2, 2), (2, 3, 5, 8, 12), 3, 5, id="2x2-odd"),
    pytest.param((3, 3), (2, 3, 5, 8, 12), (3, 2), (4, 8), id="3x3-per-gap"),
    pytest.param((3, 3), (1, 3, 2, 12, 16), 4, 8, id="3x3-aligned"),
    pytest.param((1, 4), (1, 3, 2, 8, 16), 0, (4, 8, 12), id="1x4-ragged"),
    pytest.param((4, 1), (1, 1, 1, 16, 8), (1, 7, 15), 0, id="4x1-extent-near-h"),
]


@pytest.mark.parametrize("layout", ["contiguous", "strided_w", "interleaved"])
@pytest.mark.parametrize("grid,tile_shape,y_overlap,x_overlap", ASSEMBLE_CASES)
def test_assemble_tiles_matches_eager(grid, tile_shape, y_overlap, x_overlap, layout):
    tiles = make_tiles(grid, tile_shape, layout)
    rows, cols = grid
    y_gaps = [y_overlap] * (rows - 1) if isinstance(y_overlap, int) else list(y_overlap)
    x_gaps = [x_overlap] * (cols - 1) if isinstance(x_overlap, int) else list(x_overlap)
    assert can_use_minimax_h3_vae_assemble_tiles(
        tiles, grid=grid, y_overlap=y_overlap, x_overlap=x_overlap
    )
    expected = assemble_reference(tiles, grid, y_gaps, x_gaps)
    actual = minimax_h3_vae_assemble_tiles(
        tiles, grid=grid, y_overlap=y_overlap, x_overlap=x_overlap
    )
    assert actual.is_contiguous() and actual.data_ptr() != tiles.data_ptr()
    assert_bitwise(actual, expected)


def test_assemble_tiles_rejects_bad_geometry():
    tiles = torch.randn(4, 1, 3, 2, 8, 12, device="cuda")
    assert not can_use_minimax_h3_vae_assemble_tiles(
        tiles, grid=(3, 1), y_overlap=2, x_overlap=0
    )
    assert not can_use_minimax_h3_vae_assemble_tiles(
        tiles, grid=(2, 2), y_overlap=(2, 2), x_overlap=2
    )
    assert not can_use_minimax_h3_vae_assemble_tiles(
        tiles, grid=(2, 2), y_overlap=8, x_overlap=2
    )
    assert not can_use_minimax_h3_vae_assemble_tiles(
        tiles.half(), grid=(2, 2), y_overlap=2, x_overlap=2
    )
    with pytest.raises(ValueError, match="grid"):
        minimax_h3_vae_assemble_tiles(tiles, grid=(3, 1), y_overlap=2, x_overlap=0)
    with pytest.raises(ValueError, match="overlaps"):
        minimax_h3_vae_assemble_tiles(tiles, grid=(2, 2), y_overlap=(2, 2), x_overlap=2)


@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("extent,count", [(0, 4), (2, 4), (7, 4), (2, 0)])
def test_temporal_blend_write_matches_eager(strided, extent, count):
    width = 24 if strided else 12
    part = torch.randn(2, 3, 5, 8, width, device="cuda")
    overlap = torch.randn(2, 3, 3, 8, width, device="cuda")
    storage = torch.full((2, 3, 12, 8, width), -111.0, device="cuda")
    out = storage
    if strided:
        part, overlap, out = part[..., 1::2], overlap[..., 1::2], storage[..., 1::2]
    assert can_use_minimax_h3_vae_temporal_blend_write(
        part, overlap, blend_extent=extent, out=out, start=3, count=count
    )
    expected = out.clone()
    expected[:, :, 3 : 3 + count] = blend(overlap, part, extent, 2)[:, :, :count]
    minimax_h3_vae_temporal_blend_write(
        part, overlap, blend_extent=extent, out=out, start=3, count=count
    )
    assert_bitwise(out, expected)
    if strided:
        assert torch.all(storage[..., ::2] == -111).item()


def test_temporal_blend_write_without_overlap_and_alias_guard():
    part = torch.randn(1, 3, 5, 7, 11, device="cuda")
    out = torch.full((1, 3, 8, 7, 11), -9.0, device="cuda")
    assert can_use_minimax_h3_vae_temporal_blend_write(
        part, None, blend_extent=4, out=out, start=2, count=5
    )
    minimax_h3_vae_temporal_blend_write(
        part, None, blend_extent=4, out=out, start=2, count=5
    )
    assert torch.equal(out[:, :, 2:7], part)
    assert (
        torch.all(out[:, :, :2] == -9).item() and torch.all(out[:, :, 7:] == -9).item()
    )
    # Shared storage is rejected by the predicate, and by the launcher whenever
    # the byte range of the written frames meets the byte range of the frames
    # it reads (a bounds check, so interleaved views are rejected as well).
    assert not can_use_minimax_h3_vae_temporal_blend_write(
        out[:, :, :5], None, blend_extent=0, out=out, start=0, count=5
    )
    with pytest.raises(RuntimeError, match="alias"):
        minimax_h3_vae_temporal_blend_write(
            out[:, :, :5], None, blend_extent=0, out=out, start=0, count=5
        )
    with pytest.raises(RuntimeError, match="alias"):
        minimax_h3_vae_temporal_blend_write(
            part, out[:, :, :3], blend_extent=2, out=out, start=0, count=5
        )
    # The check is per frame range, not per view: with the frame axis outermost
    # in memory (B = C = 1), disjoint frame windows of one buffer are allowed.
    single = torch.full((1, 1, 8, 7, 11), -9.0, device="cuda")
    single[:, :, :3] = torch.randn(1, 1, 3, 7, 11, device="cuda")
    minimax_h3_vae_temporal_blend_write(
        single[:, :, :3], None, blend_extent=0, out=single, start=5, count=3
    )
    assert torch.equal(single[:, :, 5:8], single[:, :, :3])
    with pytest.raises(RuntimeError, match="count"):
        minimax_h3_vae_temporal_blend_write(
            part, None, blend_extent=0, out=out, start=0, count=6
        )


@pytest.mark.parametrize("strided", [False, True])
def test_denorm_clamp_matches_eager(strided):
    value = torch.randn(2, 3, 5, 8, 24 if strided else 12, device="cuda")
    if strided:
        value = value[..., 1::2]
    value[0, 0, 0, 0, :3] = torch.tensor(
        [float("nan"), float("inf"), -float("inf")], device="cuda"
    )
    assert can_use_minimax_h3_vae_denorm_clamp(
        value, mean=IMAGENET_MEAN, std=IMAGENET_STD
    )
    expected = revert_reference(value, IMAGENET_MEAN, IMAGENET_STD)
    actual = minimax_h3_vae_denorm_clamp(value, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    assert actual.is_contiguous()
    assert_bitwise(actual, expected)
    assert torch.isnan(actual[0, 0, 0, 0, 0]).item()
    assert actual[0, 0, 0, 0, 1].item() == 1.0 and actual[0, 0, 0, 0, 2].item() == 0.0


def test_denorm_clamp_rejects_unsupported_inputs():
    value = torch.randn(2, 4, 5, 8, 12, device="cuda")
    assert not can_use_minimax_h3_vae_denorm_clamp(
        value, mean=IMAGENET_MEAN, std=IMAGENET_STD
    )
    assert not can_use_minimax_h3_vae_denorm_clamp(
        value[:, :3].half(), mean=IMAGENET_MEAN, std=IMAGENET_STD
    )
    with pytest.raises(RuntimeError):
        minimax_h3_vae_denorm_clamp(value, mean=IMAGENET_MEAN, std=IMAGENET_STD)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
