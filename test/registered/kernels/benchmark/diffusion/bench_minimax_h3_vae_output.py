"""MiniMax-H3 VAE output ops (JIT CUDA) vs the eager klvae chain.

One 448x448 frame stack per row: a 2x2 tile assembly with 64-pixel overlaps,
the temporal blend-write of one decoded window, and the RGB de-normalization.
The eager side reuses cached ramp weights, as the production VAE does.
"""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.diffusion import (
    minimax_h3_vae_assemble_tiles,
    minimax_h3_vae_denorm_clamp,
    minimax_h3_vae_temporal_blend_write,
)
from sglang.kernels.ops.diffusion.common.platform import is_cuda_sm_at_least
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=15, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

TILE = 256
OVERLAP = 64
FRAME = 2 * TILE - OVERLAP
FRAME_OVERLAP = 4
IMAGENET_MEAN = (-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225)
IMAGENET_STD = (1 / 0.229, 1 / 0.224, 1 / 0.225)


def ramp_weights(extent: int):
    positions = torch.arange(extent, device="cuda", dtype=torch.float32)
    return 1 - positions / extent, positions / extent


def eager_blend(a, b, extent, dim, weights):
    weight_a, weight_b = weights
    shape = [1] * a.ndim
    shape[dim] = extent
    blended = a.narrow(dim, a.shape[dim] - extent, extent) * weight_a.view(shape)
    blended.add_(b.narrow(dim, 0, extent) * weight_b.view(shape))
    return torch.cat((blended, b.narrow(dim, extent, b.shape[dim] - extent)), dim=dim)


def make_eager_assemble(weights):
    def fn(tiles):
        out = tiles.new_empty((*tiles.shape[1:4], FRAME, FRAME))
        for i in range(2):
            for j in range(2):
                tile = tiles[i * 2 + j]
                if i:
                    tile = eager_blend(tiles[j], tile, OVERLAP, -2, weights)
                if j:
                    tile = eager_blend(tiles[i * 2], tile, OVERLAP, -1, weights)
                h = TILE - OVERLAP if i == 0 else TILE
                w = TILE - OVERLAP if j == 0 else TILE
                y0 = 0 if i == 0 else TILE - OVERLAP
                x0 = 0 if j == 0 else TILE - OVERLAP
                out[..., y0 : y0 + h, x0 : x0 + w].copy_(tile[..., :h, :w])
        return out

    return fn


def jit_assemble(tiles):
    return minimax_h3_vae_assemble_tiles(
        tiles, grid=(2, 2), y_overlap=OVERLAP, x_overlap=OVERLAP
    )


def make_eager_temporal(weights):
    def fn(part, overlap, out):
        out[:, :, : part.shape[2]].copy_(
            eager_blend(overlap, part, FRAME_OVERLAP, 2, weights)
        )

    return fn


def jit_temporal(part, overlap, out):
    minimax_h3_vae_temporal_blend_write(
        part,
        overlap,
        blend_extent=FRAME_OVERLAP,
        out=out,
        start=0,
        count=part.shape[2],
    )


def make_eager_denorm():
    mean = torch.tensor(IMAGENET_MEAN, device="cuda").view(1, 3, 1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device="cuda").view(1, 3, 1, 1, 1)

    def fn(value):
        return (value - mean).div_(std).clamp_(0, 1)

    return fn


def jit_denorm(value):
    return minimax_h3_vae_denorm_clamp(value, mean=IMAGENET_MEAN, std=IMAGENET_STD)


@marker.parametrize("op", ["assemble_tiles", "temporal_blend_write", "denorm_clamp"])
@marker.parametrize("frames", [17, 33], [17])
@marker.benchmark("impl", ["eager", "jit"])
def benchmark(op: str, frames: int, impl: str):
    if not is_cuda_sm_at_least((10, 0)):
        marker.skip("MiniMax-H3 VAE output kernels are gated to SM100+")
    generator = torch.Generator(device="cuda").manual_seed(frames)
    if op == "assemble_tiles":
        tiles = torch.randn(
            4, 1, 3, frames, TILE, TILE, device="cuda", generator=generator
        )
        fn = (
            make_eager_assemble(ramp_weights(OVERLAP))
            if impl == "eager"
            else jit_assemble
        )
        return marker.do_bench(fn, input_args=(tiles,))
    if op == "temporal_blend_write":
        part = torch.randn(
            1, 3, frames, FRAME, FRAME, device="cuda", generator=generator
        )
        overlap = torch.randn(
            1, 3, FRAME_OVERLAP, FRAME, FRAME, device="cuda", generator=generator
        )
        out = torch.empty_like(part)
        fn = (
            make_eager_temporal(ramp_weights(FRAME_OVERLAP))
            if impl == "eager"
            else jit_temporal
        )
        # In place: `out` is the written tensor, so count it explicitly.
        return marker.do_bench(
            fn, input_args=(part, overlap, out), memory_output=(out,)
        )
    value = torch.randn(1, 3, frames, FRAME, FRAME, device="cuda", generator=generator)
    fn = make_eager_denorm() if impl == "eager" else jit_denorm
    return marker.do_bench(fn, input_args=(value,))


if __name__ == "__main__":
    benchmark.run()
