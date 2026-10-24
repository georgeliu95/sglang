"""Fused vs split producers for the MiniMax-H3 FP8 decoder.

``split`` is the eager chain the fused kernel replaces: the fp32 residual
update as two aten kernels, ``F.rms_norm`` / ``F.layer_norm``, and
``per_token_quant_fp8``; for the FFN it is ``silu_and_mul_with_activation_rounding``
(fp16 intermediate, as autocast produces) followed by ``per_token_quant_fp8``.
Rows 256 / 1792 / 7168 are one, seven and twenty-eight 256-token decoder tiles.
The 1792-row ``silu_mul_quant_fp8`` cell decides whether that kernel stays: a
fused gain under 10 us there means the two-launch chain is good enough.
"""

import torch
import torch.nn.functional as F

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.activation.activation import (
    silu_and_mul_with_activation_rounding,
)
from sglang.kernels.ops.diffusion import (
    minimax_h3_vae_residual_layernorm,
    minimax_h3_vae_residual_rmsnorm_fp8,
    minimax_h3_vae_rmsnorm_fp8,
    silu_mul_quant_fp8,
)
from sglang.kernels.ops.diffusion.common.platform import is_cuda_sm_at_least
from sglang.kernels.ops.quantization.per_token_quant_fp8 import per_token_quant_fp8
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=15, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

DEVICE = "cuda"
WIDTH = 2048
FFN_WIDTH = 8192
EPS = 1e-5
OPS = [
    "rmsnorm_fp8",
    "residual_rmsnorm_fp8",
    "residual_layernorm",
    "silu_mul_quant_fp8",
]


def quantize(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q = torch.empty_like(value, dtype=torch.float8_e4m3fn)
    scales = torch.empty((value.shape[0], 1), dtype=torch.float32, device=DEVICE)
    per_token_quant_fp8(value, q, scales)
    return q, scales


@marker.parametrize("rows", [256, 1792, 7168], [1792])
@marker.parametrize("op", OPS)
@marker.benchmark("impl", ["split", "fused"], unit="us")
def benchmark(rows: int, op: str, impl: str):
    if not is_cuda_sm_at_least((10, 0)):
        marker.skip("MiniMax-H3 VAE FP8 producers are gated to SM100+")

    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(20260920 + rows)
    fused = impl == "fused"

    if op == "silu_mul_quant_fp8":
        x = torch.randn(
            (rows, 2 * FFN_WIDTH),
            dtype=torch.float16,
            device=DEVICE,
            generator=generator,
        )

        def fn(x):
            if fused:
                return silu_mul_quant_fp8(x)
            return quantize(silu_and_mul_with_activation_rounding(x))

        return marker.do_bench(fn, input_args=(x,), graph_clone_args=(0,))

    x = torch.randn(
        (rows, WIDTH), dtype=torch.float32, device=DEVICE, generator=generator
    )
    weight = torch.rand((WIDTH,), device=DEVICE, generator=generator) + 0.5
    if op == "rmsnorm_fp8":

        def fn(x, weight):
            if fused:
                return minimax_h3_vae_rmsnorm_fp8(x, weight, eps=EPS)
            return quantize(F.rms_norm(x, (WIDTH,), weight, eps=EPS))

        return marker.do_bench(fn, input_args=(x, weight), graph_clone_args=(0,))

    projected = torch.randn(
        (rows, WIDTH), dtype=torch.float16, device=DEVICE, generator=generator
    )
    layer_scale = torch.rand((WIDTH,), device=DEVICE, generator=generator) * 0.3
    if op == "residual_rmsnorm_fp8":

        def fn(x, projected, layer_scale, weight):
            if fused:
                return minimax_h3_vae_residual_rmsnorm_fp8(
                    x, projected, layer_scale, weight, eps=EPS
                )
            residual = x + projected * layer_scale
            return (
                residual,
                *quantize(F.rms_norm(residual, (WIDTH,), weight, eps=EPS)),
            )

    else:
        bias = torch.randn((WIDTH,), device=DEVICE, generator=generator)

        def fn(x, projected, layer_scale, weight):
            if fused:
                return minimax_h3_vae_residual_layernorm(
                    x, projected, layer_scale, weight, bias, eps=EPS
                )
            return F.layer_norm(
                x + projected * layer_scale, (WIDTH,), weight, bias, EPS
            )

    return marker.do_bench(
        fn, input_args=(x, projected, layer_scale, weight), graph_clone_args=(0, 1)
    )


if __name__ == "__main__":
    benchmark.run()
