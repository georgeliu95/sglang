"""``group_norm_silu_ncthw`` (JIT CUDA) vs the eager chain and ``triton_group_norm_silu``.

Rows are the MiniMax-H3 encoder shapes (per-frame statistics) plus one
whole-clip HunyuanVAE decoder shape. The Triton column only applies where its
whole-group reduction matches the requested statistics (whole clip, or a
single frame); the other rows skip it.
"""

import torch
import torch.nn.functional as F

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.diffusion import group_norm_silu_ncthw, triton_group_norm_silu
from sglang.kernels.ops.diffusion.common.platform import is_cuda_sm_at_least
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=15, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

EPS = 1e-6

# (shape, num_groups, time_isolated)
CASES = [
    ((1, 128, 1, 256, 256), 32, True),
    ((7, 128, 1, 256, 256), 32, True),
    ((7, 256, 1, 128, 128), 32, True),
    ((7, 512, 1, 16, 16), 32, True),
    ((7, 1024, 1, 16, 16), 32, True),
    ((1, 256, 9, 64, 64), 32, True),
    # HunyuanVAE decoder shape; whole-clip statistics.
    ((1, 128, 20, 256, 256), 32, False),
]
CI_CASES = [
    ((7, 128, 1, 256, 256), 32, True),
    ((1, 256, 9, 64, 64), 32, True),
]


def eager_group_norm_silu(x, weight, bias, num_groups, time_isolated):
    if time_isolated and x.shape[2] > 1:
        b, c, t, h, w = x.shape
        merged = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        y = F.silu(F.group_norm(merged, num_groups, weight, bias, EPS))
        return y.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
    return F.silu(F.group_norm(x, num_groups, weight, bias, EPS))


@marker.parametrize("shape,num_groups,time_isolated", CASES, CI_CASES)
@marker.benchmark("impl", ["eager", "triton", "jit"])
def benchmark(shape, num_groups, time_isolated, impl):
    if not is_cuda_sm_at_least((10, 0)):
        marker.skip("group_norm_silu_ncthw is gated to SM100+")
    if impl == "triton" and time_isolated and shape[2] > 1:
        marker.skip("triton_group_norm_silu reduces over the whole clip")

    generator = torch.Generator(device="cuda").manual_seed(shape[1] * 31 + shape[-1])
    x = torch.randn(shape, device="cuda", generator=generator)
    weight = torch.randn(shape[1], device="cuda", generator=generator)
    bias = torch.randn(shape[1], device="cuda", generator=generator)

    if impl == "eager":

        def fn(x, weight, bias):
            return eager_group_norm_silu(x, weight, bias, num_groups, time_isolated)

    elif impl == "triton":

        def fn(x, weight, bias):
            return triton_group_norm_silu(
                x, weight, bias, num_groups=num_groups, eps=EPS
            )

    else:

        def fn(x, weight, bias):
            return group_norm_silu_ncthw(
                x,
                weight,
                bias,
                num_groups=num_groups,
                eps=EPS,
                time_isolated=time_isolated,
            )

    return marker.do_bench(fn, input_args=(x, weight, bias), graph_clone_args=(0,))


if __name__ == "__main__":
    benchmark.run()
