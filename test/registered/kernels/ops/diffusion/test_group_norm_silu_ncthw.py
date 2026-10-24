"""``group_norm_silu_ncthw`` (JIT CUDA) against ``F.silu(F.group_norm(...))``.

Close contract: exact two-pass fp32 moments and aten's affine factorization,
so the tolerance budget is fp32 reduction-order noise, not an fp16 budget.
Cases cover the register-resident launch (rows <= 8192 elements), the
partial/merge/apply split (rows > 8192), both addressing paths (contiguous
vectorized vs strided scalar), per-frame vs whole-clip statistics, reduction
tails, and 4-D input.
"""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.diffusion import (
    can_use_group_norm_silu_ncthw,
    group_norm_silu_ncthw,
)
from sglang.kernels.ops.diffusion.common.platform import is_cuda_sm_at_least
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not is_cuda_sm_at_least((10, 0)),
    reason="group_norm_silu_ncthw is gated to SM100+",
)

EPS = 1e-6


@pytest.fixture(autouse=True)
def inference():
    torch.manual_seed(20260919)
    with torch.inference_mode():
        yield


def reference(x, num_groups, weight, bias, time_isolated):
    if time_isolated and x.ndim == 5:
        b, c, t, h, w = x.shape
        merged = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        y = F.group_norm(merged, num_groups, weight, bias, EPS)
        y = y.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
    else:
        y = F.group_norm(x, num_groups, weight, bias, EPS)
    return F.silu(y)


def record_error(record_property, actual, expected):
    difference = (actual.float() - expected.float()).abs()
    finite = difference[torch.isfinite(difference)]
    if finite.numel():
        record_property("max_abs", finite.max().item())
        record_property("rms_abs", finite.square().mean().sqrt().item())


def strided_input(shape):
    # Channel/time permutation plus a spatial slice with a nonzero offset: the
    # scalar path must decode every stride, and no float4 load may be issued.
    b, c, t, h, w = shape
    x = torch.randn(b, t, c, h + 2, 2 * w + 1, device="cuda")
    x = x.permute(0, 2, 1, 3, 4)[..., 1 : h + 1, 1::2]
    assert tuple(x.shape) == shape and not x.is_contiguous()
    return x


# Encoder widths/spatial resolutions from ch=128, ch_mult=[1,2,2,4,4,8].
# B=7 is the measured seven-tiles-per-rank keyframe batch; temporal cases
# cover video use. Synthetic odd dimensions exercise the reduction tails.
GN_CASES = [
    pytest.param((1, 128, 1, 256, 256), 32, True, False, id="encoder-128x256"),
    pytest.param((7, 128, 1, 256, 256), 32, True, False, id="encoder-seven-tiles"),
    pytest.param((1, 256, 9, 64, 64), 32, True, False, id="encoder-temporal"),
    pytest.param((7, 512, 1, 16, 16), 32, True, False, id="encoder-count-4096"),
    pytest.param((7, 1024, 1, 16, 16), 32, True, False, id="encoder-count-8192"),
    pytest.param((1, 256, 9, 64, 64), 32, False, False, id="full-gn-video"),
    pytest.param((2, 12, 3, 17, 61), 3, True, False, id="large-partial-tail"),
    pytest.param((2, 12, 3, 17, 61), 3, False, False, id="full-gn-partial-tail"),
    pytest.param((2, 32, 3, 7, 12), 8, True, False, id="small-odd-group-vec"),
    pytest.param((2, 32, 3, 7, 11), 8, True, False, id="small-odd-group"),
    pytest.param((2, 32, 3, 7, 11), 8, False, False, id="full-small-group"),
    pytest.param((2, 32, 3, 9, 17), 8, True, True, id="strided-isolated"),
    pytest.param((2, 32, 3, 9, 17), 8, False, True, id="strided-full"),
    pytest.param((2, 12, 3, 17, 61), 3, True, True, id="strided-large-tail"),
    pytest.param((2, 12, 3, 17, 60), 3, True, True, id="strided-large-tail-w60"),
]


@pytest.mark.parametrize("shape,num_groups,time_isolated,strided", GN_CASES)
def test_group_norm_silu_ncthw(
    shape, num_groups, time_isolated, strided, record_property
):
    x = strided_input(shape) if strided else torch.randn(shape, device="cuda")
    weight = torch.randn(shape[1], device="cuda")
    bias = torch.randn_like(weight)
    assert can_use_group_norm_silu_ncthw(
        x, weight, bias, num_groups=num_groups, eps=EPS
    )
    expected = reference(x, num_groups, weight, bias, time_isolated)
    actual = group_norm_silu_ncthw(
        x, weight, bias, num_groups=num_groups, eps=EPS, time_isolated=time_isolated
    )
    assert actual.shape == x.shape and actual.dtype == torch.float32
    assert actual.is_contiguous() and actual.data_ptr() != x.data_ptr()
    record_error(record_property, actual, expected)
    # fp32 reductions use a different tree from aten's Welford; no fp16 budget.
    torch.testing.assert_close(actual, expected, atol=4e-5, rtol=3e-5)


def test_four_dimensional_input_is_one_frame(record_property):
    x = torch.randn(2, 64, 7, 9, device="cuda")
    weight = torch.randn(64, device="cuda")
    bias = torch.randn_like(weight)
    expected = F.silu(F.group_norm(x, 32, weight, bias, EPS))
    actual = group_norm_silu_ncthw(x, weight, bias, num_groups=32, eps=EPS)
    assert actual.shape == x.shape
    record_error(record_property, actual, expected)
    torch.testing.assert_close(actual, expected, atol=4e-5, rtol=3e-5)


# Widths 1024/2048/4096 cover the float4 path at 4096/8192 elements and the
# split at 16384; the odd widths retain strided-path coverage of the same.
@pytest.mark.parametrize("width", [31, 1031, 1024, 2048, 4096])
@pytest.mark.parametrize("kind", ["constant", "low_variance", "nan"])
@pytest.mark.parametrize("time_isolated", [False, True])
def test_group_norm_moments(width, kind, time_isolated, record_property):
    x = torch.randn(2, 8, 3, 1, width, device="cuda")
    if kind == "constant":
        x.fill_(3.25)
    elif kind == "low_variance":
        x.mul_(1e-3).add_(4.0)
    else:
        x[0, 0, 1, 0, 0] = float("nan")
    weight = torch.linspace(0.3, 1.3, 8, device="cuda")
    bias = torch.linspace(-0.7, 0.5, 8, device="cuda")
    expected = reference(x, 2, weight, bias, time_isolated)
    actual = group_norm_silu_ncthw(
        x, weight, bias, num_groups=2, eps=EPS, time_isolated=time_isolated
    )
    record_error(record_property, actual, expected)
    # At mean=4 and sigma=1e-3 one fp32 mean ulp is amplified by rsqrt(var+eps);
    # that ill-conditioned case keeps its own bound.
    atol = 3e-3 if kind == "low_variance" else 5e-4
    torch.testing.assert_close(actual, expected, atol=atol, rtol=1e-3, equal_nan=True)
    if kind == "constant":
        analytic = F.silu(bias)[None, :, None, None, None].expand_as(actual)
        torch.testing.assert_close(actual, analytic, atol=1e-6, rtol=1e-6)


def test_time_isolated_does_not_mix_frames():
    x = torch.randn(2, 32, 3, 9, 13, device="cuda")
    weight, bias = torch.ones(32, device="cuda"), torch.zeros(32, device="cuda")
    first = group_norm_silu_ncthw(
        x, weight, bias, num_groups=8, eps=EPS, time_isolated=True
    )
    changed = x.clone()
    changed[:, :, -1].mul_(19).add_(100)
    second = group_norm_silu_ncthw(
        changed, weight, bias, num_groups=8, eps=EPS, time_isolated=True
    )
    assert torch.equal(first[:, :, :-1], second[:, :, :-1])


def test_vectorized_and_strided_paths_agree_closely(record_property):
    # The same values through both addressing paths: a 16-byte aligned
    # contiguous tensor, and a copy of it embedded in a wider buffer so the
    # slice is not contiguous and takes the strided path.
    x = torch.randn(2, 64, 2, 16, 32, device="cuda")
    buffer = torch.empty(2, 64, 2, 16, 33, device="cuda")
    buffer[..., :32] = x
    view = buffer[..., :32]
    assert not view.is_contiguous()
    weight = torch.randn(64, device="cuda")
    bias = torch.randn_like(weight)
    vectorized = group_norm_silu_ncthw(x, weight, bias, num_groups=16, eps=EPS)
    strided = group_norm_silu_ncthw(view, weight, bias, num_groups=16, eps=EPS)
    record_error(record_property, vectorized, strided)
    torch.testing.assert_close(vectorized, strided, atol=2e-5, rtol=1e-5)


def test_rejects_unsupported_inputs():
    x = torch.randn(1, 32, 3, 7, 11, device="cuda")
    weight, bias = torch.ones(32, device="cuda"), torch.zeros(32, device="cuda")
    assert can_use_group_norm_silu_ncthw(x, weight, bias, num_groups=8, eps=EPS)
    assert not can_use_group_norm_silu_ncthw(
        x.half(), weight, bias, num_groups=8, eps=EPS
    )
    assert not can_use_group_norm_silu_ncthw(x, weight, bias, num_groups=7, eps=EPS)
    assert not can_use_group_norm_silu_ncthw(
        x, weight.half(), bias, num_groups=8, eps=EPS
    )
    assert not can_use_group_norm_silu_ncthw(
        x[:, :, :0], weight, bias, num_groups=8, eps=EPS
    )
    # The launcher, not the wrapper, is the contract check: it raises.
    with pytest.raises(RuntimeError, match="num_groups"):
        group_norm_silu_ncthw(x, weight, bias, num_groups=7, eps=EPS)
    with pytest.raises(RuntimeError):
        group_norm_silu_ncthw(x, weight[:16], bias, num_groups=8, eps=EPS)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
