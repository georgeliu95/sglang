"""GB10 CuTe FP8 GEMM with one A scale per (512 rows, 128 columns of K).

The kernel keeps a single FP32 accumulator in units of the current K block's scale and multiplies it by
scale[h - 1] / scale[h] when entering block h (sglang/kernels/ops/gemm/gb10_cute_fp8_kernel.py). That
rescaling is the derived property guarded here: the result must equal sum_h scale[h] * (A_h @ B_h) up to the
final BF16 rounding, including scale ratios spanning orders of magnitude, a partial last 512-row block and a
partial last 256-row tile. The row-scaled path shares the kernel, so it must stay bit-identical to the AOT
fp8_scaled_mm it replaces.
"""

from __future__ import annotations

import pytest
import torch


def _gb10() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 1)


pytestmark = pytest.mark.skipif(not _gb10(), reason="GB10 (SM121) CuTe kernel")

K, N, ROWS = 56 * 128, 5376, 512  # attention out_proj of MiniMax-H3
FP8 = torch.float8_e4m3fn


def _weight(gen):
    w = torch.randn((N, K), device="cuda", generator=gen) * 0.02
    scale = w.abs().amax(1, keepdim=True) / 448.0
    return (w / scale).to(FP8).t(), scale.reshape(N).contiguous()  # [K, N] column-major, per column


@pytest.mark.parametrize("m", [4728, 1300])
def test_head_scaled_matches_blockwise_reference(m: int) -> None:
    from sglang.kernels.ops.gemm.gb10_cute_fp8 import fp8_cute_mm_head_scaled

    gen = torch.Generator(device="cuda").manual_seed(m)
    b, sb = _weight(gen)
    a = (torch.randn((m, K), device="cuda", generator=gen) * 100).clamp(-448, 448).to(FP8)
    blocks = -(-m // ROWS)
    # log-uniform over six orders of magnitude, so neighbouring heads differ by up to 1e6
    head_scale = torch.exp(torch.empty((K // 128, blocks), device="cuda").uniform_(-7, 7, generator=gen))
    out = fp8_cute_mm_head_scaled(a, b, head_scale.contiguous(), sb)
    row_scale = head_scale.t().repeat_interleave(ROWS, dim=0)[:m]  # [m, heads]
    a_deq = (a.float().view(m, K // 128, 128) * row_scale[..., None]).view(m, K)
    want = (a_deq @ b.float()) * sb
    err = ((out.float() - want).square().sum() / want.square().sum()).sqrt().item()
    assert err < 3e-3  # BF16 output rounding is ~1.7e-3 RMS


def test_row_scaled_path_still_bit_identical_to_aot() -> None:
    sgl_kernel = pytest.importorskip("sgl_kernel")
    from sglang.kernels.ops.gemm.gb10_cute_fp8 import fp8_cute_mm

    gen = torch.Generator(device="cuda").manual_seed(7)
    b, sb = _weight(gen)
    m = 4728
    a = (torch.randn((m, K), device="cuda", generator=gen) * 100).clamp(-448, 448).to(FP8)
    sa = torch.rand((m, 1), device="cuda", generator=gen) + 0.5
    want = sgl_kernel.fp8_scaled_mm(a, b, sa, sb, torch.bfloat16, None)
    assert torch.equal(fp8_cute_mm(a, b, sa, sb), want)


def test_head_scale_shape_is_checked() -> None:
    from sglang.kernels.ops.gemm.gb10_cute_fp8 import fp8_cute_mm_head_scaled

    a = torch.zeros((1024, K), device="cuda").to(FP8)
    b, sb = _weight(torch.Generator(device="cuda").manual_seed(1))
    with pytest.raises(ValueError):
        fp8_cute_mm_head_scaled(a, b, torch.ones((K // 128, 3), device="cuda"), sb)  # 1024 rows = 2 blocks
