"""Producer-fused per-token E4M3 quantization is byte-identical to the unfused chain (sm_12x).

The unfused chain is what the H3 DiT runs today: the BF16 producer (``rmsnorm_indexed_scale_shift`` or
``silu_and_mul_with_activation_rounding``), then ``sglang_per_token_quant_fp8``. The fused kernels
(``sglang/kernels/ops/quantization/token_fp8_fused.py``) must produce the same e4m3 bytes and the same fp32 scales,
including rows that are all zero, rows with a single large outlier, and H3's shapes (5376-wide norm rows, 14336-wide
SwiGLU rows); then the FP8 linear given the pre-quantized tuple must return the same output as given the BF16 rows.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 12,
    reason="needs an sm_12x GPU (the JIT reference quantizer uses fast math on SM90)",
)


def _reference_quant(rows: torch.Tensor):
    from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_quant_fp8

    return sglang_per_token_quant_fp8(rows.contiguous())


def _rows(t: int, d: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(t, d, device="cuda", generator=g) * torch.logspace(-2, 1.5, t, device="cuda")[:, None]
    x[0].zero_()
    x[1, 7] = 3000.0
    return x.bfloat16()


def _bitwise(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(a.view(torch.uint8) if a.dtype == torch.float8_e4m3fn else a,
                                              b.view(torch.uint8) if b.dtype == torch.float8_e4m3fn else b)


def _assert_same_quant(q, s, ref_q, ref_s):
    """Byte-identical to the reference on every nonzero row; all-zero rows give scale 0 and zero bytes.

    Below 2 * 8 * SMs rows the reference runs per_token_quant_fp8_cta_kernel, which has no scale == 0 guard
    (1 / 0 = inf, 0 * inf = NaN, clamped to 448), so its bytes for an all-zero row are not meaningful; they
    dequantize to 0 either way. The warp kernel (H3's 4728-row shards) writes zeros, as the fused kernels do.
    """
    zero = (ref_s == 0).flatten()
    assert torch.equal(s, ref_s)
    assert _bitwise(q[~zero], ref_q[~zero])
    assert not q[zero].view(torch.uint8).any()


@pytest.mark.parametrize("rows,dim", [(4728, 5376), (333, 5376), (64, 1536)])
@pytest.mark.parametrize("keep_bf16", [True, False])
def test_rmsnorm_indexed_scale_shift_fp8_matches_unfused(rows, dim, keep_bf16):
    from sglang.kernels.ops.diffusion.norm.rmsnorm_scale_shift_triton import rmsnorm_indexed_scale_shift
    from sglang.kernels.ops.quantization.token_fp8_fused import rmsnorm_indexed_scale_shift_fp8

    x = _rows(rows, dim, 1)
    w = (torch.rand(dim, device="cuda") + 0.5).bfloat16()
    mods = 6
    shift = (torch.randn(mods, dim, device="cuda") * 0.3).bfloat16()
    scale = (torch.randn(mods, dim, device="cuda") * 0.3).bfloat16()
    idx = torch.randint(0, mods, (rows,), device="cuda", dtype=torch.int64)
    eps = 1e-6
    ref = rmsnorm_indexed_scale_shift(x, w, shift, scale, idx, eps)
    ref_q, ref_s = _reference_quant(ref)
    y, q, s = rmsnorm_indexed_scale_shift_fp8(x, w, shift, scale, idx, eps, keep_bf16=keep_bf16)
    if keep_bf16:
        assert torch.equal(y, ref)
    else:
        assert y is None
    _assert_same_quant(q, s, ref_q, ref_s)


@pytest.mark.parametrize("rows,hidden", [(4728, 14336), (333, 14336), (64, 2048)])
def test_silu_mul_fp8_matches_unfused(rows, hidden):
    from sglang.kernels.ops.activation.activation import silu_and_mul_with_activation_rounding
    from sglang.kernels.ops.quantization.token_fp8_fused import silu_mul_fp8

    h = _rows(rows, 2 * hidden, 2)
    ref = silu_and_mul_with_activation_rounding(h)
    ref_q, ref_s = _reference_quant(ref)
    q, s = silu_mul_fp8(h)
    _assert_same_quant(q, s, ref_q, ref_s)


@pytest.mark.parametrize("rows,k,n", [(4728, 5376, 1024), (300, 14336, 512)])
def test_prequantized_rows_give_the_same_linear_output(rows, k, n):
    """apply_fp8_linear on (e4m3 rows, row scales) == on the BF16 rows it would quantize itself (same GEMM path)."""
    from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_quant_fp8
    from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear, cutlass_fp8_supported

    x = _rows(rows, k, 3)
    w = torch.randn(n, k, device="cuda")
    w_scale = w.abs().amax(dim=1, keepdim=True) / 448.0
    weight = (w / w_scale).to(torch.float8_e4m3fn).t()
    cutlass = cutlass_fp8_supported()
    ref = apply_fp8_linear(x, weight, w_scale, cutlass_fp8_supported=cutlass, use_per_token_if_dynamic=False)
    q, s = sglang_per_token_quant_fp8(x)
    out = apply_fp8_linear(q, weight, w_scale, input_scale=s, cutlass_fp8_supported=cutlass,
                           use_per_token_if_dynamic=False, pre_quant_output_dtype=torch.bfloat16)
    assert out.dtype == ref.dtype and torch.equal(out, ref)
