import sys

import pytest
import torch

from sglang.kernels.ops.diffusion import (
    can_use_fused_inplace_qknorm_rope,
    fused_inplace_qknorm_rope,
    fused_qknorm_rope_out_of_place,
)
from sglang.kernels.ops.layernorm.norm import fused_inplace_qknorm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def test_out_of_place_qknorm_rope_matches_inplace_and_keeps_inputs() -> None:
    """The out-of-place variant (strided fused-qkv views in, contiguous copies
    out) is bit-equal to the in-place kernel and leaves its inputs untouched;
    VDN-H3's linear branch reads the raw q/k after it."""
    T, H, D, R = 512, 4, 128, 96
    if not can_use_fused_inplace_qknorm_rope(
        D, R, True, torch.bfloat16, torch.bfloat16, True
    ):
        pytest.skip("fused qknorm+rope JIT kernel unavailable")
    g = torch.Generator(device="cpu").manual_seed(0)
    qkv = torch.randn(T, 3 * H * D, generator=g).to("cuda", torch.bfloat16)
    q = qkv[:, : H * D].view(T, H, D)
    k = qkv[:, H * D : 2 * H * D].view(T, H, D)
    qw = (torch.rand(D, generator=g) + 0.5).to("cuda", torch.bfloat16)
    kw = (torch.rand(D, generator=g) + 0.5).to("cuda", torch.bfloat16)
    freqs = torch.randn(T, R // 2, generator=g).to("cuda")
    cache = torch.cat((freqs.cos(), freqs.sin()), -1).to(torch.bfloat16).contiguous()
    pos = torch.arange(T, device="cuda")
    kwargs = dict(
        is_neox=True, eps=1e-5, head_dim=D, rope_dim=R, round_norm_before_rope=True
    )
    q_ref, k_ref = q.clone(), k.clone()
    fused_inplace_qknorm_rope(q_ref, k_ref, qw, kw, cache, pos, **kwargs)
    q_out = torch.empty(T, H, D, device="cuda", dtype=torch.bfloat16)
    k_out = torch.empty_like(q_out)
    before = qkv.clone()
    fused_qknorm_rope_out_of_place(q, k, q_out, k_out, qw, kw, cache, pos, **kwargs)
    assert torch.equal(qkv, before)
    assert torch.equal(q_out, q_ref) and torch.equal(k_out, k_ref)


def _ulp_distance_16bit(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Representable steps between two 16-bit float tensors of one dtype; +-0 coincide."""

    def key(t: torch.Tensor) -> torch.Tensor:
        bits = t.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
        return torch.where(bits >= 0x8000, -(bits - 0x8000), bits)

    return (key(a) - key(b)).abs()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_weightless_partial_rope_matches_minimax_h3_eager_chain(dtype) -> None:
    """MiniMax-H3's ViT decoder QK path -- weightless RMSNorm over head_dim=64,
    rotate-half RoPE over the first 48 dims, on the Q/K views of the packed
    [B, S, 32, 192] QKV projection -- needs no kernel of its own: ones weights
    are an IEEE identity, the (d, d + 24) pairs are NeoX, and the compact
    [B*S, 48] cache is cos24|sin24. The fused kernel must be torch.equal to
    fused_inplace_qknorm + sgl_kernel.rotary_embedding; against the model's own
    nn.RMSNorm + rotary_embedding chain the norm stage is within one 16-bit ulp
    and the full chain within one ulp of the largest normalized operand."""
    from sgl_kernel import rotary_embedding

    B, S, H, D, R = 2, 257, 32, 64, 48
    if not can_use_fused_inplace_qknorm_rope(D, R, True, dtype, dtype, True):
        pytest.skip("fused qknorm+rope JIT kernel unavailable")
    g = torch.Generator(device="cpu").manual_seed(0)
    qkv = torch.randn(B, S, H, 3 * D, generator=g).to("cuda", dtype)
    qkv[..., :D].mul_(0.3)
    qkv[..., D : 2 * D].mul_(7)
    tokens = B * S
    q = qkv.view(tokens, H, 3 * D)[..., :D]
    k = qkv.view(tokens, H, 3 * D)[..., D : 2 * D]
    ones = torch.ones(D, device="cuda", dtype=dtype)
    freqs = torch.randn(tokens, R // 2, generator=g).to("cuda")
    cache = torch.cat((freqs.cos(), freqs.sin()), -1).to(dtype).contiguous()
    positions = torch.arange(tokens, device="cuda")
    kwargs = dict(
        is_neox=True, eps=1e-5, head_dim=D, rope_dim=R, round_norm_before_rope=True
    )

    q_out = torch.empty(tokens, H, D, device="cuda", dtype=dtype)
    k_out = torch.empty_like(q_out)
    before = qkv.clone()
    fused_qknorm_rope_out_of_place(
        q, k, q_out, k_out, ones, ones, cache, positions, **kwargs
    )
    assert torch.equal(qkv, before)

    q_jit, k_jit = q.contiguous(), k.contiguous()
    fused_inplace_qknorm(q_jit, k_jit, ones, ones, 1e-5, head_dim=D)
    q_ref, k_ref = q_jit.clone(), k_jit.clone()
    rotary_embedding(positions, q_ref, k_ref, D, cache, True)
    assert torch.equal(q_out, q_ref) and torch.equal(k_out, k_ref)

    # The model's own chain differs only in the norm stage: aten's RMSNorm and
    # the JIT warp RMSNorm reduce in fp32 in different orders, which flips the
    # 16-bit rounding of a rare element (49 of 2.1M for fp16, 3 for bf16). RoPE
    # is a rotation, so it keeps that absolute error but not its ulp count near
    # cancellation; the full chain is bounded by one ulp of the largest operand.
    norm = torch.nn.RMSNorm(D, eps=1e-5, elementwise_affine=False).to("cuda")
    q_model, k_model = norm(q).contiguous(), norm(k).contiguous()
    for jit_norm, aten_norm in ((q_jit, q_model), (k_jit, k_model)):
        assert int(_ulp_distance_16bit(jit_norm, aten_norm).max()) <= 1
    one_ulp = torch.finfo(dtype).eps * max(
        q_model.abs().max().item(), k_model.abs().max().item()
    )
    rotary_embedding(positions, q_model, k_model, D, cache, True)
    for actual, expected in ((q_out, q_model), (k_out, k_model)):
        assert (actual.float() - expected.float()).abs().max().item() <= one_ulp


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
