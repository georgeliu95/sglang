"""Cross-block MiniMax-H3 FP8 decoder dataflow on the JIT producers.

Two real ``ViT3DDecoder`` blocks with FP8 linears run once through the eager
block path and once through ``fused_decoder`` (rmsnorm_fp8 /
residual_rmsnorm_fp8 / silu_mul_quant_fp8 / residual_layernorm); the
half-input fp32-output projection GEMM and the load-time installer on the
released 36-block topology are checked on their own.
"""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.diffusion.common.platform import is_cuda_sm_at_least
from sglang.multimodal_gen.runtime.layers.attention.selector import (
    global_force_attn_backend_context_manager,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.vaes.fast_path_gate import (
    VaeFastPathGate,
    register_vae_fast_path_gate,
    use_vae_fast_path,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae import (
    fused_decoder,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.attention import (
    Attention,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.fast_path import (
    MiniMaxH3VaeFastPath,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.fp8 import (
    MiniMaxH3FP8Linear,
    install_fp8_block_linears,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.vae_vit import (
    ViT3DDecoder,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.vit_utils import (
    prepare_rotary_pos_emb,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.server_args import server_args as server_args_module
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not is_cuda_sm_at_least((10, 0)),
    reason="MiniMax-H3 VAE FP8 producers are gated to SM100+",
)


@pytest.fixture(autouse=True)
def inference():
    torch.manual_seed(20260920)
    with torch.inference_mode():
        yield


@pytest.fixture
def stub_server_args(monkeypatch):
    """Only server configuration is stubbed; every attention, quantizer, GEMM,
    norm and activation under test is the production implementation."""
    args = type("Args", (), {})()
    args.attention_backend = None
    args.attention_backend_config = None
    args.enable_attention_backend_autotune = False
    args.enable_breakable_cuda_graph = False
    args.enable_layerwise_nvtx_marker = False
    args.kv_gather_degree = 1
    args.sp_split_auto = False
    monkeypatch.setattr(server_args_module, "_global_server_args", args)


def record_error(record_property, actual, expected, prefix="output"):
    difference = (actual.float() - expected.float()).abs()
    finite = difference[torch.isfinite(difference)]
    if finite.numel():
        record_property(f"{prefix}_max_abs", finite.max().item())
        record_property(f"{prefix}_rms_abs", finite.square().mean().sqrt().item())


def _released_decoder(num_layers: int) -> ViT3DDecoder:
    """The MiniMax-H3 ViT decoder topology with ``num_layers`` blocks."""
    return ViT3DDecoder(
        num_layers=num_layers,
        heads=32,
        dim_head=64,
        in_channels=24,
        patch_size=1,
        patch_size_t=1,
        norm_type="rms_norm",
        qk_norm_type="rms_norm",
        qk_norm_affine=False,
        ffn_activation_fn="silu",
        ffn_use_gated=True,
        rope_dim_ratio=0.75,
    )


def _mount_fast_path(decoder: ViT3DDecoder) -> MiniMaxH3VaeFastPath:
    """Fill the decoder's explicit fast_path slots as the load-time installer does."""
    state = MiniMaxH3VaeFastPath(
        gate=VaeFastPathGate(),
        encoder_tile_batch=8,
        decoder_tile_batch=64,
        window_batch=1,
    )
    for module in decoder.modules():
        if isinstance(module, (ViT3DDecoder, Attention)):
            module.fast_path = state
    register_vae_fast_path_gate(decoder, state.gate)
    return state


@pytest.mark.parametrize("backend", ["TORCH_SDPA", "TORCH_CUDNN_SDPA"])
def test_two_real_decoder_blocks(backend, stub_server_args, record_property):
    """Same FP8 weights through eager blocks versus cross-block JIT dataflow."""
    with global_force_attn_backend_context_manager(AttentionBackendEnum[backend]):
        decoder = _released_decoder(num_layers=2).cuda().eval()
    decoder.prepare_autocast_linear_weights(torch.float16)
    for block in decoder.transformer_blocks:
        # Constructor scales are zero: leaving them untouched would hide all
        # attention/FFN errors behind the residual identity.
        block.scale1.uniform_(0.2, 0.8)
        block.scale2.uniform_(0.2, 0.8)
        block.norm1.weight.uniform_(0.8, 1.2)
        block.norm2.weight.uniform_(0.8, 1.2)
        # install_fp8_block_linears insists on the released 36 blocks; swap the
        # two blocks by hand with the same module it installs.
        block.attn.to_qkv = MiniMaxH3FP8Linear(block.attn.to_qkv)
        block.attn.to_out = MiniMaxH3FP8Linear(block.attn.to_out)
        block.ff.w1 = MiniMaxH3FP8Linear(block.ff.w1)
        block.ff.w2 = MiniMaxH3FP8Linear(block.ff.w2)
    decoder.fp8_installed = True
    state = _mount_fast_path(decoder)
    value = torch.randn(2, 65, 2048, device="cuda")
    token_ids = torch.randn(2, 65, 3, device="cuda")
    rope = prepare_rotary_pos_emb(
        decoder.pos_embed(token_ids), dtype=torch.float16, allow_batched_native=True
    )

    with set_forward_context(0, None), torch.autocast("cuda", dtype=torch.float16):
        with use_vae_fast_path(decoder, False):
            assert not fused_decoder.can_use_fused_decoder(decoder, value, rope)
            reference = value
            for block in decoder.transformer_blocks:
                reference = block(reference, rope)
            reference = decoder.norm_out(reference)
        with use_vae_fast_path(decoder, True):
            assert fused_decoder.can_use_fused_decoder(decoder, value, rope)
            pending = value
            for block in decoder.transformer_blocks:
                pending = fused_decoder.block_forward(block, pending, rope)
                assert isinstance(pending, fused_decoder.ResidualPending)
            actual = fused_decoder.finish(pending, decoder.norm_out)
    assert not state.gate.enabled
    # One admission for the decoder plus one paired Q/K dispatch per block.
    assert (state.used, state.fallback) == (3, 0)
    assert actual.dtype == torch.float32 and actual.shape == reference.shape
    record_error(record_property, actual, reference)
    relative_rms = (
        ((actual - reference).square().mean() / reference.square().mean()).sqrt().item()
    )
    cosine = F.cosine_similarity(actual.flatten(), reference.flatten(), dim=0).item()
    record_property("relative_rms", relative_rms)
    record_property("cosine", cosine)
    # Fused producers remove intermediate FP16 rounding before FP8. This is
    # a close-contract comparison with identical deployed FP8 weights.
    assert relative_rms < 0.01
    assert cosine > 0.99995
    torch.testing.assert_close(actual, reference, atol=0.075, rtol=0.02)


def test_install_fp8_block_linears_swaps_the_released_144_linears(stub_server_args):
    """The load-time installer converts exactly the 36 x 4 block linears once."""
    with torch.device("cuda"):
        decoder = _released_decoder(num_layers=36).eval()
    assert install_fp8_block_linears(decoder) == 144
    assert decoder.fp8_installed
    swapped = [m for m in decoder.modules() if isinstance(m, MiniMaxH3FP8Linear)]
    assert len(swapped) == 144
    assert all(m.weight.dtype == torch.float8_e4m3fn for m in swapped)
    assert all(m.weight_scale.dtype == torch.float32 for m in swapped)
    assert all(m.bias.dtype == torch.float16 for m in swapped)
    assert decoder.proj_out.weight.dtype == torch.float32
    assert decoder.x_embedder.weight.dtype == torch.float32
    # A second call and the fp16 decode cast are no-ops; bf16 is refused.
    assert install_fp8_block_linears(decoder) == 0
    assert decoder.prepare_autocast_linear_weights(torch.float16) == 0
    with pytest.raises(ValueError, match="FP16 autocast"):
        decoder.prepare_autocast_linear_weights(torch.bfloat16)


@pytest.mark.parametrize("inference_weights", [False, True])
def test_output_projection_rounding_and_cache(inference_weights, record_property):
    """Half-input fp32-output projection: rounding matches an fp64 reference and
    the cached half weight follows in-place weight updates."""
    output_projection = fused_decoder.output_projection
    with torch.inference_mode(inference_weights):
        module = torch.nn.Linear(2048, 3072, device="cuda", dtype=torch.float32).eval()
    assert module.weight.is_inference() == inference_weights
    value = torch.randn(2, 11, 2048, device="cuda")
    flags = (
        torch.backends.cuda.matmul.allow_fp16_accumulation,
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        torch.backends.cuda.matmul.allow_tf32,
    )
    if flags[0]:
        assert output_projection(module, value, cache=None)[0] is None
        return

    def check(cache):
        # FP64 reference isolates accumulation from intentional input/weight
        # rounding; it also works on Torch versions without mm(out_dtype=...).
        reference = (
            value.half().double().reshape(-1, 2048) @ module.weight.half().double().t()
        ).float()
        reference = (reference + module.bias).reshape(2, 11, 3072)
        actual, cache = output_projection(module, value, cache=cache)
        assert actual is not None and actual.dtype == torch.float32
        record_error(record_property, actual, reference)
        torch.testing.assert_close(actual, reference, atol=2e-5, rtol=1e-5)
        return actual, cache

    first, cache = check(None)
    cached_weight = cache[1]
    second, cache = check(cache)
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    if not inference_weights:
        assert cache[1] is cached_weight
    module.weight.add_(0.125)
    third, cache = check(cache)
    assert not torch.equal(first, third)
    assert cache[1] is not cached_weight
    assert flags == (
        torch.backends.cuda.matmul.allow_fp16_accumulation,
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        torch.backends.cuda.matmul.allow_tf32,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
