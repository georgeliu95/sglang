# SPDX-License-Identifier: Apache-2.0
"""Cross-block dataflow of the MiniMax-H3 FP8 decoder on the JIT producers.

``ViT3DDecoder._run_blocks`` takes this path when :func:`can_use_fused_decoder`
admits the whole decoder, i.e. the FP8 linears are installed and the request
quality mounts the fast path. Each block then reads its input from one kernel
that applies the pending residual add, the RMSNorm and the per-token FP8
quantization; the gated FFN activation is produced already quantized, and the
final residual add feeds the output LayerNorm directly. Attention and the
weightless Q/K RMSNorm+RoPE reuse ``Attention``. Selection is all-or-nothing:
a structural mismatch runs the eager blocks, a kernel failure raises.
"""

from typing import NamedTuple

import torch
from torch import nn

from sglang.kernels.ops.diffusion import (
    can_use_fused_inplace_qknorm_rope,
    can_use_minimax_h3_vae_rmsnorm_fp8,
    minimax_h3_vae_residual_layernorm,
    minimax_h3_vae_residual_rmsnorm_fp8,
    minimax_h3_vae_rmsnorm_fp8,
    silu_mul_quant_fp8,
)

from .attention import _weightless_rmsnorm
from .fp8 import MiniMaxH3FP8Linear


class ResidualPending(NamedTuple):
    """``residual + projected * scale`` left for the next norm kernel to apply."""

    residual: torch.Tensor
    projected: torch.Tensor
    scale: torch.Tensor


def _vector(value, size, device) -> bool:
    return (
        value is not None
        and value.shape == (size,)
        and value.dtype == torch.float32
        and value.device == device
        and value.is_contiguous()
    )


def _rope_ok(rope, *, rows, dim_head, rope_dim, device) -> bool:
    if len(rope) != 4:
        return False
    cache = rope[2]
    return (
        cache.device == device
        and cache.shape == (rows, rope_dim)
        and can_use_fused_inplace_qknorm_rope(
            dim_head, rope_dim, True, torch.float16, cache.dtype, True
        )
    )


def _block_ok(block, *, fast_path, width, dim_head, device) -> bool:
    if not (
        block.use_scale
        and _vector(block.scale1, width, device)
        and _vector(block.scale2, width, device)
    ):
        return False
    for norm in (block.norm1, block.norm2):
        if not (
            isinstance(norm, nn.RMSNorm)
            and _vector(norm.weight, width, device)
            and norm.eps > 0
        ):
            return False
    attn = block.attn
    if not (
        attn.fast_path is fast_path
        and attn.dim_head == dim_head
        and attn.attn is not None
        and _weightless_rmsnorm(attn.norm_q)
        and _weightless_rmsnorm(attn.norm_k)
        and attn.norm_q.eps == attn.norm_k.eps
    ):
        return False
    if not (block.ff.use_gated and isinstance(block.ff.act_fn, nn.SiLU)):
        return False
    return all(
        isinstance(module, MiniMaxH3FP8Linear)
        for module in (attn.to_qkv, attn.to_out, block.ff.w1, block.ff.w2)
    )


def can_use_fused_decoder(decoder, value, rope) -> bool:
    fast_path = decoder.fast_path
    if not (
        fast_path is not None and fast_path.active(value) and decoder.fp8_installed
    ):
        return False
    width = decoder.norm_out.normalized_shape[0]
    dim_head = decoder.config.dim_head
    device = value.device
    first_norm = decoder.transformer_blocks[0].norm1
    admitted = (
        value.dtype == torch.float32
        and value.dim() == 3
        and value.shape[-1] == width
        and not decoder.mask_enabled
        and _rope_ok(
            rope,
            rows=value.shape[0] * value.shape[1],
            dim_head=dim_head,
            rope_dim=decoder.pos_embed.dim,
            device=device,
        )
        and isinstance(decoder.norm_out, nn.LayerNorm)
        and _vector(decoder.norm_out.weight, width, device)
        and _vector(decoder.norm_out.bias, width, device)
        and all(
            _block_ok(
                block,
                fast_path=fast_path,
                width=width,
                dim_head=dim_head,
                device=device,
            )
            for block in decoder.transformer_blocks
        )
        # Later block inputs are fresh kernel and GEMM outputs; only the
        # decoder input can violate the row layout the producers require.
        and can_use_minimax_h3_vae_rmsnorm_fp8(
            value, first_norm.weight, eps=first_norm.eps
        )
    )
    return fast_path.admit(admitted)


def _quantized_input(value, norm):
    """fp32 residual plus the FP8 rows of ``RMSNorm(residual) * weight``."""
    if isinstance(value, ResidualPending):
        return minimax_h3_vae_residual_rmsnorm_fp8(
            value.residual, value.projected, value.scale, norm.weight, eps=norm.eps
        )
    q, scales = minimax_h3_vae_rmsnorm_fp8(value, norm.weight, eps=norm.eps)
    return value, q, scales


def _attention(attn, q, scales, shape, rope):
    packed = attn.to_qkv.forward_quantized(q, scales, shape).view(
        shape[0], shape[1], attn.heads, 3 * attn.dim_head
    )
    fused = attn._paired_qk(packed, rope)
    if fused is None:
        raise RuntimeError(
            "MiniMax-H3 fused decoder admitted a block whose paired Q/K kernel "
            "then refused the packed QKV"
        )
    query, key, value = fused
    attended = attn.attn(query, key, value).reshape(shape[0], shape[1], -1)
    return attn.to_out(attended)


def block_forward(module, value, rope):
    residual, q, scales = _quantized_input(value, module.norm1)
    shape = tuple(residual.shape)
    projected = _attention(module.attn, q, scales, shape, rope)
    residual, q, scales = minimax_h3_vae_residual_rmsnorm_fp8(
        residual, projected, module.scale1, module.norm2.weight, eps=module.norm2.eps
    )
    expanded = module.ff.w1.forward_quantized(q, scales, shape)
    q, scales = silu_mul_quant_fp8(expanded)
    projected = module.ff.w2.forward_quantized(
        q, scales, (*shape[:-1], expanded.shape[-1] // 2)
    )
    # The last add stays pending: the next block folds it into its norm/quant
    # kernel and the final block hands it to the output LayerNorm.
    return ResidualPending(residual, projected, module.scale2)


def finish(value, norm):
    return minimax_h3_vae_residual_layernorm(
        value.residual,
        value.projected,
        value.scale,
        norm.weight,
        norm.bias,
        eps=norm.eps,
    )


def output_projection(module, value, *, cache):
    """Half-input/FP32-output GEMM without mutating global math flags.

    Returns ``(result, cache)``; ``result`` is None when fp16 accumulation is
    already enabled globally. The caller owns ``cache``, the rounded transposed
    weight keyed by (device, dtype, data_ptr, version).
    """
    if torch.backends.cuda.matmul.allow_fp16_accumulation:
        return None, cache
    # Inference tensors have no version counter. Rebuild their rounded view
    # so an in-place checkpoint reload cannot reuse stale projection weights.
    version = None if module.weight.is_inference() else module.weight._version
    key = (module.weight.device, module.weight.dtype, module.weight.data_ptr(), version)
    if version is None or cache is None or cache[0] != key:
        cache = (key, module.weight.detach().to(torch.float16).t().contiguous())
    with torch.autocast("cuda", enabled=False):
        result = torch.mm(
            value.reshape(-1, value.shape[-1]).to(torch.float16),
            cache[1],
            out_dtype=torch.float32,
        )
        if module.bias is not None:
            result = result + module.bias.float()
    return result.reshape(*value.shape[:-1], module.out_features), cache
