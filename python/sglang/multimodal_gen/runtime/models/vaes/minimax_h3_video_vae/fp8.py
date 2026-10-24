# SPDX-License-Identifier: Apache-2.0
"""Explicit online FP8 for the H3 decoder's 144 block linears only.

The encoder, embedding, normalization, attention and output projection keep
those components' own precision policies. We reuse the existing per-token
quantizer and scaled GEMM; no private ABI manifests or alternate GEMM exist.
``install_fp8_block_linears`` runs once at load, from
``MiniMaxH3VideoVAE.quantize_decoder_fp8``, when the deployment passes
``--component-quantizations.video_vae fp8``.
"""

from functools import lru_cache
from math import prod

import torch
from torch import nn

LINEAR_SHAPES = {
    "attn.to_qkv": (6144, 2048),
    "attn.to_out": (2048, 2048),
    "ff.w1": (16384, 2048),
    "ff.w2": (2048, 8192),
}


def target_paths():
    return [
        f"transformer_blocks.{index}.{suffix}"
        for index in range(36)
        for suffix in LINEAR_SHAPES
    ]


def inspect_fp8_scope(decoder):
    if len(decoder.transformer_blocks) != 36:
        raise ValueError("MiniMax-H3 online FP8 requires its 36-block decoder")
    for name in target_paths():
        module = decoder.get_submodule(name)
        suffix = ".".join(name.split(".")[2:])
        if (
            not isinstance(module, (nn.Linear, MiniMaxH3FP8Linear))
            or tuple(module.weight.shape) != LINEAR_SHAPES[suffix]
        ):
            raise ValueError(f"Unexpected MiniMax-H3 FP8 linear scope: {name}")
        if module.bias is None or module.bias.shape != (module.out_features,):
            raise ValueError(f"Expected the released biased linear at {name}")
    return target_paths()


@lru_cache(None)
def _primitives():
    from sgl_kernel import fp8_scaled_mm

    from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_quant_fp8

    return sglang_per_token_quant_fp8, fp8_scaled_mm


class MiniMaxH3FP8Linear(nn.Module):
    """Static per-channel weights, dynamic per-token activations, FP16 output."""

    def __init__(self, source: nn.Linear):
        super().__init__()
        if (
            source.weight.dtype != torch.float16
            or source.bias is None
            or source.bias.dtype != torch.float16
            or not source.weight.is_cuda
        ):
            raise ValueError(
                "Online FP8 requires resident, prepared FP16 decoder weights"
            )
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.train(source.training)
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            rounded = source.weight.detach().float()
            amax = rounded.abs().amax(dim=1, keepdim=True)
            scale = torch.where(amax == 0, torch.ones_like(amax), amax / 448.0)
            self.weight = nn.Parameter(
                (rounded / scale).clamp(-448, 448).to(torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.register_buffer("weight_scale", scale.contiguous())
            self.bias = nn.Parameter(source.bias.detach(), requires_grad=False)

    def _apply(self, fn, recurse=True):
        # Whole-VAE dtype scopes also cover the keyframe encoder. Such a scope
        # must never expand FP8 storage or cast its scales/bias to another dtype.
        device = fn(
            torch.empty(0, dtype=self.weight.dtype, device=self.weight.device)
        ).device
        if device.type not in ("cuda", "cpu"):
            raise ValueError("MiniMax-H3 FP8 storage supports CPU/CUDA moves only")
        self.weight = nn.Parameter(
            self.weight.detach().to(device=device, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        self.weight_scale = self.weight_scale.to(device=device, dtype=torch.float32)
        self.bias = nn.Parameter(
            self.bias.detach().to(device=device, dtype=torch.float16),
            requires_grad=False,
        )
        return self

    def forward_quantized(self, values, scales, shape):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("MiniMax-H3 FP8 linears are inference-only")
        if (
            len(shape) < 2
            or shape[-1] != self.in_features
            or values.shape != (prod(shape[:-1]), self.in_features)
            or values.dtype != torch.float8_e4m3fn
            or not values.is_contiguous()
            or scales.shape != (values.shape[0], 1)
            or scales.dtype != torch.float32
            or not scales.is_contiguous()
            or not values.is_cuda
            or values.device != self.weight.device
            or scales.device != values.device
        ):
            raise ValueError("Unsupported prequantized MiniMax-H3 FP8 activation")
        _, gemm = _primitives()
        with torch.autocast("cuda", enabled=False):
            out = gemm(
                values,
                self.weight.t(),
                scales,
                self.weight_scale,
                out_dtype=torch.float16,
                bias=self.bias,
            )
        return out.reshape(*shape[:-1], self.out_features)

    def forward(self, value):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("MiniMax-H3 FP8 linears are inference-only")
        if (
            not value.is_cuda
            or value.device != self.weight.device
            or value.shape[-1] != self.in_features
            or value.dtype not in (torch.float16, torch.float32)
        ):
            raise ValueError("Unsupported MiniMax-H3 FP8 activation")
        if (
            not torch.is_autocast_enabled("cuda")
            or torch.get_autocast_dtype("cuda") != torch.float16
        ):
            raise ValueError("MiniMax-H3 FP8 requires FP16 decode autocast")
        quant, _ = _primitives()
        with torch.autocast("cuda", enabled=False):
            # Preserve the original Linear autocast rounding before FP8.
            rounded = value.to(torch.float16).reshape(-1, self.in_features).contiguous()
            values, scales = quant(rounded, dtype=torch.float8_e4m3fn)
        return self.forward_quantized(values, scales, value.shape)


def install_fp8_block_linears(decoder) -> int:
    """Swap the block linears for FP8 modules; returns how many were replaced."""
    if decoder.fp8_installed:
        return 0
    # Validate every target before replacing any module; an offloaded or
    # unprepared decoder fails here with the original modules intact.
    paths = inspect_fp8_scope(decoder)
    decoder.prepare_autocast_linear_weights(torch.float16)
    # Import the quantizer and GEMM now so a missing sgl_kernel fails the load.
    _primitives()
    for path in paths:
        parent, name = path.rsplit(".", 1)
        source = decoder.get_submodule(path)
        decoder.get_submodule(parent)._modules[name] = MiniMaxH3FP8Linear(source)
    decoder.fp8_installed = True
    decoder._autocast_linear_dtype = torch.float16
    return len(paths)
