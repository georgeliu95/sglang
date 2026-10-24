# SPDX-License-Identifier: Apache-2.0
"""Quality-gated CUDA fast path for the MiniMax-H3 video VAE.

Installed once at VAE load on CUDA SM100+. Every encode and decode scope
toggles the shared :class:`VaeFastPathGate`: ``quality="extra-high"`` and
``"high"`` batch equal-shaped tiles and temporal windows and dispatch the JIT
kernels (GroupNorm+SiLU, paired Q/K RMSNorm+RoPE, tile assembly, temporal blend
write, output denormalization), while the ``"lossless"`` default runs the
original module path bit-for-bit.
"""

import torch.nn as nn

from sglang.kernels.ops.diffusion.common.platform import is_cuda_sm_at_least
from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.models.vaes.fast_path_gate import (
    VaeFastPathGate,
    register_vae_fast_path_gate,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.attention import (
    Attention,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.fast_path import (
    MINIMAX_H3_VAE_MIN_SM,
    MiniMaxH3VaeFastPath,
    resolve_minimax_h3_vae_batch_caps,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.vae_cnn import (
    EncoderFCN3D,
    ResnetBlock3D,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.vae_vit import (
    ViT3DDecoder,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _install_fast_path_slots(
    vae: nn.Module, state: MiniMaxH3VaeFastPath
) -> tuple[int, int]:
    """Fill every explicit ``fast_path`` field; returns (norm, attention) site counts."""
    vae.fast_path = state
    vae.processor.fast_path = state
    norm_sites = 0
    attention_sites = 0
    for module in vae.modules():
        if isinstance(module, (ResnetBlock3D, EncoderFCN3D, ViT3DDecoder)):
            module.fast_path = state
            norm_sites += 2 if isinstance(module, ResnetBlock3D) else 0
            norm_sites += 1 if isinstance(module, EncoderFCN3D) else 0
        elif isinstance(module, Attention):
            module.fast_path = state
            attention_sites += 1
    return norm_sites, attention_sites


def maybe_optimize_minimax_h3_vae(vae: nn.Module) -> nn.Module:
    """Install the quality-gated MiniMax-H3 video VAE fast path."""
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3 import (
        MiniMaxH3VideoVAE,
    )

    if not isinstance(vae, MiniMaxH3VideoVAE):
        return vae
    label = "MiniMax-H3 video VAE"
    if envs.SGLANG_DIFFUSION_DISABLE_MINIMAX_H3_VAE_FAST_PATH:
        logger.info(
            "%s: fast path disabled by "
            "SGLANG_DIFFUSION_DISABLE_MINIMAX_H3_VAE_FAST_PATH.",
            label,
        )
        return vae
    if not is_cuda_sm_at_least(MINIMAX_H3_VAE_MIN_SM):
        logger.info("%s: fast path needs CUDA SM100 or newer; skipping.", label)
        return vae

    encoder_tile_batch, decoder_tile_batch, window_batch = (
        resolve_minimax_h3_vae_batch_caps()
    )
    state = MiniMaxH3VaeFastPath(
        gate=VaeFastPathGate(),
        encoder_tile_batch=encoder_tile_batch,
        decoder_tile_batch=decoder_tile_batch,
        window_batch=window_batch,
    )
    norm_sites, attention_sites = _install_fast_path_slots(vae, state)
    register_vae_fast_path_gate(vae, state.gate)
    logger.info(
        "%s: installed quality-gated fast path (%d GroupNorm+SiLU sites, "
        "%d Q/K norm+RoPE sites, batch caps enc%d/dec%d/win%d).",
        label,
        norm_sites,
        attention_sites,
        encoder_tile_batch,
        decoder_tile_batch,
        window_batch,
    )
    return vae
