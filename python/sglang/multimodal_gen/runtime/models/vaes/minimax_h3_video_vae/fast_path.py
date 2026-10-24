# SPDX-License-Identifier: Apache-2.0
"""Scope state of the quality-gated MiniMax-H3 video VAE fast path.

``minimax_h3_vae_cuda_opt`` fills one :class:`MiniMaxH3VaeFastPath` into the
explicit ``fast_path`` field of every module that reads it. A call site tests
``fast_path is not None and fast_path.active(x)`` and then the kernel-level
``can_use_*`` predicate through :meth:`MiniMaxH3VaeFastPath.admit`; when either
check fails the eager operators run unchanged.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

from sglang.kernels.ops.diffusion.common.platform import is_cuda_sm_at_least
from sglang.multimodal_gen import envs
from sglang.multimodal_gen.configs.sample.sampling_params import (
    quality_allows_kernel_fusions,
)
from sglang.multimodal_gen.runtime.models.vaes.fast_path_gate import (
    VaeFastPathGate,
    use_vae_fast_path,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Blackwell floor of the JIT kernels; the sglang.kernels predicates re-check it.
MINIMAX_H3_VAE_MIN_SM = (10, 0)
# Upper bound of all three batch caps: the assemble kernel's tile table holds
# 64 tiles; for the window cap the value is arbitrary.
MINIMAX_H3_VAE_MAX_BATCH = 64
# Encoder-side rounding changes perturb the conditioning latents and the
# denoiser amplifies them: fl2va worst-frame PSNR 28-38 dB vs 50 dB for the
# decode-only path (8x SM120, 4-5 steps), so encode follows the approximate tier.
MINIMAX_H3_VAE_ENCODE_QUALITY_LEVELS = frozenset({"high"})


class MiniMaxH3VaeFastPath:
    """Gate, batch caps and per-scope dispatch counters of one installed VAE."""

    __slots__ = (
        "gate",
        "encoder_tile_batch",
        "decoder_tile_batch",
        "window_batch",
        "used",
        "fallback",
    )

    def __init__(
        self,
        *,
        gate: VaeFastPathGate,
        encoder_tile_batch: int,
        decoder_tile_batch: int,
        window_batch: int,
    ) -> None:
        self.gate = gate
        self.encoder_tile_batch = encoder_tile_batch
        self.decoder_tile_batch = decoder_tile_batch
        self.window_batch = window_batch
        self.used = 0
        self.fallback = 0

    def active(self, value: torch.Tensor) -> bool:
        return (
            self.gate.enabled
            and value.is_cuda
            and is_cuda_sm_at_least(MINIMAX_H3_VAE_MIN_SM, value.device)
            and not torch.is_grad_enabled()
            and not torch.compiler.is_compiling()
            and not torch.cuda.is_current_stream_capturing()
        )

    def admit(self, supported: bool) -> bool:
        """Count one kernel-predicate outcome at an active call site."""
        if supported:
            self.used += 1
        else:
            self.fallback += 1
        return supported


def resolve_minimax_h3_vae_batch_caps() -> tuple[int, int, int]:
    """Read the (encoder tile, decoder tile, window) batch caps; bad values raise."""
    return (
        _checked_batch_cap(
            "SGLANG_DIFFUSION_MINIMAX_H3_VAE_ENCODER_TILE_BATCH",
            envs.SGLANG_DIFFUSION_MINIMAX_H3_VAE_ENCODER_TILE_BATCH,
            low=1,
        ),
        _checked_batch_cap(
            "SGLANG_DIFFUSION_MINIMAX_H3_VAE_DECODER_TILE_BATCH",
            envs.SGLANG_DIFFUSION_MINIMAX_H3_VAE_DECODER_TILE_BATCH,
            low=1,
        ),
        _checked_batch_cap(
            "SGLANG_DIFFUSION_MINIMAX_H3_VAE_WINDOW_BATCH",
            envs.SGLANG_DIFFUSION_MINIMAX_H3_VAE_WINDOW_BATCH,
            low=0,
        ),
    )


def _checked_batch_cap(name: str, value: int, *, low: int) -> int:
    if not low <= value <= MINIMAX_H3_VAE_MAX_BATCH:
        raise ValueError(
            f"{name} must be an integer in [{low}, {MINIMAX_H3_VAE_MAX_BATCH}], "
            f"got {value}"
        )
    return value


@contextmanager
def minimax_h3_vae_fast_path_scope(vae, *, quality: str, stage: str):
    """Mount the installed fast path for one encode or decode and log its use."""
    if stage == "encode":
        enabled = quality in MINIMAX_H3_VAE_ENCODE_QUALITY_LEVELS
    else:
        enabled = quality_allows_kernel_fusions(quality)
    state = vae.fast_path
    with use_vae_fast_path(vae, enabled):
        if state is None or not enabled:
            yield
            return
        state.used = 0
        state.fallback = 0
        logger.info(
            "[H3 VAE] %s fast path mounted: quality=%s caps=enc%d/dec%d/win%d fp8=%s",
            stage,
            quality,
            state.encoder_tile_batch,
            state.decoder_tile_batch,
            state.window_batch,
            vae.decoder.fp8_installed,
        )
        try:
            yield
        finally:
            logger.info(
                "[H3 VAE] %s fast path: used=%d fallback=%d",
                stage,
                state.used,
                state.fallback,
            )
