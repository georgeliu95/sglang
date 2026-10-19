# SPDX-License-Identifier: Apache-2.0
"""FlashInfer prepared Sage2 QKV exchange followed by CAKE SSA attention.

FlashInfer PR #4924 quantizes Q/K/V before the combined byte all-to-all.
The merged PCIe/RDMA backend (#4876) transports those bytes through
exchange_chunks; NCCL remains the statistics AllGather backend. CAKE (#4951
and subsequent tail-specialization fixes) consumes INT8 Q/K, FP8 V and the
same scales directly. No external SageAttention Ulysses ABI is required.

Unsupported geometries decline before communication and use the standalone
Sage SSA implementation. Preparation or execution failures propagate to the
caller, rather than attempting a rank-local fallback after a collective.
"""

from __future__ import annotations

import functools
import os

import torch
from sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse import (
    SubBlockRouter,
    load_bsa_attn_sm120_blk64_sage_fwd,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse_attn import (
    SubBlockSparseSchedule as SubBlockSparseScheduleSM120,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse_attn_sm120_sage import (
    SubBlockSparseSageAttentionSM120Backend,
    SubBlockSparseSageAttentionSM120Impl,
)
from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

BLOCK = 64


def _dequantize_for_routing(
    q_int8: torch.Tensor, k_int8: torch.Tensor, k_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rebuild BF16 Q/K good enough to rank key blocks.

    The router only needs the *order* of the per-(query block, key block)
    scores, so two of the three quantization terms can be skipped outright:

    * Q's per-32-token scale is one positive factor per query row -- it scales
      a whole score row, and ``topk`` within a row is invariant to that.
    * K's channel mean was subtracted before quantization; adding it back
      contributes ``q · k_mean``, the same constant to every key block in a
      row, so it cannot reorder them either.

    K's per-64-tile scale is the one that *does* vary across key blocks, so it
    has to be applied.  ``k_scale`` is one value per 64-token tile, laid out
    ``[B, H, ceil(S/64)]``.
    """
    q_bf16 = q_int8.to(torch.bfloat16)
    seq_len = k_int8.shape[2]
    tiles = -(-seq_len // BLOCK)
    # Expand the per-tile scale to per-token and slice: the sequence is not a
    # multiple of 64 (H3 packs video+audio into one document, e.g. S=37874),
    # so reshaping into whole tiles would not fit.
    per_token = (
        k_scale[:, :, :tiles]
        .to(torch.bfloat16)
        .repeat_interleave(BLOCK, dim=-1)[:, :, :seq_len]
    )
    k_bf16 = k_int8.to(torch.bfloat16) * per_token.unsqueeze(-1)
    return q_bf16, k_bf16


class _SparseSageA2ACompute:
    """Consume the prepared INT8/FP8 operands without requantizing them."""

    def __init__(
        self,
        schedule: SubBlockSparseScheduleSM120,
        query_block_mask=None,
        layer_enabled=True,
    ) -> None:
        self.query_block_mask = query_block_mask
        self.layer_enabled = layer_enabled
        self.schedule = schedule
        self.router = (
            None
            if schedule.sparsity <= 0.0
            else SubBlockRouter(n_k=schedule.n_k, n_q=schedule.n_q)
        )

    def _route_this_step(self, seq_len: int) -> bool:
        """Mirror ``SubBlockSparseAttentionSM120Impl._sparse_ready`` for this path.

        The Impl owns that gate, and this compute callback is invoked from V2-G
        rather than through the Impl, so the gate has to be restated here or it
        is simply absent. It was absent: n6/n9/n10 ran sparse from step 0 while
        n3/n4/n5 kept the SSA-default first ten steps dense, which is worth
        about ten dense steps of attention -- roughly the whole apparent gain of
        those groups.

        Only two of the Impl's conditions can vary here. ``skip_first_layers``
        defaults to 0 and this callback has no layer identity to test it with;
        ``causal`` is false for H3 and dtype is fixed by the quantized operands.
        """
        if self.router is None or not self.layer_enabled:
            return False
        if seq_len < self.schedule.min_seq_len:
            return False
        return get_forward_context().current_timestep >= self.schedule.skip_first_steps

    def __call__(
        self,
        *,
        q_int8: torch.Tensor,
        k_int8: torch.Tensor,
        v_fp8: torch.Tensor,
        q_scale: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        out: torch.Tensor,
        used_sequence: int,
        global_sequence: int,
        softmax_scale: float,
        output_dtype: torch.dtype,
    ) -> None:
        sparse_fwd = load_bsa_attn_sm120_blk64_sage_fwd()

        fused = (
            q_int8.is_cuda
            and os.environ.get("SGLANG_DIFFUSION_H3_SAGE_FUSED_PREPOST", "1") != "0"
        )
        if fused:
            from .subblock_sparse.sage_a2a_kernels import (
                mixed_row_indices,
                pack_cake_operands,
                route_prequantized,
                static_metadata,
                write_cake_output,
            )

            q_bhsd, k_bhsd, v_bhds, q_scale_used, k_scale_used = pack_cake_operands(
                q_int8, k_int8, v_fp8, q_scale, k_scale, used_sequence
            )
        else:
            # NHD -> BHSD for Q/K, [B, D, H, S] -> [B, H, D, S] for V.  The tail
            # padding rides along in the sequence axis and is dropped here, the
            # same way the dense path slices it off.
            q_bhsd = q_int8[:, :used_sequence].transpose(1, 2).contiguous()
            k_bhsd = k_int8[:, :used_sequence].transpose(1, 2).contiguous()
            # CAKE requires exactly ceil(Sk / 64) * 64 V slots. The Ulysses
            # allocation can include additional rank-alignment padding.
            v_slots = -(-used_sequence // BLOCK) * BLOCK
            v_bhds = v_fp8[..., :v_slots].transpose(1, 2).contiguous()
            q_slots = ((used_sequence + 127) // 128) * 4
            k_slots = (used_sequence + BLOCK - 1) // BLOCK
            q_scale_used = q_scale[:, :, :q_slots].contiguous()
            k_scale_used = k_scale[:, :, :k_slots].contiguous()

        batch, heads, seq_len, _ = q_bhsd.shape
        num_blocks = -(-seq_len // BLOCK)
        if not self._route_this_step(seq_len):
            # Full density through the same kernel: same operands, same
            # arithmetic, every key block kept. That is what "dense for the
            # first N steps" means here, and it keeps one code path.
            if not fused:
                index = (
                    torch.arange(num_blocks, device=q_bhsd.device, dtype=torch.int32)
                    .view(1, 1, 1, num_blocks)
                    .expand(batch, heads, num_blocks, num_blocks)
                    .contiguous()
                )
            topk = num_blocks
        else:
            if fused:
                index, topk = route_prequantized(
                    q_int8,
                    k_int8,
                    k_scale,
                    used_sequence,
                    self.schedule.n_q,
                    self.schedule.n_k,
                    self.schedule.sparsity,
                    softmax_scale,
                )
            else:
                q_route, k_route = _dequantize_for_routing(q_bhsd, k_bhsd, k_scale_used)
                plan = self.router.route(
                    q_route.transpose(1, 2),
                    k_route.transpose(1, 2),
                    sparsity=self.schedule.sparsity,
                    softmax_scale=softmax_scale,
                )
                index, topk = plan.index, plan.topk
            # Only announce a routed step. Logging the first call instead would
            # report "keeping 592/592" from the dense warmup and read as though
            # routing had been switched off for the whole run.
            logger.info_once(
                f"SubBlock Sage sparse attention over low-precision A2A active: "
                f"S={seq_len} heads={heads} keeping {topk}/{num_blocks} key blocks "
                f"per query block (sparsity {1 - topk / num_blocks:.4f}), dense for "
                f"the first {self.schedule.skip_first_steps} denoise steps"
            )

        block_counts = None
        if self._route_this_step(seq_len) and self.query_block_mask is not None:
            if fused:
                index, block_counts = mixed_row_indices(
                    index, self.query_block_mask, num_blocks
                )
                topk = num_blocks
            else:
                mask = self.query_block_mask.to(
                    device=q_bhsd.device, dtype=torch.bool
                ).view(-1)
                if mask.numel() != num_blocks:
                    raise ValueError(
                        "H3 sparse query-block mask does not match the live sequence"
                    )
                dense_index = torch.arange(
                    num_blocks, device=q_bhsd.device, dtype=torch.int32
                )
                dense_index = (
                    dense_index.view(1, 1, 1, -1)
                    .expand(batch, heads, num_blocks, -1)
                    .clone()
                )
                dense_index[..., :topk] = torch.where(
                    mask.view(1, 1, -1, 1), index, dense_index[..., :topk]
                )
                block_counts = (
                    torch.where(mask.view(1, 1, -1), topk, num_blocks)
                    .expand(batch, heads, num_blocks)
                    .to(torch.int32)
                    .contiguous()
                )
                index, topk = dense_index, num_blocks

        if fused:
            block_sizes, dense_index = static_metadata(
                batch, heads, used_sequence, q_bhsd.device
            )
            if not self._route_this_step(seq_len):
                index = dense_index
        else:
            start = (
                torch.arange(num_blocks, device=q_bhsd.device, dtype=torch.int32)
                * BLOCK
            )
            block_sizes = (seq_len - start).clamp(0, BLOCK).to(torch.int32)
        attn = sparse_fwd(
            q_bhsd,
            k_bhsd,
            v_bhds,
            q_scale_used,
            k_scale_used,
            v_scale,
            index,
            topk,
            block_sizes=block_sizes,
            q2k_block_nums=block_counts,
            softmax_scale=softmax_scale,
            out=torch.empty_like(q_bhsd, dtype=torch.bfloat16),
            uniform_block_count=block_counts is None,
            contiguous_block_indices=False,
            backend="cake",
        )
        if fused:
            write_cake_output(attn, out, used_sequence, global_sequence)
        else:
            if used_sequence != global_sequence:
                out.zero_()
            out[:, :used_sequence] = attn.transpose(1, 2).to(output_dtype)


def subblock_sparse_sage_a2a_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    cu_seqlens_host: tuple[int, ...] | None,
    max_seqlen: int,
    softmax_scale: float,
    ulysses_active: bool,
    ring_active: bool,
    transport: object | None = None,
    query_block_mask: torch.Tensor | None = None,
    layer_enabled: bool = True,
) -> torch.Tensor | None:
    """FlashInfer Sage2 quantization + A2A followed by CAKE SSA compute.

    Geometry guards run before preparation. FlashInfer collectively validates
    kernels/configuration during prepare_qkv; errors after that point propagate
    rather than letting one rank fall back while peers enter a collective.
    """
    from sglang.multimodal_gen.runtime.distributed.parallel_state import get_sp_group
    from sglang.multimodal_gen.runtime.layers.attention.backends.sage_ulysses_route import (
        check_v2g_route,
    )

    decision = check_v2g_route(
        query,
        key,
        value,
        cu_seqlens_host=cu_seqlens_host,
        max_seqlen=max_seqlen,
        ulysses_active=ulysses_active,
        ring_active=ring_active,
        # FlashInfer's collective prepare_qkv replaces the legacy external
        # SageAttention ABI/fingerprint checks. No SageAttention fork is needed.
        capability_check=lambda device: True,
        group_agreement_check=lambda device: True,
    )
    if not decision.enabled:
        logger.info_once(
            f"FlashInfer Sage QKV fell back before exchange: {decision.reason}"
        )
        return None
    if transport is None:
        transport = _nccl_sage_transport(
            get_sp_group().ulysses_group, tuple(query.shape), query.dtype, query.device
        )
    used = int(cu_seqlens_host[1])
    encoded = transport.exchange_sage_qkv(
        query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0), used_sequence=used
    )
    out = torch.empty_like(encoded.q, dtype=query.dtype)
    _SparseSageA2ACompute(
        SubBlockSparseScheduleSM120.from_server_args(), query_block_mask, layer_enabled
    )(
        q_int8=encoded.q,
        k_int8=encoded.k,
        v_fp8=encoded.v,
        q_scale=encoded.q_scale,
        k_scale=encoded.k_scale,
        v_scale=encoded.v_scale,
        out=out,
        used_sequence=encoded.used_sequence,
        global_sequence=encoded.logical_sequence,
        softmax_scale=softmax_scale,
        output_dtype=query.dtype,
    )
    logger.info_once(
        f"MiniMax-H3 FlashInfer Sage QKV + CAKE SSA active: input a2a={transport.backend}"
    )
    return out[0]


@functools.lru_cache(maxsize=4)
def _nccl_sage_transport(group, shape, dtype, device):
    """NCCL has no registered native resources; cache per-shape QKV storage."""
    import torch.distributed as dist
    from flashinfer.comm import UlyssesCommunicator
    from sglang.multimodal_gen.runtime.distributed.device_communicators.flashinfer_ulysses_a2a import (
        FlashInferUlyssesA2A,
    )

    capacity = 3 * torch.Size(shape).numel()
    comm = UlyssesCommunicator(
        group,
        max_bytes=capacity * dtype.itemsize,
        dtype=dtype,
        device=device,
        backend="nccl",
    )
    return FlashInferUlyssesA2A(
        comm, dist.get_world_size(group), max_shapes=1, max_elems=capacity, strict=True
    )


class SubBlockSparseSageA2AAttentionSM120Backend(
    SubBlockSparseSageAttentionSM120Backend
):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [128]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.SUBBLOCK_SPARSE_ATTN_SM120_SAGE_A2A

    @staticmethod
    def get_impl_cls() -> type[SubBlockSparseSageAttentionSM120Impl]:
        # The BF16 fallback keeps the sparse compute: if the low-precision A2A
        # declines, this arm should degrade to the standalone sparse-sage
        # backend, not to dense attention.
        return SubBlockSparseSageAttentionSM120Impl


__all__ = [
    "SubBlockSparseSageA2AAttentionSM120Backend",
    "subblock_sparse_sage_a2a_forward",
]
