# SPDX-License-Identifier: Apache-2.0
"""SM120 CAKE SSA using the native H3 schedule and modality-aware routing."""

import torch
from sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse.sage_sm120 import (
    load_quantize_sage_qkv_sm120,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse_attn import (
    SubBlockSparseAttentionBackend,
    SubBlockSparseAttentionImpl,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


class SubBlockSparseSageAttentionSM120Backend(SubBlockSparseAttentionBackend):
    @staticmethod
    def get_enum():
        return AttentionBackendEnum.SUBBLOCK_SPARSE_ATTN_SM120_SAGE

    @staticmethod
    def get_impl_cls():
        return SubBlockSparseSageAttentionSM120Impl


class SubBlockSparseSageAttentionSM120Impl(SubBlockSparseAttentionImpl):
    def _sparse_attention(self, q, k, v, sparse_query_block_mask=None):
        from .subblock_sparse_attn_sm120_sage_a2a import _SparseSageA2ACompute

        operands = load_quantize_sage_qkv_sm120()(
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
        )
        qi, ki, vf, qs, ks, vs = operands
        out = torch.empty_like(q)
        _SparseSageA2ACompute(
            self.schedule, sparse_query_block_mask, self.layer_enabled
        )(
            q_int8=qi.transpose(1, 2).contiguous(),
            k_int8=ki.transpose(1, 2).contiguous(),
            v_fp8=vf.transpose(1, 2).contiguous(),
            q_scale=qs,
            k_scale=ks,
            v_scale=vs,
            out=out,
            used_sequence=q.shape[1],
            global_sequence=q.shape[1],
            softmax_scale=self.softmax_scale,
            output_dtype=q.dtype,
        )
        return out
