# SPDX-License-Identifier: Apache-2.0
"""Sage uses one quantized compute path on dense and sparse schedule steps."""

from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.attention.backends import (
    subblock_sparse_attn as base,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse_attn import (
    SubBlockSparseSchedule,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse_attn_sm120_sage import (
    SubBlockSparseSageAttentionSM120Impl,
)


def sage_impl() -> SubBlockSparseSageAttentionSM120Impl:
    impl = SubBlockSparseSageAttentionSM120Impl.__new__(
        SubBlockSparseSageAttentionSM120Impl
    )
    impl.schedule = SubBlockSparseSchedule(0.75, 10, 0, 4, 4, 64, "bf16")
    impl.layer_idx = 0
    impl.layer_enabled = True
    impl.head_size = 128
    impl.causal = False
    return impl


@pytest.mark.parametrize("step", [0, 9, 10])
def test_video_selects_quantized_compute_on_dense_and_sparse_steps(
    monkeypatch: pytest.MonkeyPatch, step: int
) -> None:
    # Given an eligible DiT video segment and the real scheduling state.
    impl = sage_impl()
    q = torch.zeros((1, 64, 2, 128), dtype=torch.bfloat16)
    monkeypatch.setattr(
        base, "get_forward_context", lambda: SimpleNamespace(current_timestep=step)
    )
    # When the single-card adapter selects its compute path.
    ready = impl._sparse_ready(q, q)
    # Then early full-density steps use CAKE as well as later sparse steps.
    assert ready


@pytest.mark.parametrize("step", [0, 10])
@pytest.mark.parametrize(
    "causal,layer_idx,length,dtype",
    [
        (True, 0, 64, torch.bfloat16),
        (False, None, 64, torch.bfloat16),
        (False, 0, 32, torch.bfloat16),
        (False, 0, 64, torch.float16),
    ],
)
def test_ineligible_segments_keep_dense_fallback(
    monkeypatch: pytest.MonkeyPatch,
    step: int,
    causal: bool,
    layer_idx: int | None,
    length: int,
    dtype: torch.dtype,
) -> None:
    # Given a segment outside the Sage compute contract.
    impl = sage_impl()
    impl.causal = causal
    impl.layer_idx = layer_idx
    impl.layer_enabled = layer_idx is not None
    q = torch.zeros((1, length, 2, 128), dtype=dtype)
    monkeypatch.setattr(
        base, "get_forward_context", lambda: SimpleNamespace(current_timestep=step)
    )
    # When the adapter selects its compute path.
    ready = impl._sparse_ready(q, q)
    # Then it retains the existing dense fallback.
    assert not ready


@pytest.mark.parametrize("step,expected", [(0, False), (9, False), (10, True)])
def test_compute_preserves_dense_to_sparse_schedule_boundary(
    monkeypatch: pytest.MonkeyPatch, step: int, expected: bool
) -> None:
    from sglang.multimodal_gen.runtime.layers.attention.backends import (
        subblock_sparse_attn_sm120_sage_a2a as adapter,
    )

    # Given the compute callback shared by local Sage and prepared multi-GPU QKV.
    schedule = SubBlockSparseSchedule(0.75, 10, 0, 4, 4, 64, "bf16")
    compute = adapter._SparseSageA2ACompute(schedule)
    monkeypatch.setattr(
        adapter, "get_forward_context", lambda: SimpleNamespace(current_timestep=step)
    )
    # When a video segment asks whether to remove key blocks.
    routed = compute._route_this_step(64)
    # Then first ten steps stay full-density, and step ten enables SSA.
    assert routed is expected


@pytest.mark.parametrize("step", [0, 10])
def test_disabled_layer_keeps_fallback(
    monkeypatch: pytest.MonkeyPatch, step: int
) -> None:
    # Given a DiT layer excluded by skip_first_layers.
    impl = sage_impl()
    impl.layer_enabled = False
    q = torch.zeros((1, 64, 2, 128), dtype=torch.bfloat16)
    monkeypatch.setattr(
        base, "get_forward_context", lambda: SimpleNamespace(current_timestep=step)
    )
    # When selecting compute, then preserve the configured fallback.
    assert not impl._sparse_ready(q, q)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("step", [0, 9, 10])
def test_single_card_quantized_attention_executes_with_dense_schedule(
    step: int,
) -> None:
    from sglang.multimodal_gen.runtime.managers.forward_context import (
        set_forward_context,
    )

    # Given a non-aligned video document using the public single-card adapter.
    torch.manual_seed(1101)
    impl = sage_impl()
    impl.softmax_scale = 128**-0.5
    length = 2053
    q, k, v = (
        torch.randn((length, 2, 128), device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    )
    cu = torch.tensor([0, length, length], device="cuda", dtype=torch.int32)
    # When the scheduled attention executes the real quantizer and CAKE kernel.
    with torch.inference_mode(), set_forward_context(step, None):
        out = impl.forward_varlen(
            q,
            k,
            v,
            cu_seqlens=cu,
            cu_seqlens_host=(0, length, length),
            max_seqlen=length,
        )
        reference = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
        )[0].transpose(0, 1)
    # Then dense-scheduled steps approximate full attention, not sparse routing.
    assert torch.isfinite(out).all()
    relative_l2 = (out.float() - reference.float()).norm() / reference.float().norm()
    if step < 10:
        assert relative_l2.item() < 0.05
    else:
        assert relative_l2.item() > 0.05
