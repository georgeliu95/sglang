"""CPU contract tests for the CAKE adapter, without compiling a CUDA kernel."""

from types import SimpleNamespace

import pytest
import torch
from sglang.multimodal_gen.runtime.layers.attention.backends import (
    subblock_sparse_attn_sm120_sage_a2a as adapter,
)


@pytest.mark.parametrize("used", [64, 130])
@pytest.mark.parametrize("step", [0, 10])
@pytest.mark.parametrize("heterogeneous", [False, True])
def test_cake_receives_exact_tail_and_noncontiguous_routing(
    monkeypatch, used, step, heterogeneous
):
    global_sequence = 256
    blocks = (used + 63) // 64
    topk = max(1, blocks - 1)
    index = torch.arange(blocks, dtype=torch.int32)[::2][:topk]
    topk = index.numel()
    index = index.view(1, 1, 1, topk).expand(1, 2, blocks, topk).contiguous()
    monkeypatch.setattr(
        adapter,
        "SubBlockRouter",
        lambda **kwargs: SimpleNamespace(
            route=lambda *args, **kwargs: SimpleNamespace(index=index, topk=topk)
        ),
    )
    monkeypatch.setattr(
        adapter, "get_forward_context", lambda: SimpleNamespace(current_timestep=step)
    )
    calls = []
    mask = torch.arange(blocks) % 2 == 0 if heterogeneous else None

    def kernel(q, k, v, qs, ks, vs, indices, count, **kwargs):
        assert q.shape == k.shape == (1, 2, used, 128)
        assert v.shape == (1, 2, 128, blocks * 64)
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
        assert qs.shape == (1, 2, ((used + 127) // 128) * 4)
        assert ks.shape == (1, 2, blocks)
        assert kwargs["backend"] == "cake"
        assert kwargs["uniform_block_count"] is not (heterogeneous and step >= 10)
        assert kwargs["contiguous_block_indices"] is False
        if heterogeneous and step >= 10:
            counts = kwargs["q2k_block_nums"]
            assert counts.shape == (1, 2, blocks)
            assert counts[0, 0].tolist() == [
                topk if selected else blocks for selected in mask.tolist()
            ]
            assert indices[0, 0, 0, :topk].tolist() == index[0, 0, 0].tolist()
            if blocks > 1:
                assert indices[0, 0, 1].tolist() == list(range(blocks))
        else:
            assert kwargs["q2k_block_nums"] is None
        assert kwargs["block_sizes"].tolist() == [64] * (blocks - 1) + [
            used - 64 * (blocks - 1)
        ]
        out = kwargs["out"]
        assert (
            out.shape == q.shape and out.dtype == torch.bfloat16 and out.is_contiguous()
        )
        calls.append((indices, count))
        out.fill_(3)
        return out

    monkeypatch.setattr(adapter, "load_bsa_attn_sm120_blk64_sage_fwd", lambda: kernel)
    schedule = SimpleNamespace(
        sparsity=0.75, n_k=1, n_q=1, min_seq_len=0, skip_first_steps=10
    )
    q = torch.zeros((1, global_sequence, 2, 128), dtype=torch.int8)
    v = torch.zeros((1, 128, 2, global_sequence), dtype=torch.float8_e4m3fn)
    out = torch.full(q.shape, 99, dtype=torch.bfloat16)
    adapter._SparseSageA2ACompute(schedule, mask)(
        q_int8=q,
        k_int8=q,
        v_fp8=v,
        q_scale=torch.ones((1, 2, 8)),
        k_scale=torch.ones((1, 2, 4)),
        v_scale=torch.ones((1, 2, 128)),
        out=out,
        used_sequence=used,
        global_sequence=global_sequence,
        softmax_scale=128**-0.5,
        output_dtype=torch.bfloat16,
    )
    assert torch.all(out[:, :used] == 3)
    assert torch.count_nonzero(out[:, used:]) == 0
    indices, count = calls[0]
    if heterogeneous and step >= 10:
        assert count == blocks
    elif step < 10:
        assert count == blocks
        assert indices[0, 0, 0].tolist() == list(range(blocks))
    else:
        assert count == topk and indices is index


@pytest.mark.parametrize("eligible", [True, False])
def test_forward_reuses_flashinfer_prequantized_operands(monkeypatch, eligible):
    from sglang.multimodal_gen.runtime.layers.attention.backends import (
        sage_ulysses_route as routing,
    )

    def check_route(
        q,
        k,
        v,
        *,
        cu_seqlens_host,
        max_seqlen,
        ulysses_active,
        ring_active,
        capability_check,
        group_agreement_check,
    ):
        assert cu_seqlens_host == (0, 130, 256)
        return SimpleNamespace(enabled=eligible, reason="test")

    monkeypatch.setattr(routing, "check_v2g_route", check_route)
    monkeypatch.setattr(
        adapter,
        "SubBlockSparseScheduleSM120",
        SimpleNamespace(from_server_args=lambda: "schedule"),
    )
    q = torch.zeros((64, 8, 128), dtype=torch.bfloat16)
    encoded = SimpleNamespace(
        q=torch.zeros((1, 256, 2, 128), dtype=torch.int8),
        k=torch.zeros((1, 256, 2, 128), dtype=torch.int8),
        v=torch.zeros((1, 128, 2, 256), dtype=torch.float8_e4m3fn),
        q_scale=torch.ones((1, 2, 8)),
        k_scale=torch.ones((1, 2, 3)),
        v_scale=torch.ones((1, 2, 128)),
        used_sequence=130,
        logical_sequence=256,
    )
    calls = []

    def exchange(q_in, k_in, v_in, *, used_sequence):
        assert eligible and q_in.shape == (1, 64, 8, 128) and used_sequence == 130
        calls.append("quantize+exchange")
        return encoded

    def compute(**kwargs):
        assert kwargs["q_int8"] is encoded.q
        assert kwargs["k_int8"] is encoded.k
        assert kwargs["v_fp8"] is encoded.v
        assert kwargs["q_scale"] is encoded.q_scale
        assert kwargs["k_scale"] is encoded.k_scale
        assert kwargs["v_scale"] is encoded.v_scale
        assert kwargs["used_sequence"] == 130 and kwargs["global_sequence"] == 256
        calls.append("compute")
        kwargs["out"].fill_(5)

    monkeypatch.setattr(
        adapter, "_SparseSageA2ACompute", lambda schedule, *args: compute
    )
    transport = SimpleNamespace(exchange_sage_qkv=exchange, backend="pcie")
    result = adapter.subblock_sparse_sage_a2a_forward(
        q,
        q,
        q,
        cu_seqlens_host=(0, 130, 256),
        max_seqlen=130,
        softmax_scale=128**-0.5,
        ulysses_active=True,
        ring_active=False,
        transport=transport,
    )
    if eligible:
        assert calls == ["quantize+exchange", "compute"]
        assert result.shape == (256, 2, 128) and torch.all(result == 5)
    else:
        assert calls == [] and result is None
