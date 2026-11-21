"""CPU contract tests for the CAKE adapter, without compiling a CUDA kernel."""

from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT,
)
from sglang.multimodal_gen.runtime.layers.attention.backends import (
    sage_ulysses_route as routing,
)
from sglang.multimodal_gen.runtime.layers.attention.backends import (
    subblock_sparse_attn_sm120_sage_a2a as adapter,
)

# Live packed rows of the 768p brass-band request, as the 4-Spark run logged them.
_H3_768P_USED = 37765
_H3_HEADS = 56


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


@pytest.fixture
def h3_768p_route(monkeypatch):
    """Run check_v2g_route on one rank's 768p H3 shard; CUDA and the group are faked."""
    monkeypatch.delenv("SGLANG_DIFFUSION_DISABLE_SM120_LOWP_A2A_V2G", raising=False)
    monkeypatch.setattr(
        routing.torch.cuda, "is_current_stream_capturing", lambda: False
    )
    monkeypatch.setattr(routing.dist, "get_world_size", lambda group: group.size)

    def route(
        *,
        world,
        capability=(12, 1),
        gb10_delivery=True,
        heads=_H3_HEADS,
        group_world=None,
    ):
        if gb10_delivery:
            monkeypatch.setenv("SGLANG_H3_GB10_DELIVERY", "1")
        else:
            monkeypatch.delenv("SGLANG_H3_GB10_DELIVERY", raising=False)
        monkeypatch.setattr(
            routing.torch.cuda, "get_device_capability", lambda device: capability
        )
        monkeypatch.setattr(routing, "get_ulysses_parallel_world_size", lambda: world)
        group = SimpleNamespace(size=world if group_world is None else group_world)
        monkeypatch.setattr(
            routing, "get_sp_group", lambda: SimpleNamespace(ulysses_group=group)
        )
        alignment = MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT
        padded = -(-_H3_768P_USED // alignment) * alignment
        assert padded % world == 0, "the packed H3 sequence must split evenly"
        shard = SimpleNamespace(
            ndim=3,
            shape=(padded // world, heads, 128),
            dtype=torch.bfloat16,
            device="cuda:0",
            is_cuda=True,
            requires_grad=False,
        )
        with torch.no_grad():
            return routing.check_v2g_route(
                shard,
                shard,
                shard,
                cu_seqlens_host=(0, _H3_768P_USED, padded),
                max_seqlen=_H3_768P_USED,
                ulysses_active=True,
                ring_active=False,
            )

    return route


@pytest.mark.parametrize("world", [2, 4, 8])
def test_gb10_route_admits_h3_768p_shards_up_to_eight_sparks(h3_768p_route, world):
    decision = h3_768p_route(world=world)

    assert decision.enabled, decision.reason


@pytest.mark.parametrize(("world", "heads"), [(3, 48), (6, 48), (16, 48), (8, 60)])
def test_gb10_route_declines_geometries_flashinfer_sage2_rejects(
    h3_768p_route, world, heads
):
    """FlashInfer's Sage2 prepare_qkv takes only 2/4/8 ranks with heads % world == 0
    and fails collectively; the route has to decline before the exchange instead."""
    decision = h3_768p_route(world=world, heads=heads)

    assert decision.reason == "unsupported Ulysses geometry"


@pytest.mark.parametrize(
    ("capability", "gb10_delivery", "world", "enabled"),
    [
        ((12, 1), False, 8, False),
        ((12, 0), True, 2, False),
        ((12, 0), False, 8, True),
    ],
)
def test_gb10_world_list_needs_both_the_delivery_switch_and_sm121(
    h3_768p_route, capability, gb10_delivery, world, enabled
):
    decision = h3_768p_route(
        world=world, capability=capability, gb10_delivery=gb10_delivery
    )

    assert decision.enabled is enabled, decision.reason


def test_eight_rank_route_declines_a_ulysses_group_of_another_size(h3_768p_route):
    """The NCCL Sage transport derives its rank count from the Ulysses group; a group
    that disagrees with the Ulysses world would scatter the wrong head shards."""
    decision = h3_768p_route(world=8, group_world=4)

    assert decision.reason == "process group mismatch"
