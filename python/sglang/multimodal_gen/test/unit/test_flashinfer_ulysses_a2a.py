# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from sglang.multimodal_gen.runtime.distributed.device_communicators.flashinfer_ulysses_a2a import (
    FlashInferUlyssesA2A,
    _TRANSPORTS,
    get_flashinfer_ulysses_a2a,
    reset_flashinfer_ulysses_request_stats,
    validate_flashinfer_ulysses_request,
)


class _FakeCommunicator:
    backend = "pcie"
    transport = "hybrid"
    device = torch.device("cpu")
    dtype = torch.float16

    def __init__(self, world_size: int = 2):
        self.world_size = world_size
        self.allocations = []
        self.exchanges = 0
        self.closed = False

    def allocate_output(self, sample: torch.Tensor, op: str) -> torch.Tensor:
        self.allocations.append((op, tuple(sample.shape)))
        batch, tokens, heads, head_dim = sample.shape
        if op == "scatter_heads":
            shape = (
                batch,
                tokens * self.world_size,
                heads // self.world_size,
                head_dim,
            )
        else:
            shape = (
                batch,
                tokens // self.world_size,
                heads * self.world_size,
                head_dim,
            )
        return torch.empty(shape, dtype=sample.dtype)

    def scatter_heads(self, source: torch.Tensor, *, out: torch.Tensor):
        self.exchanges += 1
        return out

    def gather_heads(self, source: torch.Tensor, *, out: torch.Tensor):
        self.exchanges += 1
        return out

    def close(self):
        self.closed = True


def _transport(*, strict: bool = True, max_shapes: int = 4, max_elems: int = 4096):
    comm = _FakeCommunicator()
    return (
        FlashInferUlyssesA2A(
            comm,
            world_size=2,
            max_shapes=max_shapes,
            max_elems=max_elems,
            strict=strict,
        ),
        comm,
    )


def test_prepare_geometry_registers_pair_once_and_reuses_qkv_buffer():
    transport, comm = _transport()

    transport.prepare_geometry(4, 8, 4, torch.float16)
    transport.prepare_geometry(4, 8, 4, torch.float16)
    qkv = transport.qkv_buffer(4, 8, 4, torch.float16)

    assert qkv is not None
    assert tuple(qkv.shape) == (4, 8, 3, 4)
    assert qkv.data_ptr() == transport.qkv_buffer(4, 8, 4, torch.float16).data_ptr()
    assert comm.allocations == [
        ("scatter_heads", (1, 4, 24, 4)),
        ("gather_heads", (1, 8, 4, 4)),
    ]
    assert transport.stats()["registered_geometries"] == 1


def test_packed_geometry_registers_the_quantized_input_side_instead():
    """A caller that quantizes first must not also hold the BF16 scatter output.

    That output is nearly twice the packed one and lives for the process, so
    registering both would give up most of what the smaller payload buys.
    """
    transport, comm = _transport()
    record = 3 * 4 + 16

    transport.prepare_geometry(4, 8, 4, torch.float16, packed_record_bytes=record)

    assert comm.allocations == [
        ("scatter_heads", (1, 4, 8, record // 2)),
        ("gather_heads", (1, 8, 4, 4)),
    ]
    assert transport.stats()["registered_geometries"] == 1


def test_exchange_packed_input_carries_bytes_over_the_bf16_communicator():
    transport, comm = _transport()
    record = 3 * 4 + 16
    transport.prepare_geometry(4, 8, 4, torch.float16, packed_record_bytes=record)

    payload = torch.zeros((1, 4, 8, record), dtype=torch.uint8)
    out = transport.exchange_packed_input(payload)

    # Bytes in, bytes out: the head axis is split and the token axis grows by
    # the world size, exactly as the BF16 scatter does.
    assert out.dtype == torch.uint8
    assert tuple(out.shape) == (1, 8, 4, record)
    assert comm.exchanges == 1
    assert transport.stats()["sglang_exchanges"] == 1
    # The registration was reused, not repeated.
    assert len(comm.allocations) == 2


def _construct_with_route(monkeypatch, route: str):
    """Build the transport at eight ranks over a communicator on `route`."""
    import flashinfer.comm as flashinfer_comm

    class _RoutedCommunicator(_FakeCommunicator):
        transport = route

        def __init__(self, group, **kwargs):
            super().__init__(world_size=8)
            self.device = kwargs["device"]

    monkeypatch.setattr(flashinfer_comm, "UlyssesCommunicator", _RoutedCommunicator)
    monkeypatch.setattr(dist, "get_world_size", lambda group: 8)
    monkeypatch.setattr(dist, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    _TRANSPORTS.clear()
    with torch.inference_mode():
        return get_flashinfer_ulysses_a2a(
            SimpleNamespace(group_name=f"route-{route}"),
            torch.device("cpu"),
            torch.float16,
            capacity=4096,
            strict=True,
            max_shapes=4,
        )


@pytest.mark.parametrize("route", ["hybrid", "rdma"])
def test_strict_accepts_either_nic_route_at_eight_ranks(monkeypatch, route):
    monkeypatch.delenv("FLASHINFER_ULYSSES_PCIE_ROUTE", raising=False)
    # Which of the two a topology resolves to is FlashInfer's decision: "hybrid"
    # needs a 4+4 NUMA split, "rdma" covers the rest. Both reach peers over the
    # NIC, which is the property strict mode is guarding.
    try:
        assert _construct_with_route(monkeypatch, route) is not None
    finally:
        _TRANSPORTS.clear()


def test_strict_rejects_p2p_route_at_eight_ranks(monkeypatch):
    monkeypatch.delenv("FLASHINFER_ULYSSES_PCIE_ROUTE", raising=False)
    # Eight ranks do not fit in one NUMA node, so all-P2P here means the NIC was
    # unusable -- and it measures slower than NCCL. Fail instead of quietly
    # taking it.
    try:
        with pytest.raises(RuntimeError, match="expected one of"):
            _construct_with_route(monkeypatch, "p2p")
    finally:
        _TRANSPORTS.clear()


@pytest.mark.parametrize("requested", ["p2p", "rdma", "hybrid"])
def test_strict_honors_explicit_route(monkeypatch, requested):
    monkeypatch.setenv("FLASHINFER_ULYSSES_PCIE_ROUTE", requested)
    try:
        assert _construct_with_route(monkeypatch, requested) is not None
        wrong = "rdma" if requested == "p2p" else "p2p"
        with pytest.raises(RuntimeError, match="expected one of"):
            _construct_with_route(monkeypatch, wrong)
    finally:
        _TRANSPORTS.clear()


def test_constructor_builds_transport_once(monkeypatch):
    import flashinfer.comm as flashinfer_comm

    captured = {}

    class _FakeNativeCommunicator(_FakeCommunicator):
        def __init__(self, group, **kwargs):
            super().__init__(world_size=2)
            self.device = kwargs["device"]
            captured.update(kwargs)

    monkeypatch.setattr(flashinfer_comm, "UlyssesCommunicator", _FakeNativeCommunicator)
    monkeypatch.setattr(dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(dist, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    group = SimpleNamespace(group_name="constructor-once")
    _TRANSPORTS.clear()
    try:
        with torch.inference_mode():
            transport = get_flashinfer_ulysses_a2a(
                group,
                torch.device("cpu"),
                torch.float16,
                capacity=4096,
                strict=True,
                max_shapes=4,
            )
            cached = get_flashinfer_ulysses_a2a(
                group,
                torch.device("cpu"),
                torch.float16,
                capacity=4096,
                strict=True,
                max_shapes=4,
            )
        assert transport is not None
        assert captured["backend"] == "pcie"
        assert captured["max_bytes"] == 4096 * torch.float16.itemsize
        assert cached is transport
    finally:
        _TRANSPORTS.clear()


def test_exchange_pair_is_counted_and_preserves_contract_shapes():
    transport, comm = _transport()
    transport.prepare_geometry(4, 8, 4, torch.float16)

    qkv = transport.qkv_buffer(4, 8, 4, torch.float16)
    q, k, v = transport.exchange_input(qkv)
    gathered = transport.exchange_output(torch.empty((8, 4, 4), dtype=torch.float16))

    assert tuple(q.shape) == tuple(k.shape) == tuple(v.shape) == (8, 4, 4)
    assert tuple(gathered.shape) == (4, 8, 4)
    assert transport.stats()["sglang_exchanges"] == 2
    assert comm.exchanges == 2


def test_strict_mode_rejects_capacity_and_geometry_budget_overflow():
    transport, _ = _transport(max_shapes=1, max_elems=383)

    try:
        transport.prepare_geometry(4, 8, 4, torch.float16)
    except RuntimeError as error:
        assert "declared capacity" in str(error)
    else:
        raise AssertionError("strict capacity overflow must fail")

    transport, _ = _transport(max_shapes=1)
    transport.prepare_geometry(4, 8, 4, torch.float16)
    try:
        transport.prepare_geometry(2, 8, 4, torch.float16)
    except RuntimeError as error:
        assert "geometry budget" in str(error)
    else:
        raise AssertionError("strict geometry budget overflow must fail")


def test_stats_reset_and_collective_shutdown():
    transport, comm = _transport()
    transport.prepare_geometry(4, 8, 4, torch.float16)
    transport.exchange_input(transport.qkv_buffer(4, 8, 4, torch.float16))

    assert transport.stats(reset=True)["sglang_exchanges"] == 1
    assert transport.stats()["sglang_exchanges"] == 0
    transport.shutdown()
    assert comm.closed is True


def test_request_interval_requires_exact_exchange_delta():
    transport, _ = _transport()
    _TRANSPORTS["test"] = transport
    try:
        transport._exchanges = 17
        reset_flashinfer_ulysses_request_stats()
        transport._exchanges = 20
        assert (
            validate_flashinfer_ulysses_request(20, strict=True)[0]["sglang_exchanges"]
            == 20
        )
        try:
            validate_flashinfer_ulysses_request(19, strict=True)
        except RuntimeError as error:
            assert "expected 19" in str(error)
        else:
            raise AssertionError("strict request exchange mismatch must fail")
    finally:
        _TRANSPORTS.pop("test")


def test_prepared_sage_qkv_reuses_workspace_and_encoded_output():
    transport, comm = _transport(max_shapes=1)
    prepared = []
    outputs = []
    encoded = SimpleNamespace(q=torch.empty((1, 8, 4, 4), dtype=torch.int8))

    def prepare(shape, **kwargs):
        prepared.append((tuple(shape), kwargs))
        return "workspace"

    def scatter(q, k, v, *, workspace, out):
        assert workspace == "workspace"
        outputs.append(out)
        return encoded

    comm.prepare_qkv = prepare
    comm.scatter_qkv = scatter
    q = torch.zeros((1, 4, 8, 4), dtype=torch.float16)
    transport.prepare_geometry(4, 8, 4, q.dtype, sage_qkv=True)
    assert [op for op, shape in comm.allocations] == ["gather_heads"]
    for _ in range(2):
        assert transport.exchange_sage_qkv(q, q, q, used_sequence=7) is encoded
    assert prepared == [(tuple(q.shape), {"quantization": "sage2", "used_sequence": 7})]
    assert outputs == [None, encoded]
    assert transport.stats()["sglang_exchanges"] == 2
    with pytest.raises(RuntimeError, match="geometry budget"):
        transport.exchange_sage_qkv(q, q, q, used_sequence=8)
