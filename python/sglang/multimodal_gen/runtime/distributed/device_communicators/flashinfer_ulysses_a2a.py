# SPDX-License-Identifier: Apache-2.0
"""FlashInfer PCIe transport for the MiniMax-H3 Ulysses exchange.

On a single node without NVLink the exchange is link-bound. FlashInfer's PCIe
Ulysses backend moves the payload with copy engines rather than SM stores, and
at world size 8 it routes the cross-NUMA half over each rank's local mlx5 NIC
while the same-NUMA half stays on CUDA P2P.

The adapter exposes ``qkv_buffer`` / ``exchange_input`` / ``exchange_output``
to the attention core and pre-registers each scatter/gather pair before the
projection for that sequence geometry begins.

Q/K/V travel as one operand. The projection already emits
``[tokens, heads, 3, head_dim]``, so the head axis carries ``3 * H`` entries and
splitting it across ranks hands each one ``H/ws`` heads with their own Q, K and
V beside them. Nothing is packed going in and nothing is copied coming out, and
a layer costs two exchanges rather than four.

``qkv_buffer`` hands the projection a persistent destination reused across
layers, sparing a fresh QKV allocation per layer. The transport stages both
directions itself: FlashInfer owns the buffers the NIC touches, and a caller
operand is copied there inside the exchange.

That last part is the point, and it is not what an isolated benchmark would
tell you: per byte, this backend is no faster fused than unfused. But its
NIC-bearing routes block the calling thread inside every exchange, so the number
of exchanges is a cost of its own, separate from what each one moves.

Registration is collective and permanent -- constructing the communicator and
allocating an output both need every rank -- so every gate below is a function
of values identical on every rank.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.nvtx_pytorch_hooks import (
    maybe_nvtx_range,
    ulysses_nvtx_enabled,
)

logger = init_logger(__name__)

_SUPPORTED_WORLD_SIZES = (2, 4, 8)
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)

# The PCIe backend's routes that reach a peer over the NIC: "hybrid" groups
# ranks by NUMA node and keeps same-node peers on CUDA P2P, "rdma" puts every
# rank in its own group and sends everything over the NIC. Which one a topology
# resolves to is FlashInfer's decision; both are acceptable here and the
# remaining route, all-P2P, is not. FlashInfer keeps the same list private, so
# it is restated rather than imported -- this module depends on its public
# surface only.
_RDMA_ROUTES = ("hybrid", "rdma")

_TRANSPORTS: dict[str, "FlashInferUlyssesA2A | None"] = {}


def _unusable(world_size: int, dtype: torch.dtype) -> str | None:
    """Why this exchange cannot use the PCIe backend, or None if it can.

    Only conditions that exist before a group does, and only ones every rank
    evaluates identically.
    """
    try:
        import flashinfer.comm  # noqa: F401
    except ImportError as error:
        return f"flashinfer is not installed ({error})"
    if world_size not in _SUPPORTED_WORLD_SIZES:
        return f"world size {world_size} is not one of {_SUPPORTED_WORLD_SIZES}"
    if dtype not in _SUPPORTED_DTYPES:
        return f"dtype {dtype} is unsupported"
    if torch.is_grad_enabled():
        return "autograd is enabled"
    if torch.compiler.is_compiling():
        return "running under torch.compile"
    if torch.cuda.is_current_stream_capturing():
        return "running under CUDA graph capture"
    return None


class FlashInferUlyssesA2A:
    """One communicator, plus per-shape send buffers and registered outputs."""

    def __init__(
        self,
        comm,
        world_size: int,
        max_shapes: int,
        max_elems: int,
        *,
        strict: bool,
    ) -> None:
        self._comm = comm
        self.world_size = world_size
        self.device = comm.device
        # The construction dtype is what every operand and registered output
        # must carry; a packed payload is reinterpreted into it rather than
        # assuming what it happens to be.
        self.dtype = comm.dtype
        self._max_shapes = max_shapes
        self.max_elems = max_elems
        self._sage_qkv = {}
        self.strict = strict
        # Constructing a transport says nothing about whether anything routes
        # through it. An integration that silently keeps calling NCCL looks
        # exactly like one that works, so count what is actually carried and
        # say so on the first one and again at teardown. A benchmark run whose
        # log has no such line measured NCCL, whatever it was labelled.
        self._exchanges = 0
        # (role, source shape, dtype) -> registered exchange output
        self._outputs: dict[tuple, torch.Tensor] = {}
        # Same key -> a persistent exchange source, so the projection does not
        # allocate a fresh operand per layer.
        self._sends: dict[tuple, torch.Tensor] = {}
        self._declined: set[tuple] = set()
        self._geometries: set[tuple] = set()

    @property
    def backend(self) -> str:
        return self._comm.backend

    @property
    def transport(self) -> str | None:
        return self._comm.transport

    def _output(self, role: str, sample: torch.Tensor, op: str):
        """The registered exchange output for one role and operand shape.

        Allocated on first sight. ``allocate_output`` is collective and the key
        comes from rank-identical values, so every rank allocates the same
        outputs in the same order or none of them. The operand itself stays the
        caller's -- the transport no longer registers or tracks it, so there is
        nothing to project or stage into.
        """
        source_shape, dtype = tuple(sample.shape), sample.dtype
        key = (role, source_shape, dtype)
        cached = self._outputs.get(key)
        if cached is not None:
            return cached
        if key in self._declined:
            return None
        # A geometry registers one scatter and one gather; a quantized input
        # side swaps its scatter for a packed one rather than adding to it, so
        # two per geometry still bounds it. Count registrations against geometry
        # pairs so a budget of four really admits four request shapes.
        if len(self._outputs) >= 2 * self._max_shapes:
            self._declined.add(key)
            reason = (
                f"{self._max_shapes} sequence geometries are already registered; "
                f"cannot register {key}"
            )
            if self.strict:
                raise RuntimeError(f"flashinfer ulysses strict mode: {reason}")
            logger.info("flashinfer ulysses: %s; staying on NCCL", reason)
            return None
        if torch.Size(source_shape).numel() > self.max_elems:
            self._declined.add(key)
            reason = (
                f"{key} exceeds the declared capacity of {self.max_elems} elements; "
                "raise --minimax-h3-ulysses-max-seq-len"
            )
            if self.strict:
                raise RuntimeError(f"flashinfer ulysses strict mode: {reason}")
            logger.warning("flashinfer ulysses: %s; staying on NCCL", reason)
            return None
        try:
            out = self._comm.allocate_output(sample, op)
        except Exception as error:  # noqa: BLE001
            self._declined.add(key)
            if self.strict:
                raise RuntimeError(
                    f"flashinfer ulysses strict mode: could not register {key}"
                ) from error
            logger.warning(
                "flashinfer ulysses: could not register %s (%s); staying on NCCL "
                "for this shape",
                key,
                error,
            )
            return None
        self._outputs[key] = out
        return out

    def prepare_geometry(
        self,
        tokens: int,
        heads: int,
        head_dim: int,
        dtype: torch.dtype,
        *,
        packed_record_bytes: int | None = None,
        sage_qkv: bool = False,
    ) -> None:
        """Collectively pre-register the paired scatter/gather workspaces.

        ``packed_record_bytes`` registers the quantized input side instead of
        the BF16 one: a caller that quantizes before the exchange never issues
        the BF16 scatter, and registering it anyway would hold an output nearly
        twice the packed one's size for the whole process lifetime.
        ``sage_qkv`` registers only the output gather here; prepare_qkv owns
        the input byte-chunk registration when its live length is available.
        """
        geometry = (tokens, heads, head_dim, dtype, packed_record_bytes, sage_qkv)
        if geometry in self._geometries:
            return
        if len(self._geometries) >= self._max_shapes:
            reason = f"geometry budget {self._max_shapes} exhausted before {geometry}"
            if self.strict:
                raise RuntimeError(f"flashinfer ulysses strict mode: {reason}")
            logger.warning("flashinfer ulysses: %s; staying on NCCL", reason)
            return

        if packed_record_bytes is not None:
            packed = torch.empty(
                (1, tokens, heads, packed_record_bytes // dtype.itemsize),
                dtype=dtype,
                device=self.device,
            )
            if self._output("v2g_qkv", packed, "scatter_heads") is None:
                return
        elif not sage_qkv:
            qkv = torch.empty(
                (1, tokens, heads * 3, head_dim), dtype=dtype, device=self.device
            )
            scatter = self._output("qkv", qkv, "scatter_heads")
            if scatter is None:
                return
            # Keep the sample as this geometry's persistent projection
            # destination; the transport stages every operand on the way in, so
            # a stable source spares an allocation per layer, not a copy.
            self._sends[("qkv", tuple(qkv.shape), dtype)] = qkv

        gathered_source = torch.empty(
            (1, tokens * self.world_size, heads // self.world_size, head_dim),
            dtype=dtype,
            device=self.device,
        )
        if self._output("attn_out", gathered_source, "gather_heads") is None:
            return
        self._geometries.add(geometry)

    def exchange_sage_qkv(self, q, k, v, *, used_sequence: int):
        """Quantize before A2A using FlashInfer's prepared Sage2 QKV API.

        The byte-chunk exchange uses the communicator's PCIe/RDMA transport;
        statistics still use one NCCL AllGather. Reuse both the workspace and
        received operands across the model's serialized attention layers.
        """
        key = (tuple(q.shape), q.dtype, used_sequence)
        state = self._sage_qkv.get(key)
        if state is None:
            if len(self._sage_qkv) >= self._max_shapes:
                raise RuntimeError("FlashInfer Sage QKV geometry budget exhausted")
            workspace = self._comm.prepare_qkv(
                q.shape, quantization="sage2", used_sequence=used_sequence
            )
            encoded = None
        else:
            workspace, encoded = state
        encoded = self._comm.scatter_qkv(q, k, v, workspace=workspace, out=encoded)
        self._sage_qkv[key] = (workspace, encoded)
        self._count(q)
        return encoded

    def stats(self, reset: bool = False) -> dict[str, object]:
        stats = {
            "sglang_exchanges": self._exchanges,
            "registered_geometries": len(self._geometries),
            "geometry_budget": self._max_shapes,
            "declined_geometries": len(self._declined),
            "strict": self.strict,
        }
        if reset:
            self._exchanges = 0
        return stats

    # -------------------------------------------------------------- input --
    def qkv_buffer(self, tokens: int, heads: int, head_dim: int, dtype):
        """The projection's destination, reused across layers.

        The transport stages every scatter operand on the way in, so this
        buffer spares an allocation per layer rather than a copy. Returns None
        when the shape is not registrable, in which case the caller projects
        wherever it likes and the exchange stays on NCCL for that shape.
        """
        shape = (1, tokens, heads * 3, head_dim)
        key = ("qkv", shape, dtype)
        send = self._sends.get(key)
        if send is None:
            # allocate_output wants an operand to take its geometry from; the
            # sample then stays on as this shape's persistent source.
            sample = torch.empty(shape, dtype=dtype, device=self.device)
            if self._output("qkv", sample, "scatter_heads") is None:
                return None
            self._sends[key] = sample
            send = sample
        # The projection wants [tokens, heads, 3, head_dim] and the exchange
        # wants [1, tokens, 3 * heads, head_dim]: same memory, same order.
        return send.view(tokens, heads, 3, head_dim)

    def exchange_input(self, qkv: torch.Tensor):
        """``[T_local, H, 3, D]`` -> three ``[T_global, H/ws, D]`` views."""
        tokens, heads, three, head_dim = qkv.shape
        assert three == 3, f"expected [T, H, 3, D], got {tuple(qkv.shape)}"
        source = qkv.view(1, tokens, heads * 3, head_dim)
        out = self._output("qkv", source, "scatter_heads")
        if out is None:
            return None
        with maybe_nvtx_range("fi_a2a_scatter_heads", ulysses_nvtx_enabled()):
            exchanged = self._comm.scatter_heads(source, out=out)
        self._count(source)
        merged = exchanged.view(
            tokens * self.world_size, heads // self.world_size, 3, head_dim
        )
        return merged[:, :, 0], merged[:, :, 1], merged[:, :, 2]

    def exchange_packed_input(self, payload: torch.Tensor):
        """``[1, T_local, H, R]`` uint8 -> ``[1, T_global, H/ws, R]`` uint8.

        The scatter is a byte move, so a quantized payload rides the same
        communicator as the BF16 one: its record axis is reinterpreted as the
        communicator's own element type on the way in and back to bytes on the
        way out, which keeps both the construction dtype and every registration
        already held. A record must therefore be a whole number of those
        elements -- V2-G's is a multiple of 16 bytes for its own vector loads,
        so this costs nothing.

        The declared capacity still applies, in those same elements: the payload
        is fewer bytes than the unquantized operand, so a geometry that fits
        unquantized fits this too.
        """
        batch, tokens, heads, record = payload.shape
        element = self.dtype.itemsize
        assert payload.dtype == torch.uint8, f"expected uint8, got {payload.dtype}"
        assert record % element == 0, (
            f"record must be a multiple of the communicator element size "
            f"{element}, got {record}"
        )
        source = payload.view(self.dtype).view(batch, tokens, heads, record // element)
        out = self._output("v2g_qkv", source, "scatter_heads")
        if out is None:
            return None
        with maybe_nvtx_range("fi_a2a_scatter_heads_packed", ulysses_nvtx_enabled()):
            exchanged = self._comm.scatter_heads(source, out=out)
        self._count(source)
        return exchanged.view(torch.uint8)

    # ------------------------------------------------------------- output --
    def exchange_output(self, attn_out: torch.Tensor):
        """``[T_global, H/ws, D]`` -> ``[T_local, H, D]``, head axis merged."""
        tokens, heads, head_dim = attn_out.shape
        source = attn_out.view(1, tokens, heads, head_dim)
        out = self._output("attn_out", source, "gather_heads")
        if out is None:
            return None
        with maybe_nvtx_range("fi_a2a_gather_heads", ulysses_nvtx_enabled()):
            exchanged = self._comm.gather_heads(source, out=out)
        self._count(source)
        return exchanged.view(
            tokens // self.world_size, heads * self.world_size, head_dim
        )

    # --------------------------------------------------------------- life --
    def _count(self, source: torch.Tensor) -> None:
        self._exchanges += 1
        if self._exchanges == 1:
            logger.info(
                "flashinfer ulysses carried its first exchange "
                "(backend=%s, transport=%s, operand=%s)",
                self.backend,
                self.transport,
                tuple(source.shape),
            )

    def shutdown(self) -> None:
        """Collective: every rank must call this before any rank exits."""
        logger.info(
            "flashinfer ulysses carried %d exchanges over its lifetime",
            self._exchanges,
        )
        self._sends.clear()
        self._outputs.clear()
        self._geometries.clear()
        self._comm.close()


def get_flashinfer_ulysses_a2a(
    group,
    device: torch.device,
    dtype: torch.dtype,
    capacity: int,
    *,
    strict: bool,
    max_shapes: int,
) -> FlashInferUlyssesA2A | None:
    """The transport for this process group, constructing it once.

    ``capacity`` is the declared maximum and sizes every registration. It cannot
    be raised in place, so an operand beyond it stays on NCCL rather than
    forcing a rebuild -- which is what every other communication workspace in
    both trees does, and measurably cheaper than the alternative.
    """
    world_size = dist.get_world_size(group)
    reason = _unusable(world_size, dtype)
    name = getattr(group, "group_name", None) or str(id(group))
    if reason is not None:
        if strict:
            raise RuntimeError(f"flashinfer ulysses strict mode: {reason}")
        if name not in _TRANSPORTS:
            _TRANSPORTS[name] = None
            logger.info("flashinfer ulysses is unusable (%s); staying on NCCL", reason)
        return None

    if name in _TRANSPORTS:
        return _TRANSPORTS[name]

    from flashinfer.comm import UlyssesCommunicator

    # The constructor raises rather than returning None, and FlashInfer makes
    # topology failures collective. Vote anyway: a rank-local failure it did not
    # classify as collective would otherwise split the group across two paths,
    # which is a hang rather than a slowdown.
    comm = None
    error: BaseException | None = None
    try:
        comm = UlyssesCommunicator(
            group,
            max_bytes=capacity * dtype.itemsize,
            dtype=dtype,
            backend="pcie",
            device=device,
        )
    except BaseException as exc:  # noqa: BLE001
        error = exc

    vote = torch.tensor(
        [1 if comm is not None else 0], dtype=torch.int32, device=device
    )
    dist.all_reduce(vote, op=dist.ReduceOp.MIN, group=group)
    if vote.item() == 0:
        if comm is not None:
            try:
                comm.close()
            except Exception:  # noqa: BLE001
                logger.exception("flashinfer ulysses: close after a failed start")
        _TRANSPORTS[name] = None
        if strict:
            raise RuntimeError(
                "flashinfer ulysses strict mode: communicator construction failed"
            ) from error
        logger.warning(
            "flashinfer ulysses could not start on some rank (%s); staying on NCCL",
            error,
        )
        return None

    # The route facts are rank-invariant attributes; construction already made
    # backend="pcie" collective, so the only gate left is the route itself.
    # An explicit route is also a benchmark contract: P2P must remain usable
    # under strict mode, and RDMA must not silently resolve to hybrid/P2P.
    requested_route = (
        os.environ.get("FLASHINFER_ULYSSES_PCIE_ROUTE", "auto").strip().lower()
    )
    expected_routes = (
        (requested_route,)
        if requested_route in ("p2p", "rdma", "hybrid")
        else _RDMA_ROUTES
    )
    if (
        strict
        and (world_size == 8 or requested_route != "auto")
        and comm.transport not in expected_routes
    ):
        comm.close()
        raise RuntimeError(
            f"flashinfer ulysses strict mode: world size {world_size} resolved to "
            f"the {comm.transport!r} route, expected one of {expected_routes} "
            f"for requested route {requested_route!r}."
        )

    transport = FlashInferUlyssesA2A(
        comm,
        world_size,
        max_shapes,
        capacity,
        strict=strict,
    )
    _TRANSPORTS[name] = transport
    logger.info(
        "flashinfer ulysses started for this group "
        "(backend=%s, transport=%s, world_size=%d, capacity=%d)",
        transport.backend,
        transport.transport,
        world_size,
        capacity,
    )
    return transport


def shutdown_flashinfer_ulysses_a2a() -> None:
    """Collective: every rank must call this before any rank exits."""
    for name in list(_TRANSPORTS):
        transport = _TRANSPORTS.pop(name)
        if transport is None:
            continue
        try:
            transport.shutdown()
        except Exception:  # noqa: BLE001
            logger.exception("flashinfer ulysses: teardown failed for %s", name)


def reset_flashinfer_ulysses_request_stats() -> None:
    """Start a request-local interval without disturbing registrations."""
    for transport in _TRANSPORTS.values():
        if transport is not None:
            transport.stats(reset=True)


def validate_flashinfer_ulysses_request(
    expected_exchanges: int | None, *, strict: bool
) -> list[dict[str, object]]:
    """Validate that every live transport carried the whole request.

    ``None`` means the caller cannot state a count -- with cache-dit the number
    of denoising steps that actually reach the blocks is decided at runtime from
    a residual threshold. The check then weakens to "carried something", which
    still catches a transport that was constructed and never used; what it can
    no longer catch is a *partial* fallback, so the counter-based checks stay
    the load-bearing ones there.
    """
    reports = [
        transport.stats() for transport in _TRANSPORTS.values() if transport is not None
    ]
    problems = []
    if not reports:
        problems.append("no active FlashInfer Ulysses communicator")
    for report in reports:
        actual = int(report["sglang_exchanges"])
        if expected_exchanges is None:
            if actual == 0:
                problems.append("transport carried no exchange this request")
        elif actual != expected_exchanges:
            problems.append(
                f"rank-local exchange delta is {actual}, expected {expected_exchanges}"
            )
    if problems:
        message = "MiniMax-H3 FlashInfer Ulysses request validation: " + "; ".join(
            problems
        )
        if strict:
            raise RuntimeError(message)
        logger.warning("%s; request used an NCCL fallback", message)
    else:
        logger.info(
            "MiniMax-H3 FlashInfer Ulysses request completed with %s exchanges",
            expected_exchanges
            if expected_exchanges is not None
            else int(reports[0]["sglang_exchanges"]),
        )
    return reports
