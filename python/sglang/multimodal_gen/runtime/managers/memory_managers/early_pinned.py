"""SGLANG_DIFFUSION_EARLY_PINNED: place the process's long-lived pinned host memory in clean memory.

On a shared host/device pool (GB10) kernel compaction isolates pages and tries to migrate them; pinned pages cannot
move, and every failed migration slows every kernel on the GPU. How many fail depends on where the pinned pages sit:
allocated into clean memory, 21 GiB of pinned memory let ~10% of isolated pages fail (2 s of slowdown in the probe);
allocated while the page cache filled memory, 60-70% failed (80-190 s). On a fresh worker the page cache is usually
still full of the checkpoints a previous run read.

So, at worker start, before the distributed environment and before any model is read:

1. drop this model's own checkpoint page cache (posix_fadvise DONTNEED; clean pages only, mapped pages are kept);
2. reserve the pinned buffers known to live for the whole process: the MiniMax-H3 AdaLN host tier
   (``--minimax-h3-adaln-host-cache-gb``), handed to the tier by :func:`take`;
3. only with SGLANG_DIFFUSION_EARLY_PINNED_NCCL=1: set ``NCCL_RUNTIME_CONNECT=0`` and, once the groups exist, run one
   collective on every NCCL group (:func:`warm_up_process_groups`), so NCCL connects all transports now (without GPU
   Direct RDMA its network buffers are pinned host memory). Off by default: on 8 GB10 ranks (H3, 2026-11-23) the eager
   connection pinned 54.2 GiB per rank instead of the 9.4-16.6 GiB that lazy connection reaches, and the measured
   request got 1.6% slower (rdma-g7-e4h-ep-mem vs rdma-g7-e4h-mem).
"""

from __future__ import annotations

import glob
import os

import torch

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

ADALN_HOST_TIER = "minimax_h3_adaln_host_tier"
_reserved: dict[str, torch.Tensor] = {}


def enabled() -> bool:
    return bool(envs.SGLANG_DIFFUSION_EARLY_PINNED)


def nccl_eager() -> bool:
    return enabled() and os.environ.get("SGLANG_DIFFUSION_EARLY_PINNED_NCCL", "0") == "1"


def _meminfo_gib() -> str:
    try:
        fields = {}
        with open("/proc/meminfo") as handle:
            for line in handle:
                key, value = line.split(":", 1)
                if key in ("MemFree", "Cached", "Shmem"):
                    fields[key] = int(value.split()[0]) / 2**20
        return " ".join(f"{k}={v:.1f}GiB" for k, v in fields.items())
    except OSError:
        return "meminfo unavailable"


def drop_checkpoint_cache(model_path: str | None) -> int:
    """posix_fadvise(DONTNEED) over every *.safetensors under model_path; returns the number of files."""
    if not model_path or not os.path.isdir(model_path):
        return 0
    count = 0
    for file in glob.glob(os.path.join(model_path, "**", "*.safetensors"), recursive=True):
        try:
            fd = os.open(file, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            count += 1
        finally:
            os.close(fd)
    return count


def prepare(server_args) -> None:
    """Worker start, before the distributed environment and any model (no-op unless enabled)."""
    if not enabled():
        return
    before = _meminfo_gib()
    files = drop_checkpoint_cache(getattr(server_args, "model_path", None))
    if nccl_eager():
        os.environ.setdefault("NCCL_RUNTIME_CONNECT", "0")
    reserved = []
    host_gb = float(getattr(server_args, "minimax_h3_adaln_host_cache_gb", 0) or 0)
    if getattr(server_args, "minimax_h3_adaln_online", False) and host_gb > 0 and torch.cuda.is_available():
        nbytes = int(host_gb * 1e9)
        try:
            _reserved[ADALN_HOST_TIER] = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
            reserved.append(f"AdaLN host tier {nbytes / 2**30:.2f} GiB")
        except RuntimeError as exc:
            logger.warning("early pinned: could not reserve the AdaLN host tier (%s); it allocates later", exc)
    logger.info(
        "early pinned: dropped the page cache of %d checkpoint files, reserved [%s], NCCL_RUNTIME_CONNECT=%s; "
        "memory before %s, after %s",
        files,
        ", ".join(reserved) or "nothing",
        os.environ.get("NCCL_RUNTIME_CONNECT"),
        before,
        _meminfo_gib(),
    )


def take(name: str, nbytes: int) -> torch.Tensor | None:
    """The reserved pinned buffer for ``name`` (uint8, at least ``nbytes`` long), or None; each buffer is handed out once."""
    buffer = _reserved.pop(name, None)
    if buffer is None:
        return None
    if buffer.numel() < nbytes:
        logger.warning("early pinned: %s reserved %d bytes, %d requested; not used", name, buffer.numel(), nbytes)
        return None
    return buffer


def reserved_bytes(name: str) -> int:
    buffer = _reserved.get(name)
    return 0 if buffer is None else buffer.numel()


def warm_up_process_groups() -> None:
    """One all_reduce on every multi-rank NCCL group, in a rank-independent order, so their communicators (and, with
    NCCL_RUNTIME_CONNECT=0, all their connections and pinned buffers) exist before any model is read."""
    if not nccl_eager() or not torch.distributed.is_initialized():
        return
    from sglang.multimodal_gen.runtime.distributed import group_coordinator

    before = _meminfo_gib()
    warmed = []
    for name in sorted(group_coordinator._groups):
        coordinator = group_coordinator._groups[name]()
        group = getattr(coordinator, "device_group", None) if coordinator is not None else None
        if group is None or torch.distributed.get_world_size(group) < 2:
            continue
        if torch.distributed.get_backend(group) != "nccl":
            continue
        probe = torch.zeros(1, device=torch.cuda.current_device())
        torch.distributed.all_reduce(probe, group=group)
        warmed.append(name)
    torch.cuda.synchronize()
    logger.info("early pinned: NCCL groups warmed up %s; memory before %s, after %s", warmed, before, _meminfo_gib())
