"""SGLANG_DIFFUSION_EARLY_PINNED (runtime/managers/memory_managers/early_pinned.py).

Off: nothing happens. On: the model's own *.safetensors page cache is dropped, the AdaLN host-tier buffer is pinned
before anything else, handed to the tier exactly once and used as its slab. NCCL is told to connect at init only with
SGLANG_DIFFUSION_EARLY_PINNED_NCCL=1 (eager connection pinned 54 GiB per rank on 8 GB10 ranks).
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.managers.memory_managers import early_pinned


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    early_pinned._reserved.clear()
    monkeypatch.delenv("NCCL_RUNTIME_CONNECT", raising=False)
    monkeypatch.delenv("SGLANG_DIFFUSION_EARLY_PINNED_NCCL", raising=False)
    yield
    early_pinned._reserved.clear()


def _args(tmp_path, **kw):
    return SimpleNamespace(model_path=str(tmp_path), minimax_h3_adaln_online=True, minimax_h3_adaln_host_cache_gb=0.001, **kw)


def test_disabled_is_a_no_op(tmp_path, monkeypatch):
    monkeypatch.setattr(early_pinned, "enabled", lambda: False)
    early_pinned.prepare(_args(tmp_path))
    assert early_pinned._reserved == {}
    assert "NCCL_RUNTIME_CONNECT" not in os.environ


def test_drop_checkpoint_cache_counts_only_safetensors(tmp_path):
    (tmp_path / "transformer").mkdir()
    for name in ("transformer/a.safetensors", "transformer/b.safetensors", "config.json"):
        (tmp_path / name).write_bytes(b"x" * 4096)
    assert early_pinned.drop_checkpoint_cache(str(tmp_path)) == 2
    assert early_pinned.drop_checkpoint_cache(str(tmp_path / "missing")) == 0


def test_take_hands_out_once_and_refuses_short_buffers():
    early_pinned._reserved["x"] = torch.empty(100, dtype=torch.uint8)
    assert early_pinned.reserved_bytes("x") == 100
    assert early_pinned.take("x", 200) is None  # too short: not used (and dropped)
    early_pinned._reserved["x"] = torch.empty(100, dtype=torch.uint8)
    assert early_pinned.take("x", 64).numel() == 100
    assert early_pinned.take("x", 64) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned memory needs CUDA")
def test_prepare_reserves_pinned_adaln_buffer_and_leaves_nccl_lazy(tmp_path, monkeypatch):
    monkeypatch.setattr(early_pinned, "enabled", lambda: True)
    early_pinned.prepare(_args(tmp_path))
    assert "NCCL_RUNTIME_CONNECT" not in os.environ
    buffer = early_pinned._reserved[early_pinned.ADALN_HOST_TIER]
    assert buffer.is_pinned() and buffer.numel() == int(0.001 * 1e9)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned memory needs CUDA")
def test_adaln_host_tier_uses_the_reserved_buffer(monkeypatch):
    from sglang.multimodal_gen.runtime.models.dits.minimax_h3_adaln_cache import MiniMaxH3AdalnHostTier

    layers, width, final = 4, 96, 32
    page_bytes = (layers * width + final) * 2
    buffer = torch.empty(page_bytes * 5 + 7, dtype=torch.uint8, pin_memory=True)
    early_pinned._reserved[early_pinned.ADALN_HOST_TIER] = buffer
    tier = MiniMaxH3AdalnHostTier(num_layers=layers, block_width=width, final_width=final,
                                  capacity_bytes=page_bytes * 3, device=torch.device("cuda"))
    assert tier.num_pages == 3 and tier.pinned
    assert tier._slab.data_ptr() == buffer.data_ptr() and tier._slab.dtype == torch.bfloat16
    assert early_pinned.ADALN_HOST_TIER not in early_pinned._reserved  # handed out once


def test_eager_nccl_connection_is_opt_in(tmp_path, monkeypatch):
    monkeypatch.setattr(early_pinned, "enabled", lambda: True)
    monkeypatch.setattr(early_pinned, "drop_checkpoint_cache", lambda path: 0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)  # no pinned reservation needed here
    assert not early_pinned.nccl_eager()
    early_pinned.prepare(_args(tmp_path))
    assert "NCCL_RUNTIME_CONNECT" not in os.environ
    monkeypatch.setenv("SGLANG_DIFFUSION_EARLY_PINNED_NCCL", "1")
    assert early_pinned.nccl_eager()
    early_pinned.prepare(_args(tmp_path))
    assert os.environ["NCCL_RUNTIME_CONNECT"] == "0"
