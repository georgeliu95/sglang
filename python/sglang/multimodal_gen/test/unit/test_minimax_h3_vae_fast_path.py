# SPDX-License-Identifier: Apache-2.0
"""H3 batching order, fast-path installation and request isolation."""

import logging
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

from sglang.multimodal_gen.runtime.models.vaes.fast_path_gate import (
    VaeFastPathGate,
    register_vae_fast_path_gate,
    use_vae_fast_path,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_vae_cuda_opt import (
    _install_fast_path_slots,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae import (
    AutoencoderKLLegacy,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.attention import (
    Attention,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.batching import (
    decode_windows,
    flat_decode_tiles,
    forward_many,
    partitions,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.fast_path import (
    MiniMaxH3VaeFastPath,
    minimax_h3_vae_fast_path_scope,
    resolve_minimax_h3_vae_batch_caps,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.fp8 import (
    LINEAR_SHAPES,
    inspect_fp8_scope,
    target_paths,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.vae_cnn import (
    EncoderFCN3D,
    ResnetBlock3D,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.vit_utils import (
    prepare_rotary_pos_emb,
)

ATTENTION_MODULE = (
    "sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.attention"
)
ENCODER_CAP = "SGLANG_DIFFUSION_MINIMAX_H3_VAE_ENCODER_TILE_BATCH"
DECODER_CAP = "SGLANG_DIFFUSION_MINIMAX_H3_VAE_DECODER_TILE_BATCH"
WINDOW_CAP = "SGLANG_DIFFUSION_MINIMAX_H3_VAE_WINDOW_BATCH"


def _tiny_vae() -> AutoencoderKLLegacy:
    """Two-level 3D CNN encoder and one-block ViT decoder; 32 px tiles of 16."""
    torch.manual_seed(0)
    with mock.patch(f"{ATTENTION_MODULE}.current_platform.is_cuda", return_value=False):
        vae = AutoencoderKLLegacy(
            in_channels=3,
            out_ch=3,
            ch=32,
            embed_dim=4,
            z_channels=4,
            use_3d_conv=True,
            num_res_blocks=1,
            ch_mult=[1, 1],
            space_down=[2, 1],
            space_up=[1, 2],
            time_down=[1, 1],
            padding_mode="reflect",
            use_t_isolated_gn=True,
            causal_encoder=True,
            causal_decoder=False,
            use_vit_decoder=True,
            vit_decoder_kwargs={
                "dim_head": 16,
                "heads": 2,
                "num_layers": 1,
                "norm_type": "rms_norm",
                "qk_norm_type": "rms_norm",
                "qk_norm_affine": False,
                "ffn_activation_fn": "silu",
                "ffn_use_gated": True,
                "rope_dim_ratio": 0.75,
                "rope_theta": 100.0,
                "num_register_tokens": 1,
            },
            clip_length=4,
            token_drop=1,
            encoder_tiling=True,
            decoder_tiling=True,
            tile_size=16,
            tile_overlap_min=4,
        )
    for parameter in vae.parameters():
        parameter.data.normal_(0, 0.2)
    return vae.eval()


def _install(vae) -> MiniMaxH3VaeFastPath:
    state = MiniMaxH3VaeFastPath(
        gate=VaeFastPathGate(),
        encoder_tile_batch=8,
        decoder_tile_batch=64,
        window_batch=0,
    )
    _install_fast_path_slots(vae, state)
    register_vae_fast_path_gate(vae, state.gate)
    return state


def test_adjacent_batches_preserve_complete_sample_order_and_capacity():
    values = [torch.full((2, 3, 4), float(i)) for i in range(5)]
    calls = []
    collector = mock.Mock()

    def forward(x):
        calls.append(x.clone())
        return x + 7

    outputs = forward_many(values, forward=forward, capacity=5, collector=collector)
    assert [x.shape[0] for x in calls] == [4, 4, 2]
    for original, actual in zip(values, outputs):
        torch.testing.assert_close(actual, original + 7)
    assert collector.collect_stacked.call_args_list == [
        mock.call(2, 2),
        mock.call(2, 2),
    ]
    collector.collect.assert_called_once()
    values.insert(2, torch.zeros(2, 3, 5))
    assert list(partitions(values, capacity=8)) == [(0, 2), (2, 3), (3, 6)]


@pytest.mark.parametrize("capacity", [1, 3, 8, 64])
def test_decoder_flatten_restores_rank_tile_and_window_order(capacity):
    tiles = [torch.arange(5 * 2 * 3).reshape(5, 2, 3) + 100 * i for i in range(7)]
    indices = [1, 4, 6]
    sizes = []

    def forward(x):
        sizes.append(x.shape[0])
        return x * 2

    result = flat_decode_tiles(
        tiles, indices=indices, forward=forward, capacity=capacity
    )
    assert max(sizes) <= capacity
    for index, actual in zip(indices, result):
        torch.testing.assert_close(actual, tiles[index] * 2)


@pytest.mark.parametrize("limit", [0, 1, 2, 3, 64])
@pytest.mark.parametrize("use_head_tail", [False, True])
def test_windows_keep_time_axis_and_boundary_shapes(limit, use_head_tail):
    z = torch.arange(2 * 3 * 13).reshape(2, 3, 13, 1, 1)
    head = torch.full((2, 3, 1, 1, 1), -1) if use_head_tail else None
    tail = torch.full((2, 3, 1, 1, 1), -2) if use_head_tail else None
    calls = []

    def decode(x):
        calls.append(x.clone())
        return x.repeat_interleave(2, dim=2)

    vae = SimpleNamespace(tokens_chunk_size=3, token_overlap=1, _adaptive_decode=decode)
    outputs = dict(
        decode_windows(vae, z, head=head, tail=tail, count=4, window_limit=limit)
    )
    for index in range(4):
        expected = z[:, :, index * 3 : index * 3 + 4]
        if index == 0 and head is not None:
            expected = torch.cat([head, expected], dim=2)
        if index == 3 and tail is not None:
            expected = torch.cat([expected, tail], dim=2)
        torch.testing.assert_close(outputs[index], expected.repeat_interleave(2, dim=2))
    assert all(x.shape[0] % 2 == 0 for x in calls)
    if limit:
        assert max(x.shape[0] for x in calls) <= 2 * limit
    if use_head_tail:
        assert calls[0].shape[2] == 5 and calls[-1].shape[2] == 5


def test_batched_rope_has_reference_fallback_on_cpu():
    cos = torch.randn(3, 7, 1, 48)
    sin = torch.randn_like(cos)
    for enabled in (False, True):
        result = prepare_rotary_pos_emb(
            (cos, sin), dtype=torch.float16, allow_batched_native=enabled
        )
        assert len(result) == 2
        torch.testing.assert_close(result[0], cos)
        torch.testing.assert_close(result[1], sin)


def test_fp8_scope_is_exactly_the_released_144_linears():
    # Meta parameters validate the real topology without allocating 10 GB.
    decoder = nn.Module()
    decoder.transformer_blocks = nn.ModuleList()
    with torch.device("meta"):
        for _ in range(36):
            block = nn.Module()
            block.attn = nn.Module()
            block.ff = nn.Module()
            for path, (out_features, in_features) in LINEAR_SHAPES.items():
                owner, name = path.split(".")
                setattr(
                    getattr(block, owner), name, nn.Linear(in_features, out_features)
                )
            decoder.transformer_blocks.append(block)
        decoder.proj_out = nn.Linear(2048, 3072)
    paths = inspect_fp8_scope(decoder)
    assert len(paths) == len(set(paths)) == 144
    assert paths == target_paths()
    assert "proj_out" not in paths
    decoder.transformer_blocks[5].ff.w2 = nn.Linear(1, 1, device="meta")
    with pytest.raises(ValueError, match="transformer_blocks.5.ff.w2"):
        inspect_fp8_scope(decoder)


def test_install_fills_explicit_slots_and_gate_resets():
    vae = _tiny_vae()
    assert vae.fast_path is None and vae.processor.fast_path is None
    state = MiniMaxH3VaeFastPath(
        gate=VaeFastPathGate(),
        encoder_tile_batch=8,
        decoder_tile_batch=64,
        window_batch=0,
    )
    # Two ResnetBlock3D (two norms each) plus the encoder norm_out; one block.
    assert _install_fast_path_slots(vae, state) == (5, 1)
    register_vae_fast_path_gate(vae, state.gate)
    assert vae.fast_path is state
    assert vae.processor.fast_path is state
    assert vae.decoder.fast_path is state
    slotted = [
        module
        for module in vae.modules()
        if isinstance(module, (ResnetBlock3D, EncoderFCN3D, Attention))
    ]
    assert len(slotted) == 4 and all(m.fast_path is state for m in slotted)
    assert "fast_path" not in vars(vae.encoder.conv_in)
    with pytest.raises(RuntimeError, match="request failed"):
        with use_vae_fast_path(vae, True):
            assert state.gate.enabled
            # A CPU tensor never activates the SM100+ kernels.
            assert not state.active(torch.ones(2))
            with use_vae_fast_path(vae, False):
                assert not state.gate.enabled
            raise RuntimeError("request failed")
    assert not state.gate.enabled


def test_scope_mounts_by_quality_and_logs_counters(caplog):
    vae = _tiny_vae()
    state = _install(vae)
    with caplog.at_level(logging.INFO):
        with minimax_h3_vae_fast_path_scope(vae, quality="extra-high", stage="decode"):
            assert state.gate.enabled
            assert not state.admit(False)
            assert state.admit(True)
        assert not state.gate.enabled
        with minimax_h3_vae_fast_path_scope(vae, quality="lossless", stage="decode"):
            assert not state.gate.enabled
    messages = [
        record.getMessage()
        for record in caplog.records
        if "H3 VAE" in record.getMessage()
    ]
    assert messages == [
        "[H3 VAE] decode fast path mounted: quality=extra-high caps=enc8/dec64/win0 "
        "fp8=False",
        "[H3 VAE] decode fast path: used=1 fallback=1",
    ]


def test_scope_encode_mounts_only_at_high(caplog):
    vae = _tiny_vae()
    state = _install(vae)
    with caplog.at_level(logging.INFO):
        with minimax_h3_vae_fast_path_scope(vae, quality="extra-high", stage="encode"):
            assert not state.gate.enabled
        with minimax_h3_vae_fast_path_scope(vae, quality="high", stage="encode"):
            assert state.gate.enabled
        assert not state.gate.enabled
    messages = [
        record.getMessage()
        for record in caplog.records
        if "H3 VAE" in record.getMessage()
    ]
    assert messages == [
        "[H3 VAE] encode fast path mounted: quality=high caps=enc8/dec64/win0 "
        "fp8=False",
        "[H3 VAE] encode fast path: used=0 fallback=0",
    ]


def test_batched_encode_and_decode_match_eager_on_cpu(monkeypatch):
    vae = _tiny_vae()
    x = torch.rand(2, 3, 9, 32, 32)
    z = torch.randn(2, 4, 11, 16, 16)
    with torch.no_grad():
        eager_z = vae.encode_temporal(x)
        eager_dec = vae.decode_temporal(z)
        eager_pixels = vae.processor.revert_tensor(eager_dec)

    state = _install(vae)
    # Stand in for the SM100+ device check; the kernel predicates still refuse
    # CPU tensors, so only the batching changes.
    monkeypatch.setattr(
        MiniMaxH3VaeFastPath, "active", lambda self, value: self.gate.enabled
    )
    decode_batches = []
    eager_decode = vae.decode

    def recording_decode(latent):
        decode_batches.append(int(latent.shape[0]))
        return eager_decode(latent)

    monkeypatch.setattr(vae, "decode", recording_decode)
    with torch.no_grad(), use_vae_fast_path(vae, True):
        fast_z = vae.encode_temporal(x)
        fast_dec = vae.decode_temporal(z)
        fast_pixels = vae.processor.revert_tensor(fast_dec)

    # Two temporal windows x nine tiles x two samples share one decoder forward.
    assert decode_batches == [36]
    assert state.used == 0 and state.fallback > 0
    torch.testing.assert_close(fast_z, eager_z, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(fast_dec, eager_dec, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(fast_pixels, eager_pixels, rtol=1e-4, atol=1e-4)


def test_batch_cap_env_defaults_and_bounds(monkeypatch):
    for name in (ENCODER_CAP, DECODER_CAP, WINDOW_CAP):
        monkeypatch.delenv(name, raising=False)
    assert resolve_minimax_h3_vae_batch_caps() == (8, 64, 1)
    monkeypatch.setenv(ENCODER_CAP, "1")
    monkeypatch.setenv(DECODER_CAP, "64")
    monkeypatch.setenv(WINDOW_CAP, "64")
    assert resolve_minimax_h3_vae_batch_caps() == (1, 64, 64)


@pytest.mark.parametrize(
    "name,value",
    [
        (ENCODER_CAP, "0"),
        (ENCODER_CAP, "65"),
        (DECODER_CAP, "0"),
        (DECODER_CAP, "65"),
        (WINDOW_CAP, "-1"),
        (WINDOW_CAP, "65"),
    ],
)
def test_batch_cap_env_ranges(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        resolve_minimax_h3_vae_batch_caps()
