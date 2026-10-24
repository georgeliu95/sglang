# SPDX-License-Identifier: Apache-2.0
"""Entry points for FlashInfer's SM120 Sage block-sparse attention (PR #4691).

``bsa_attn_sm120_blk64_sage_fwd`` is the QK-INT8 / PV-FP8 sibling of
``bsa_attn_sm120_blk64_fwd``: same 64-token block granularity, same
``q2k_block_index`` contract, so a :class:`~..subblock_sparse.SubBlockRouter`
plan feeds either one unchanged. Two things do differ and both are the caller's
problem:

* **Layout.** The Sage kernel is BHSD (batch, heads, seqlen, head_dim)
  throughout, matching upstream Block-Sparse-Attention's native contract and
  the layout its Triton quantizers emit. The bf16 kernel is BSHD.
* **Operands.** Q/K/V must be pre-quantized by ``quantize_sage_*_sm120`` --
  per-32-token-group Q, channel-mean-centered per-K64-tile K, per-channel V in
  an HDS layout with a 16-token physical permutation baked in for the FP8 PV
  MMA. There is no in-kernel quantization path.

The kernel computes no LSE and is MHA-only (``num_kv_heads == num_heads``),
which is why PR #4691 leaves it out of ``BlockSparseAttentionWrapper``. The DiT
self-attention it is used for here is MHA and never asks for LSE.
"""

from __future__ import annotations

import functools
import importlib.util
import sys
import types
from pathlib import Path


def _isolated_sparse_module(rel_path: str, pkg: str):
    """Load one file out of ``flashinfer/cute_dsl/sparse`` under a synthetic parent.

    ``rel_path`` is relative to that directory, e.g. ``bsa_attn_sm120.py`` or
    ``bsa_utils/sage_quant_sm120.py``.

    Same escape hatch, and the same reason, as ``router.py``'s loaders: importing
    the ``flashinfer.cute_dsl.sparse`` *package* has historically dragged in the
    blk128 CuTe-DSL backend, which hard-requires ``quack`` and tracks a moving
    ``cutlass.cute`` API. On the PR #4691 tree that package ``__init__`` is lazy
    (``__getattr__``) so the direct submodule import below normally succeeds;
    this stays as the fallback for trees where it does not.
    """
    import flashinfer

    base = Path(flashinfer.__file__).resolve().parent / "cute_dsl" / "sparse"

    def _load(name: str, path: Path, is_pkg: bool):
        spec = importlib.util.spec_from_file_location(
            name,
            path,
            submodule_search_locations=[str(path.parent)] if is_pkg else None,
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    if pkg not in sys.modules:
        parent = types.ModuleType(pkg)
        # Relative imports inside the loaded file (``.bsa_utils.cache_utils``,
        # ``.sm120_blk64.flash_fwd_sm120_sage``) resolve through this __path__.
        parent.__path__ = [str(base)]
        sys.modules[pkg] = parent
    dotted = rel_path.removesuffix(".py").replace("/", ".")
    return _load(f"{pkg}.{dotted}", base / rel_path, False)


@functools.lru_cache(maxsize=1)
def load_bsa_attn_sm120_blk64_sage_fwd():
    """FlashInfer's SM120 Sage 64-block sparse attention entry point (PR #4691)."""
    try:
        from flashinfer.cute_dsl.sparse.bsa_attn_sm120 import (
            bsa_attn_sm120_blk64_sage_fwd,
        )

        return bsa_attn_sm120_blk64_sage_fwd
    except Exception:
        pass
    module = _isolated_sparse_module(
        "bsa_attn_sm120.py", "_flashinfer_sparse_sm120_sage_only"
    )
    return module.bsa_attn_sm120_blk64_sage_fwd


@functools.lru_cache(maxsize=1)
def load_quantize_sage_qkv_sm120():
    """The Triton quantizer that produces this kernel's operand layout.

    Kept next to the kernel loader because the two are one unit: the kernel
    asserts on dtypes and on V's padded HDS shape, so hand-rolled quantization
    is not an option.
    """
    try:
        from flashinfer.cute_dsl.sparse.bsa_utils.sage_quant_sm120 import (
            quantize_sage_qkv_sm120,
        )

        return quantize_sage_qkv_sm120
    except Exception:
        pass
    module = _isolated_sparse_module(
        "bsa_utils/sage_quant_sm120.py", "_flashinfer_sparse_sm120_sage_only"
    )
    return module.quantize_sage_qkv_sm120


__all__ = [
    "load_bsa_attn_sm120_blk64_sage_fwd",
    "load_quantize_sage_qkv_sm120",
]
