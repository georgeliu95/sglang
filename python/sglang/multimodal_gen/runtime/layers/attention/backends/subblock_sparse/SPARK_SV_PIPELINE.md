# Spark zero-SM Ulysses pipeline for MiniMax-H3 (`SGLANG_H3_SV_PIPELINE=1`)

With `SGLANG_H3_SV_PIPELINE=1`, every DiT attention block that runs the SubBlock
Sage A2A backend under pure Ulysses goes through **one chunked pipeline**. The
whole block is pipelined: the QKV GEMM through the out_proj.

```
per chunk of heads, overlapped across chunks:
  QKV GEMM (CuTe FP8, q/k RMSNorm + RoPE + Sage statistics in the epilogue)
  -> Sage INT8/E4M3 quantize + pack straight into RDMA-registered host memory
  -> RDMA all-to-all (verbs WRITEs: zero SMs, no copy engine)
  -> Sage attention with the output E4M3 quantize/pack fused into its epilogue
  -> RDMA all-to-all of the E4M3 output
then: merge + requantize, FP8 out_proj
```

The QKV GEMM of chunk `c+1`, the exchanges of chunk `c` and the attention of
chunk `c-1` run at the same time. No SM ever waits for or moves network data.
The DGX Spark NIC can only register host memory (there is no GPUDirect RDMA on
GB10), so the kernels write and read the registered buffers directly.

The hook is `MiniMaxH3Attention._sv_pipeline_forward`. Without the variable
nothing changes, and the package is never imported. It has the same shape as
`SGLANG_H3_RDMA_OVERLAP`: the pipeline is an external package.

## What it combines

| Piece | Source |
| --- | --- |
| Attention kernel: FA4-style Sage dense / block-sparse kernels (CAKE API) | gitlab-master `guangyunh/gb10-sage-attn` (`gb10_sage_cake_attn`) |
| Same kernels with the Ulysses FP8 output quantize/pack fused into the epilogue (`gb10_fa4_wire`) | `communication_ulysses_overlap` (generated from the source above by text edits) |
| Routing: `route_prequantized`, `mixed_row_indices`, and the gate of `_SparseSageA2ACompute._route_this_step` (dense first steps, `layer_enabled`, `min_seq_len`) | this branch (`subblock_sparse/sage_a2a_kernels.py`) |
| FP8 E4M3 per-channel weights of the delivery, used in place (QKV rows permuted once) | this branch |
| Token FP8 fusion: the producer's `x_prequant` is the QKV GEMM operand itself (no copy) | this branch (`SGLANG_H3_TOKEN_FP8_FUSION`) |
| CuTe FP8 GEMM with RMSNorm + rotate-half RoPE in the epilogue | this branch's `gb10_cute_fp8` lineage (h3-spark-unified) |
| Cross-stream dependencies as device-memory flags on their own 128-byte lines (`SV_SYNC=flags`) | the GB10 finding in `ulysses_head_pipeline/stream_signals.py` |
| Chunk / group schedule, zero-SM verbs transport, INT8 Sage wire, deferred FP8 out_proj, merge + requant | `communication_ulysses_overlap` |

## Running it

The pipeline needs these on `PYTHONPATH`:

- `communication_ulysses_overlap/src` and `communication_ulysses_overlap/shared`
  (gitlab-master `yanwa/communication_ulysses_overlap`, branch `gb10-vsa/bsa-fp8`, commit `5822282` or later). Build
  `spark_verbs/libsv.so` next to `sv_channel.py`:
  `gcc -O2 -shared -fPIC -o libsv.so sv.c -libverbs -lpthread`.
- `Block-Sparse-Attention` (the SM120 blk64 Sage sources and operand helpers).
- `gb10-sage-attn`. Its CUDA extension is JIT-built on first use, with the
  CUTLASS headers taken from the FlashInfer install.

Environment for the worker and the main process (the packed-sequence alignment
is read at import):

```bash
SGLANG_H3_SV_PIPELINE=1
SV_H3_BACKEND=verbs
SV_ATTN_DENSE=fa4w SV_ATTN_ROUTED=fa4w   # FA4 kernels with the fused wire epilogue
SV_STATS_WIRE=1
GB10_SAGE_CAKE_BUILD_DIR=<writable dir>  # extension build cache
```

The container also needs `--device /dev/infiniband --ulimit memlock=-1` and
the rdma-core headers. These are the same settings as the
`SGLANG_H3_RDMA_OVERLAP` path.

The routing config is unchanged (`--attention-backend-config`: sparsity,
n_q/n_k, skip_first_steps, min_seq_len).

## Results

Real H3 generation on DGX Spark (GB10, 200G RoCE, one switch).
Each row pair ran on the same nodes.
- **Baseline:** this branch's best delivery profile, which is `SGLANG_H3_RDMA_OVERLAP=1` with band quant, e4m3h output,
  flags sync, `SGLANG_H3_TOKEN_FP8_FUSION`, and the sp8 settings scaled to N Sparks.
- **Same work on both sides:** the same request and seed, and the same 13 of 50 steps served from the DiT step cache in every
  run.
- **What is reported:** the second clip of each run (steady state), and the mean per computed denoise step.

| Sparks | | dense step (steps 1-9) | routed step | denoise | end to end |
| --- | --- | ---: | ---: | ---: | ---: |
| 2 | baseline | 12467 ms | 7626 ms | 333.0 s | 352.3 s |
| 2 | `SGLANG_H3_SV_PIPELINE=1` | 11571 ms (-7.2%) | 7373 ms (-3.3%) | 317.1 s (-4.8%) | 336.4 s (-4.5%) |
| 4 | baseline | 6309 ms | 3913 ms | 170.0 s | 181.7 s |
| 4 | `SGLANG_H3_SV_PIPELINE=1` | 5766 ms (-8.6%) | 3733 ms (-4.6%) | 159.7 s (-6.1%) | 171.3 s (-5.7%) |
| 8 | baseline | 3341 ms | 2140 ms | 91.9 s | 100.5 s |
| 8 | `SGLANG_H3_SV_PIPELINE=1` | 2979 ms (-10.8%) | 1943 ms (-9.2%) | 83.0 s (-9.7%) | 91.4 s (-9.1%) |

The gain grows with the Ulysses size, because more of the step is communication that the pipeline hides.
- **Dense steps** gain most from the attention kernel.
- **Routed steps** gain from the overlap. The routed-step kernel (fa4w vs the BSA-derived kernel) measured the same in real
  H3 routing (4 Sparks: 3733 vs 3736 ms).

Attention kernels alone on one GB10, at the H3 shape (S = 37765, H heads per launch, routed = top-k 152 of 591 key blocks
with H3 mixed rows):

| | CAKE (delivery) | BSA blk64 + fused wire (before) | **fa4w** (gb10-sage-attn + fused wire) |
| --- | ---: | ---: | ---: |
| dense H=2 | 8.96 ms | 9.26 ms | **8.55 ms** |
| dense H=7 | 30.75 ms | 31.75 ms | **28.09 ms** |
| routed H=2 | 2.42 ms | 2.64 ms | **2.48 ms** |
| routed H=7 | 8.32 ms | 9.13 ms | **8.34 ms** |

Accuracy:
- **Per layer:** y relative RMSE against an FP32 reference of the delivery math is 4.4% on dense steps and 4.9% on routed
  steps (`tests/check_h3_adapter.py`).
- **Across Ulysses sizes:** U2, U4 and U8 outputs are bitwise identical to U1 (`tests/check_u1_alignment.py`).
- **Video:** the baseline's video is byte-identical run to run.
  - Against it, the pipeline's video has PSNR-Y 20.8 / 24.3 / 20.7 dB and SSIM 0.81 / 0.84 / 0.81 at 2 / 4 / 8 Sparks.
  - That is the same spread as the baseline against itself across Ulysses sizes: 2 vs 4 Sparks gives 21.0 dB / 0.82, and
    2 vs 8 gives 20.7 dB / 0.79.
  - Swapping only the attention kernel between two of our own runs already gives 22.9 dB: 50 layers x 37 steps amplify any
    rounding difference into a different sample of the same prompt.
  - The frames show the same scene, composition and motion, with no artifacts.

## Bit-identical to main (`SV_H3_ALIGN=main`)

The default path above is faster than main but not bit-identical to it: its Sage statistics / quantization details,
norm/RoPE in the GEMM epilogue, the fa4w kernel and the FP8 output with a per-row requant differ from main, so the
video is a different sample of the same prompt (PSNR-Y ~21 dB, the same spread main has across Ulysses sizes).

With `SV_H3_ALIGN=main` the package runs main's own numerics on the pipeline's schedule and transport: main's CuTe FP8
GEMM per group of head bands (the GEMM is column-local, so a column subset has the full GEMM's bits),
`fused_inplace_qknorm_rope`, FlashInfer `_ulysses_lowp` statistics per head group (the K-mean rank sum runs on main's
full-width layout: torch's reduction order changes with the width at 8 ranks), per-band `quant_and_pack`, CAKE operands,
`route_prequantized`, CAKE, the e4m3h encode and the head-scaled out_proj. Main's 64-row packing is kept.

CAKE and the GEMMs stay main's kernels. The memory passes around them run as fused kernels that reproduce main's bytes
(package `h3_fused_*`, all on by default, each checked byte for byte against main's kernels):
- QK-norm + RoPE and the Sage statistics (Q group amax, K sum in FlashInfer's lane order, V amax) in one pass per head
  group; Q is not written back, the quantization recomputes it (`SV_ALIGN_FUSED_NORM`, `SV_ALIGN_LAZY_Q`,
  `SV_ALIGN_FUSE_KSUM`);
- quantize-and-pack of a whole head group in one launch, K group amax included, with this rank's own row written straight
  into its receive slot (`SV_ALIGN_FUSED_QUANT`, `SV_ALIGN_OWN_DIRECT`);
- the gated residual + norm2 + AdaLN + per-token E4M3 before the MLP (`SV_H3_FUSE_BLOCK`).

The head bands go through the QKV GEMM in two groups (U4: 7 + 7, `SV_ALIGN_GROUPS`).

- **Real H3 at 4 Sparks:** both clips are byte-identical to main's (md5 equal), and every attention call took this path.
- **One layer:** main's `PreparedPipeline` and this path give the same out_proj output bits on every rank, dense and routed
  (`tests/check_main_aligned_layer.py`: U4 with the current defaults; U2 with the first, unfused version).
- **Speed (4 Sparks, main and this path back to back on the same nodes):** denoise 166.2 s vs 171.2 s (-2.9%); dense
  step 6194 vs 6340 ms (-2.3%), routed step 3816 vs 3946 ms (-3.3%). One attention layer at step 20: 43.7 vs 45.4-45.6
  ms (-3.6% to -4.1%). CAKE and the GEMMs are about 31 of the 44 ms of kernel time per layer, so the remaining room is
  in the memory passes.
- **Tried, kept off:** the fc1 SwiGLU in the GEMM epilogue (`SV_H3_FUSE_SWIGLU=1`: byte-exact, slower on GB10) and
  CAKE reading Q / K in place from the RDMA receive buffers (byte-exact, slower: those buffers are pinned host memory).

## Constraints

- Pure Ulysses with world 2, 4 or 8; no ring. Every head count divides by the
  world size, and head_dim is 128.
- The packed sequence is padded to a multiple of 512 rows (64 × 8): each rank's
  INT8 wire shard must be a multiple of 64 rows. For H3, 37765 rows pad to
  37888 (main's 64-row alignment gives 37824). `SV_H3_ALIGN=main` keeps main's
  alignment.
- The delivery's FP8 per-channel QKV and out_proj weights are required. The QKV
  rows are permuted in place once, so a prepared layer can no longer run the
  delivery path. A non-admitted call on such a layer raises instead of
  computing with permuted weights.
- Admission depends only on shapes and dtypes, so every rank of the Ulysses
  group takes the same decision.
- Not admitted: layers whose parameters are streamed by DiT layerwise offload
  (their storage is swapped per use, which would undo the one-time row
  permutation), and calls under CUDA graph capture (a replay would skip the
  host side of the transport). A prepared layer whose weights were replaced
  anyway raises.
- Set `SGLANG_H3_SV_PIPELINE=1` in the main process and in the workers: the
  packed-sequence alignment is decided where the request is packed.
- Each distinct packed geometry (seq_len, used) gets its own pipeline with its
  own pinned RDMA buffers, kept for the life of the process.

## Concurrency and determinism

- Inside the pipeline no kernel runs next to the attention. Every kernel is on
  one compute stream; the other streams only hold RDMA memops.
- The default attention kernel (gb10-sage-attn + fused wire) and CAKE return
  the same bits even next to concurrent GEMMs or small CTAs of other kernels
  (`tests/check_attn_concurrency.py` on GB10, 20 runs per case). The
  Block-Sparse-Attention blk64 kernels do not: they return wrong rows next to
  small CTAs. They are only used when set explicitly, or as the last fallback,
  with a warning.
- The RoPE cache is bound by tensor identity and version, not by address. H3
  builds it per request, and a later request with the same token count but
  other positions can reuse the freed address.
- Transport sequence numbers compare cyclically and may pass 2^32
  (`tests/check_own_direct.py --seq-wrap`). A failed RDMA post or completion
  aborts the process instead of re-posting.
