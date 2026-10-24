// Fused GroupNorm + SiLU over fp32 NCTHW activations, with statistics per
// (batch, group) or per (batch, frame, group) when `time_isolated`.
//
// Statistics are exact two-pass central moments. The epilogue is the direct
// form ((x - mean) * rstd) * gamma + beta with one rounding per op, then SiLU
// with the precise expf: a constant group comes out as exactly silu(beta),
// which aten's factorized a * x + b loses to cancellation, and otherwise the
// output tracks F.silu(F.group_norm(x)) up to fp32 reduction-order noise.

#pragma once

#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/cta.cuh>
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <limits>

namespace sglang {

namespace group_norm_silu_ncthw {

constexpr uint32_t kBlockSize = 256;
constexpr uint32_t kWarps = kBlockSize / device::kWarpThreads;
constexpr uint32_t kVecWidth = 4;  // fp32 lanes per 128-bit load
constexpr int64_t kVecAlignment = kVecWidth * sizeof(float);
// Row sizes served by one launch with the whole row in registers; larger rows
// take the partial -> merge -> apply path. 8192 fp32 is 32 registers a thread.
constexpr uint32_t kSmallRow = 4096;
constexpr uint32_t kLargeRow = 8192;
// Elements per statistics chunk on the split path. Arbitrary: 16 fp32 per
// thread at kBlockSize threads, so a chunk also stays in registers.
constexpr uint32_t kChunk = 4096;
constexpr uint32_t kMaxGridY = 65535;

/// \brief Geometry of one launch: input strides plus the group decomposition.
struct GroupNormLayout {
  int64_t strides[5];  ///< `x` strides in elements, (B, C, T, H, W) order
  uint32_t channels;
  uint32_t time;
  uint32_t height;
  uint32_t width;
  uint32_t groups;
  uint32_t channels_per_group;
  uint32_t span;   ///< elements of one channel inside a row: H*W, or T*H*W when time is reduced too
  uint32_t count;  ///< elements per statistics row: channels_per_group * span
  bool time_isolated;
};

/// \brief Coordinates shared by every element of one statistics row.
struct RowCoords {
  uint32_t batch;
  uint32_t time;  ///< 0 unless `time_isolated`
  uint32_t first_channel;
  int64_t base;  ///< element offset of (batch, first_channel, time, 0, 0) in `x`
};

/// \brief Per-channel statistics and affine parameters of one epilogue element.
struct ChannelAffine {
  float mean;
  float rstd;
  float gamma;
  float beta;
};

/// \brief Mean and sum of squared deviations of one row or chunk.
struct Moments {
  float mean;
  float m2;
};

/// \brief Input and output element offsets of one in-row element.
struct ElementOffsets {
  int64_t in;
  int64_t out;
};

SGL_DEVICE RowCoords decode_row(const GroupNormLayout& layout, uint32_t row) {
  RowCoords coords;
  const uint32_t batch_time = row / layout.groups;
  coords.first_channel = (row % layout.groups) * layout.channels_per_group;
  coords.time = layout.time_isolated ? batch_time % layout.time : 0u;
  coords.batch = layout.time_isolated ? batch_time / layout.time : batch_time;
  coords.base =
      coords.batch * layout.strides[0] + coords.first_channel * layout.strides[1] + coords.time * layout.strides[2];
  return coords;
}

/// \brief General-stride addressing: decode `index` into (channel, t, y, x) and
/// apply the `x` strides; the output is always contiguous NCTHW.
SGL_DEVICE ElementOffsets strided_offsets(const GroupNormLayout& layout, const RowCoords& row, uint32_t index) {
  const uint32_t x = index % layout.width;
  const uint32_t lines = index / layout.width;
  const uint32_t y = lines % layout.height;
  const uint32_t planes = lines / layout.height;
  const uint32_t frame = layout.time_isolated ? row.time : planes % layout.time;
  const uint32_t channel = layout.time_isolated ? planes : planes / layout.time;
  const int64_t frame_offset = layout.time_isolated ? 0 : frame * layout.strides[2];
  ElementOffsets offsets;
  offsets.in = row.base + channel * layout.strides[1] + frame_offset + y * layout.strides[3] + x * layout.strides[4];
  offsets.out =
      (((static_cast<int64_t>(row.batch) * layout.channels + row.first_channel + channel) * layout.time + frame) *
           layout.height +
       y) *
          layout.width +
      x;
  return offsets;
}

/// \brief Contiguous-input addressing: channel `index / span` of the row, then
/// `index % span` inside that channel's run. Input and output offsets coincide.
SGL_DEVICE int64_t contiguous_offset(const GroupNormLayout& layout, const RowCoords& row, uint32_t index) {
  const uint32_t channel = index / layout.span;
  return row.base + channel * layout.strides[1] + (index - channel * layout.span);
}

template <bool kVectorized>
SGL_DEVICE ElementOffsets element_offsets(const GroupNormLayout& layout, const RowCoords& row, uint32_t index) {
  if constexpr (kVectorized) {
    const int64_t offset = contiguous_offset(layout, row, index);
    return ElementOffsets{offset, offset};
  } else {
    return strided_offsets(layout, row, index);
  }
}

SGL_DEVICE ChannelAffine channel_affine(float mean, float rstd, float gamma, float beta) {
  return ChannelAffine{mean, rstd, gamma, beta};
}

/// \brief ((x - mean) * rstd) * gamma + beta, one rounding per op, then SiLU; the
/// one activation both addressing paths use, so they agree bit-for-bit given
/// equal statistics. Spelled with intrinsics so nvcc cannot contract the
/// affine step into an fma, which would reintroduce the cancellation.
SGL_DEVICE float affine_silu(float value, ChannelAffine affine) {
  const float normalized = __fmul_rn(__fsub_rn(value, affine.mean), affine.rstd);
  const float y = __fadd_rn(__fmul_rn(normalized, affine.gamma), affine.beta);
  return y / (1.f + device::math::exp(-y));
}

/// \brief Two-pass moments of the register-resident values of a CTA.
///
/// Element `j` of a thread has in-row index `first + j * stride` and takes part
/// only when that index is below `end`; `length` is the element count of the
/// row or chunk being reduced. Both reductions leave their result in shared
/// memory, so the caller must not reuse `smem_sum`/`smem_m2` afterwards.
template <uint32_t kWidth, uint32_t kVecN>
SGL_DEVICE Moments cta_moments(
    const device::AlignedVector<float, kWidth> (&values)[kVecN],
    uint32_t first,
    uint32_t stride,
    uint32_t end,
    uint32_t length,
    float* smem_sum,
    float* smem_m2) {
  float sum = 0.f;
#pragma unroll
  for (uint32_t j = 0; j < kVecN; ++j) {
    if (first + j * stride < end) {
#pragma unroll
      for (uint32_t e = 0; e < kWidth; ++e) {
        sum += values[j][e];
      }
    }
  }
  device::cta::reduce_sum(sum, smem_sum);
  __syncthreads();
  const float mean = smem_sum[0] / static_cast<float>(length);

  float m2 = 0.f;
#pragma unroll
  for (uint32_t j = 0; j < kVecN; ++j) {
    if (first + j * stride < end) {
#pragma unroll
      for (uint32_t e = 0; e < kWidth; ++e) {
        const float delta = values[j][e] - mean;
        m2 += delta * delta;
      }
    }
  }
  device::cta::reduce_sum(m2, smem_m2);
  __syncthreads();
  return Moments{mean, smem_m2[0]};
}

/**
 * \brief GroupNorm + SiLU for rows of at most `kBlockSize * kVecN * width`
 *        elements; one CTA per statistics row.
 *
 * The row is loaded once into registers, reduced twice (mean, then squared
 * deviations) and written back through the fused affine + SiLU epilogue.
 *
 * \tparam kVectorized Contiguous input addressed in 128-bit vectors; otherwise
 *                     general strides, one element at a time.
 * \tparam kVecN Vectors (or scalars) each thread keeps in registers; sets the
 *               row capacity of the launch.
 * \param x Input [B, C, T, H, W] fp32; contiguous when `kVectorized`.
 * \param weight Per-channel scale [C] fp32.
 * \param bias Per-channel shift [C] fp32.
 * \param out Contiguous output [B, C, T, H, W] fp32.
 * \param layout Sizes, strides and the group decomposition.
 * \param eps Variance epsilon.
 */
template <bool kVectorized, uint32_t kVecN>
__global__ void group_norm_silu_ncthw_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    const GroupNormLayout __grid_constant__ layout,
    float eps) {
  constexpr uint32_t kWidth = kVectorized ? kVecWidth : 1;
  constexpr uint32_t kStride = kBlockSize * kWidth;
  using Vec = device::AlignedVector<float, kWidth>;
  __shared__ float smem_sum[kWarps];
  __shared__ float smem_m2[kWarps];

  const RowCoords row = decode_row(layout, blockIdx.x);
  const uint32_t first = threadIdx.x * kWidth;
  Vec values[kVecN];
#pragma unroll
  for (uint32_t j = 0; j < kVecN; ++j) {
    const uint32_t index = first + j * kStride;
    if (index < layout.count) {
      values[j].load(x + element_offsets<kVectorized>(layout, row, index).in);
    }
  }

  const Moments moments =
      cta_moments<kWidth, kVecN>(values, first, kStride, layout.count, layout.count, smem_sum, smem_m2);
  const float rstd = device::math::rsqrt(moments.m2 / static_cast<float>(layout.count) + eps);

#pragma unroll
  for (uint32_t j = 0; j < kVecN; ++j) {
    const uint32_t index = first + j * kStride;
    if (index >= layout.count) continue;
    const uint32_t channel = row.first_channel + index / layout.span;
    const ChannelAffine affine = channel_affine(moments.mean, rstd, weight[channel], bias[channel]);
#pragma unroll
    for (uint32_t e = 0; e < kWidth; ++e) {
      values[j][e] = affine_silu(values[j][e], affine);
    }
    values[j].store(out + element_offsets<kVectorized>(layout, row, index).out);
  }
}

/**
 * \brief Split path, stage 1: mean and squared-deviation sum of one `kChunk`
 *        slice of a row. Grid is (rows, parts).
 *
 * \tparam kVectorized See `group_norm_silu_ncthw_kernel`.
 * \param x Input [B, C, T, H, W] fp32.
 * \param partial Output `[rows, parts]` of (mean, m2) per chunk.
 * \param layout Sizes, strides and the group decomposition.
 * \param parts Chunks per row, `ceil(count / kChunk)`.
 */
template <bool kVectorized>
__global__ void group_norm_silu_ncthw_partial_kernel(
    const float* __restrict__ x,
    fp32x2_t* __restrict__ partial,
    const GroupNormLayout __grid_constant__ layout,
    uint32_t parts) {
  constexpr uint32_t kWidth = kVectorized ? kVecWidth : 1;
  constexpr uint32_t kStride = kBlockSize * kWidth;
  constexpr uint32_t kVecN = kChunk / kStride;
  static_assert(kChunk % kStride == 0, "kChunk must be a whole number of CTA passes");
  using Vec = device::AlignedVector<float, kWidth>;
  __shared__ float smem_sum[kWarps];
  __shared__ float smem_m2[kWarps];

  const RowCoords row = decode_row(layout, blockIdx.x);
  const uint32_t start = blockIdx.y * kChunk;
  const uint32_t end = min(start + kChunk, layout.count);
  const uint32_t first = start + threadIdx.x * kWidth;
  Vec values[kVecN];
#pragma unroll
  for (uint32_t j = 0; j < kVecN; ++j) {
    const uint32_t index = first + j * kStride;
    if (index < end) {
      values[j].load(x + element_offsets<kVectorized>(layout, row, index).in);
    }
  }

  const Moments moments = cta_moments<kWidth, kVecN>(values, first, kStride, end, end - start, smem_sum, smem_m2);
  if (threadIdx.x == 0) {
    partial[static_cast<int64_t>(blockIdx.x) * parts + blockIdx.y] = make_float2(moments.mean, moments.m2);
  }
}

/**
 * \brief Split path, stage 2: fold the chunk moments of one row into
 *        (mean, rstd); one CTA per row.
 *
 * Chan's parallel update: the tail chunk contributes its actual length to both
 * the weighted mean and the merged second moment.
 *
 * \param partial `[rows, parts]` of (mean, m2) per chunk.
 * \param stats Output `[rows]` of (mean, rstd).
 * \param count Elements per row.
 * \param parts Chunks per row.
 * \param eps Variance epsilon.
 */
__global__ void group_norm_silu_ncthw_merge_kernel(
    const fp32x2_t* __restrict__ partial, fp32x2_t* __restrict__ stats, uint32_t count, uint32_t parts, float eps) {
  __shared__ float smem_sum[kWarps];
  __shared__ float smem_m2[kWarps];
  const fp32x2_t* row_partial = partial + static_cast<int64_t>(blockIdx.x) * parts;

  float sum = 0.f;
  for (uint32_t part = threadIdx.x; part < parts; part += kBlockSize) {
    const uint32_t length = min(kChunk, count - part * kChunk);
    sum += row_partial[part].x * static_cast<float>(length);
  }
  device::cta::reduce_sum(sum, smem_sum);
  __syncthreads();
  const float mean = smem_sum[0] / static_cast<float>(count);

  float m2 = 0.f;
  for (uint32_t part = threadIdx.x; part < parts; part += kBlockSize) {
    const uint32_t length = min(kChunk, count - part * kChunk);
    const fp32x2_t moment = row_partial[part];
    const float delta = moment.x - mean;
    m2 += moment.y + delta * delta * static_cast<float>(length);
  }
  device::cta::reduce_sum(m2, smem_m2);
  __syncthreads();
  if (threadIdx.x == 0) {
    stats[blockIdx.x] = make_float2(mean, device::math::rsqrt(smem_m2[0] / static_cast<float>(count) + eps));
  }
}

/**
 * \brief Split path, stage 3: normalize, affine and SiLU one `kChunk` slice of
 *        a row with the merged statistics. Grid is (rows, parts).
 *
 * \tparam kVectorized See `group_norm_silu_ncthw_kernel`.
 * \param x Input [B, C, T, H, W] fp32.
 * \param weight Per-channel scale [C] fp32.
 * \param bias Per-channel shift [C] fp32.
 * \param stats `[rows]` of (mean, rstd).
 * \param out Contiguous output [B, C, T, H, W] fp32.
 * \param layout Sizes, strides and the group decomposition.
 */
template <bool kVectorized>
__global__ void group_norm_silu_ncthw_apply_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const fp32x2_t* __restrict__ stats,
    float* __restrict__ out,
    const GroupNormLayout __grid_constant__ layout) {
  constexpr uint32_t kWidth = kVectorized ? kVecWidth : 1;
  constexpr uint32_t kStride = kBlockSize * kWidth;
  constexpr uint32_t kVecN = kChunk / kStride;
  using Vec = device::AlignedVector<float, kWidth>;

  const RowCoords row = decode_row(layout, blockIdx.x);
  const fp32x2_t stat = stats[blockIdx.x];
  const uint32_t start = blockIdx.y * kChunk;
  const uint32_t end = min(start + kChunk, layout.count);
  const uint32_t first = start + threadIdx.x * kWidth;
#pragma unroll
  for (uint32_t j = 0; j < kVecN; ++j) {
    const uint32_t index = first + j * kStride;
    if (index >= end) continue;
    const ElementOffsets offsets = element_offsets<kVectorized>(layout, row, index);
    const uint32_t channel = row.first_channel + index / layout.span;
    const ChannelAffine affine = channel_affine(stat.x, stat.y, weight[channel], bias[channel]);
    Vec value;
    value.load(x + offsets.in);
#pragma unroll
    for (uint32_t e = 0; e < kWidth; ++e) {
      value[e] = affine_silu(value[e], affine);
    }
    value.store(out + offsets.out);
  }
}

/**
 * \brief Validate the tensors and dispatch by row size: one register-resident
 *        launch up to `kLargeRow` elements per row, otherwise
 *        partial -> merge -> apply with scratch from the caching allocator.
 *
 * \tparam kVectorized Chosen by the Python wrapper: `x` contiguous, 16-byte
 *                     aligned and `W % 4 == 0`. Enforced here.
 */
template <bool kVectorized>
struct GroupNormSiluNcthwKernel {
  /**
   * \param x Input [B, C, T, H, W] fp32 on CUDA; any strides unless `kVectorized`.
   * \param weight Per-channel scale [C] fp32.
   * \param bias Per-channel shift [C] fp32.
   * \param out Contiguous output [B, C, T, H, W] fp32.
   * \param num_groups Groups along C; `C % num_groups == 0`.
   * \param time_isolated Reduce over (C/G, H, W) per frame instead of (C/G, T, H, W).
   * \param eps Variance epsilon.
   */
  static void
  run(tvm::ffi::TensorView x,
      tvm::ffi::TensorView weight,
      tvm::ffi::TensorView bias,
      tvm::ffi::TensorView out,
      int64_t num_groups,
      bool time_isolated,
      double eps) {
    using namespace host;

    auto B = SymbolicSize{"batch"};
    auto C = SymbolicSize{"channels"};
    auto T = SymbolicSize{"time"};
    auto H = SymbolicSize{"height"};
    auto W = SymbolicSize{"width"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    if constexpr (kVectorized) {
      TensorMatcher({B, C, T, H, W})
          .with_dtype<fp32_t>()
          .with_device(device)
          .ensure_alignment(kVecAlignment)
          .verify(x)
          .verify(out);
    } else {
      TensorMatcher({B, C, T, H, W})
          .with_strides({-1, -1, -1, -1, -1})
          .with_dtype<fp32_t>()
          .with_device(device)
          .verify(x);
      TensorMatcher({B, C, T, H, W}).with_dtype<fp32_t>().with_device(device).verify(out);
    }
    TensorMatcher({C}).with_dtype<fp32_t>().with_device(device).verify(weight).verify(bias);

    const int64_t batch = B.unwrap();
    const int64_t channels = C.unwrap();
    const int64_t time = T.unwrap();
    const int64_t height = H.unwrap();
    const int64_t width = W.unwrap();
    CHECK_HOST(num_groups > 0 && channels % num_groups == 0)
        << "group_norm_silu_ncthw: channels=" << channels << " is not divisible by num_groups=" << num_groups;
    const int64_t channels_per_group = channels / num_groups;
    const int64_t span = height * width * (time_isolated ? 1 : time);
    const int64_t count = channels_per_group * span;
    const int64_t rows = batch * num_groups * (time_isolated ? time : 1);
    CHECK_HOST(count > 0 && rows > 0) << "group_norm_silu_ncthw: empty input";
    constexpr int64_t kIndexMax = std::numeric_limits<uint32_t>::max();
    CHECK_HOST(
        count <= kIndexMax && channels <= kIndexMax && time <= kIndexMax && height <= kIndexMax && width <= kIndexMax)
        << "group_norm_silu_ncthw: dimensions exceed uint32 indexing";
    CHECK_HOST(rows <= std::numeric_limits<int32_t>::max())
        << "group_norm_silu_ncthw: too many statistics rows for gridDim.x: " << rows;
    if constexpr (kVectorized) {
      // ensure_alignment skips size-1 dims, so a [.., 1, W] frame needs this explicitly.
      CHECK_HOST(width % kVecWidth == 0) << "group_norm_silu_ncthw: vectorized path needs W % 4 == 0, got W=" << width;
    }

    GroupNormLayout layout{};
    for (int d = 0; d < 5; ++d) {
      layout.strides[d] = x.stride(d);
    }
    layout.channels = static_cast<uint32_t>(channels);
    layout.time = static_cast<uint32_t>(time);
    layout.height = static_cast<uint32_t>(height);
    layout.width = static_cast<uint32_t>(width);
    layout.groups = static_cast<uint32_t>(num_groups);
    layout.channels_per_group = static_cast<uint32_t>(channels_per_group);
    layout.span = static_cast<uint32_t>(span);
    layout.count = static_cast<uint32_t>(count);
    layout.time_isolated = time_isolated;

    const auto* x_ptr = static_cast<const float*>(x.data_ptr());
    const auto* weight_ptr = static_cast<const float*>(weight.data_ptr());
    const auto* bias_ptr = static_cast<const float*>(bias.data_ptr());
    auto* out_ptr = static_cast<float*>(out.data_ptr());
    const DLDevice dev = device.unwrap();
    const auto grid_rows = static_cast<uint32_t>(rows);
    const auto eps_f32 = static_cast<float>(eps);
    constexpr uint32_t kWidth = kVectorized ? kVecWidth : 1;

    if (count <= kSmallRow) {
      LaunchKernel(grid_rows, kBlockSize, dev)(
          group_norm_silu_ncthw_kernel<kVectorized, kSmallRow / (kBlockSize * kWidth)>,
          x_ptr,
          weight_ptr,
          bias_ptr,
          out_ptr,
          layout,
          eps_f32);
    } else if (count <= kLargeRow) {
      LaunchKernel(grid_rows, kBlockSize, dev)(
          group_norm_silu_ncthw_kernel<kVectorized, kLargeRow / (kBlockSize * kWidth)>,
          x_ptr,
          weight_ptr,
          bias_ptr,
          out_ptr,
          layout,
          eps_f32);
    } else {
      const int64_t parts = div_ceil(count, static_cast<int64_t>(kChunk));
      CHECK_HOST(parts <= kMaxGridY) << "group_norm_silu_ncthw: row of " << count << " elements needs " << parts
                                     << " chunks, more than gridDim.y allows";
      auto workspace = ffi::alloc_workspace_tensor(static_cast<size_t>(rows * (parts + 1)) * sizeof(fp32x2_t), dev);
      auto* partial = static_cast<fp32x2_t*>(workspace.data_ptr());
      auto* stats = partial + rows * parts;
      const dim3 grid(grid_rows, static_cast<uint32_t>(parts));
      LaunchKernel(grid, kBlockSize, dev)(
          group_norm_silu_ncthw_partial_kernel<kVectorized>, x_ptr, partial, layout, static_cast<uint32_t>(parts));
      LaunchKernel(grid_rows, kBlockSize, dev)(
          group_norm_silu_ncthw_merge_kernel, partial, stats, layout.count, static_cast<uint32_t>(parts), eps_f32);
      LaunchKernel(grid, kBlockSize, dev)(
          group_norm_silu_ncthw_apply_kernel<kVectorized>, x_ptr, weight_ptr, bias_ptr, stats, out_ptr, layout);
    }
  }
};

}  // namespace group_norm_silu_ncthw

}  // namespace sglang
