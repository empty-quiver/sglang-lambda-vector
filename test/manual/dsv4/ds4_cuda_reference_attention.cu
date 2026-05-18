#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace {

constexpr int kNopeDim = 448;
constexpr int kRopeDim = 64;
constexpr int kHeadDim = kNopeDim + kRopeDim;
constexpr int kScaleGroup = 64;
constexpr int kScaleOffset = kNopeDim + kRopeDim * 2;
constexpr int kPackedBytes = kScaleOffset + kNopeDim / kScaleGroup;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kScaleCount = kNopeDim / kScaleGroup;
constexpr int kScaleCacheOffset = kWarps;
constexpr int kOptimizedV2SharedFloats = kWarps;
constexpr int kOptimizedV3SharedFloats = kScaleCacheOffset + kScaleCount;
constexpr int kOptimizedV5TotalOffset = kWarps;
constexpr int kOptimizedV5SharedFloats = kOptimizedV5TotalOffset + 1;
constexpr int kV4WarpsPerHead = 2;
constexpr int kV4HeadsPerBlock = kWarps / kV4WarpsPerHead;
constexpr int kV4HeadThreads = kV4WarpsPerHead * 32;
constexpr int kV4DimsPerThread = kHeadDim / kV4HeadThreads;

enum class AttentionVariant : int {
  kReference = 0,
  kOptimizedV1 = 1,
  kOptimizedV2 = 2,
  kOptimizedV3 = 3,
  kOptimizedV4 = 4,
  kOptimizedV5 = 5,
};

__device__ __forceinline__ float fp8_e4m3fn_to_float(uint8_t bits) {
  const int sign = bits & 0x80;
  const int exponent = (bits >> 3) & 0x0f;
  const int mantissa = bits & 0x07;

  float value;
  if (exponent == 0) {
    value = mantissa == 0 ? 0.0f : ldexpf(static_cast<float>(mantissa), -9);
  } else {
    value = ldexpf(1.0f + static_cast<float>(mantissa) * 0.125f, exponent - 7);
  }
  return sign ? -value : value;
}

__device__ __forceinline__ float bf16_bytes_to_float(const uint8_t* ptr) {
  const uint16_t low = static_cast<uint16_t>(ptr[0]);
  const uint16_t high = static_cast<uint16_t>(ptr[1]) << 8;
  const uint32_t bits = static_cast<uint32_t>(low | high) << 16;
  return __uint_as_float(bits);
}

__device__ __forceinline__ float load_dsv4_packed_dim(
    const uint8_t* packed,
    int dim) {
  if (dim < kNopeDim) {
    const float q = fp8_e4m3fn_to_float(packed[dim]);
    const int group = dim / kScaleGroup;
    const float scale = exp2f(static_cast<float>(packed[kScaleOffset + group]) - 127.0f);
    return q * scale;
  }

  const int rope_dim = dim - kNopeDim;
  return bf16_bytes_to_float(packed + kNopeDim + rope_dim * 2);
}

__device__ __forceinline__ float load_dsv4_packed_dim_with_scales(
    const uint8_t* packed,
    int dim,
    const float* scales) {
  if (dim < kNopeDim) {
    const float q = fp8_e4m3fn_to_float(packed[dim]);
    return q * scales[dim / kScaleGroup];
  }

  const int rope_dim = dim - kNopeDim;
  return bf16_bytes_to_float(packed + kNopeDim + rope_dim * 2);
}

__device__ __forceinline__ void reduce_sum(float* shared, float& value) {
  shared[threadIdx.x] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  value = shared[0];
  __syncthreads();
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
  value += __shfl_down_sync(0xffffffff, value, 16);
  value += __shfl_down_sync(0xffffffff, value, 8);
  value += __shfl_down_sync(0xffffffff, value, 4);
  value += __shfl_down_sync(0xffffffff, value, 2);
  value += __shfl_down_sync(0xffffffff, value, 1);
  return value;
}

__device__ __forceinline__ void reduce_sum_warp_block(float* shared, float& value) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = warp_reduce_sum(value);
  if (lane == 0) {
    shared[warp] = value;
  }
  __syncthreads();

  value = threadIdx.x < kWarps ? shared[lane] : 0.0f;
  if (warp == 0) {
    value = warp_reduce_sum(value);
  }
  if (threadIdx.x == 0) {
    shared[0] = value;
  }
  __syncthreads();
  value = shared[0];
  __syncthreads();
}

__device__ __forceinline__ void reduce_sum_warp_block_total_slot(
    float* shared,
    float& value) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = warp_reduce_sum(value);
  if (lane == 0) {
    shared[warp] = value;
  }
  __syncthreads();

  value = threadIdx.x < kWarps ? shared[lane] : 0.0f;
  if (warp == 0) {
    value = warp_reduce_sum(value);
    if (lane == 0) {
      shared[kOptimizedV5TotalOffset] = value;
    }
  }
  __syncthreads();
  value = shared[kOptimizedV5TotalOffset];
}

__device__ __forceinline__ void cache_row_scales(
    const uint8_t* packed,
    float* shared) {
  if (threadIdx.x < kScaleCount) {
    shared[kScaleCacheOffset + threadIdx.x] =
        exp2f(static_cast<float>(packed[kScaleOffset + threadIdx.x]) - 127.0f);
  }
  __syncthreads();
}

__device__ __forceinline__ void warp_broadcast_softmax_update(
    float score,
    float& running_max,
    float& running_sum,
    float& old_scale,
    float& row_scale) {
  const int lane = threadIdx.x & 31;
  float new_max = running_max;
  float new_sum = running_sum;
  if (lane == 0) {
    new_max = fmaxf(running_max, score);
    old_scale = running_sum == 0.0f ? 0.0f : expf(running_max - new_max);
    row_scale = expf(score - new_max);
    new_sum = running_sum * old_scale + row_scale;
  }
  new_max = __shfl_sync(0xffffffff, new_max, 0);
  new_sum = __shfl_sync(0xffffffff, new_sum, 0);
  old_scale = __shfl_sync(0xffffffff, old_scale, 0);
  row_scale = __shfl_sync(0xffffffff, row_scale, 0);
  running_max = new_max;
  running_sum = new_sum;
}

__device__ __forceinline__ const uint8_t* selected_row_ptr(
    const uint8_t* cache,
    const int32_t* indices,
    int width,
    int page_size,
    int row_stride,
    int batch,
    int row) {
  int32_t index = indices[batch * width + row];
  if (index < 0) {
    index = 0;
  }
  int page;
  int token;
  if (page_size > 0 && (page_size & (page_size - 1)) == 0) {
    const int shift = __ffs(page_size) - 1;
    page = index >> shift;
    token = index & (page_size - 1);
  } else {
    page = index / page_size;
    token = index - page * page_size;
  }
  return cache + (static_cast<int64_t>(page) * page_size + token) * row_stride;
}

__device__ __forceinline__ void consume_row(
    const uint8_t* packed,
    float softmax_scale,
    float& running_max,
    float& running_sum,
    float& acc0,
    float& acc1,
    float q0,
    float q1,
    int dim0,
    int dim1,
    float* shared) {
  float dot_part = 0.0f;
  const float v0 = dim0 < kHeadDim ? load_dsv4_packed_dim(packed, dim0) : 0.0f;
  const float v1 = dim1 < kHeadDim ? load_dsv4_packed_dim(packed, dim1) : 0.0f;
  if (dim0 < kHeadDim) {
    dot_part += q0 * v0;
  }
  if (dim1 < kHeadDim) {
    dot_part += q1 * v1;
  }

  reduce_sum(shared, dot_part);
  const float score = dot_part * softmax_scale;
  const float new_max = fmaxf(running_max, score);
  const float old_scale = running_sum == 0.0f ? 0.0f : expf(running_max - new_max);
  const float row_scale = expf(score - new_max);
  running_sum = running_sum * old_scale + row_scale;
  running_max = new_max;
  if (dim0 < kHeadDim) {
    acc0 = acc0 * old_scale + row_scale * v0;
  }
  if (dim1 < kHeadDim) {
    acc1 = acc1 * old_scale + row_scale * v1;
  }
}

__device__ __forceinline__ void consume_row_optimized(
    const uint8_t* packed,
    float softmax_scale,
    float& running_max,
    float& running_sum,
    float& acc0,
    float& acc1,
    float q0,
    float q1,
    int dim0,
    int dim1,
    float* shared) {
  float dot_part = 0.0f;
  const float v0 = dim0 < kHeadDim ? load_dsv4_packed_dim(packed, dim0) : 0.0f;
  const float v1 = dim1 < kHeadDim ? load_dsv4_packed_dim(packed, dim1) : 0.0f;
  if (dim0 < kHeadDim) {
    dot_part += q0 * v0;
  }
  if (dim1 < kHeadDim) {
    dot_part += q1 * v1;
  }

  reduce_sum_warp_block(shared, dot_part);
  const float score = dot_part * softmax_scale;
  const float new_max = fmaxf(running_max, score);
  const float old_scale = running_sum == 0.0f ? 0.0f : expf(running_max - new_max);
  const float row_scale = expf(score - new_max);
  running_sum = running_sum * old_scale + row_scale;
  running_max = new_max;
  if (dim0 < kHeadDim) {
    acc0 = acc0 * old_scale + row_scale * v0;
  }
  if (dim1 < kHeadDim) {
    acc1 = acc1 * old_scale + row_scale * v1;
  }
}

__device__ __forceinline__ void consume_row_optimized_v5(
    const uint8_t* packed,
    float softmax_scale,
    float& running_max,
    float& running_sum,
    float& acc0,
    float& acc1,
    float q0,
    float q1,
    int dim0,
    int dim1,
    float* shared) {
  float dot_part = 0.0f;
  const float v0 = dim0 < kHeadDim ? load_dsv4_packed_dim(packed, dim0) : 0.0f;
  const float v1 = dim1 < kHeadDim ? load_dsv4_packed_dim(packed, dim1) : 0.0f;
  if (dim0 < kHeadDim) {
    dot_part += q0 * v0;
  }
  if (dim1 < kHeadDim) {
    dot_part += q1 * v1;
  }

  reduce_sum_warp_block_total_slot(shared, dot_part);
  const float score = dot_part * softmax_scale;
  const float new_max = fmaxf(running_max, score);
  const float old_scale = running_sum == 0.0f ? 0.0f : expf(running_max - new_max);
  const float row_scale = expf(score - new_max);
  running_sum = running_sum * old_scale + row_scale;
  running_max = new_max;
  if (dim0 < kHeadDim) {
    acc0 = acc0 * old_scale + row_scale * v0;
  }
  if (dim1 < kHeadDim) {
    acc1 = acc1 * old_scale + row_scale * v1;
  }
}

__device__ __forceinline__ void consume_row_optimized_v2(
    const uint8_t* packed,
    float softmax_scale,
    float& running_max,
    float& running_sum,
    float& acc0,
    float& acc1,
    float q0,
    float q1,
    int dim0,
    int dim1,
    float* shared) {
  float dot_part = 0.0f;
  const float v0 = dim0 < kHeadDim ? load_dsv4_packed_dim(packed, dim0) : 0.0f;
  const float v1 = dim1 < kHeadDim ? load_dsv4_packed_dim(packed, dim1) : 0.0f;
  if (dim0 < kHeadDim) {
    dot_part += q0 * v0;
  }
  if (dim1 < kHeadDim) {
    dot_part += q1 * v1;
  }

  reduce_sum_warp_block(shared, dot_part);
  float old_scale = 0.0f;
  float row_scale = 0.0f;
  warp_broadcast_softmax_update(
      dot_part * softmax_scale, running_max, running_sum, old_scale, row_scale);
  if (dim0 < kHeadDim) {
    acc0 = acc0 * old_scale + row_scale * v0;
  }
  if (dim1 < kHeadDim) {
    acc1 = acc1 * old_scale + row_scale * v1;
  }
}

__device__ __forceinline__ void consume_row_optimized_v3(
    const uint8_t* packed,
    float softmax_scale,
    float& running_max,
    float& running_sum,
    float& acc0,
    float& acc1,
    float q0,
    float q1,
    int dim0,
    int dim1,
    float* shared) {
  cache_row_scales(packed, shared);
  const float* scales = shared + kScaleCacheOffset;

  float dot_part = 0.0f;
  const float v0 =
      dim0 < kHeadDim ? load_dsv4_packed_dim_with_scales(packed, dim0, scales) : 0.0f;
  const float v1 =
      dim1 < kHeadDim ? load_dsv4_packed_dim_with_scales(packed, dim1, scales) : 0.0f;
  if (dim0 < kHeadDim) {
    dot_part += q0 * v0;
  }
  if (dim1 < kHeadDim) {
    dot_part += q1 * v1;
  }

  reduce_sum_warp_block(shared, dot_part);
  float old_scale = 0.0f;
  float row_scale = 0.0f;
  warp_broadcast_softmax_update(
      dot_part * softmax_scale, running_max, running_sum, old_scale, row_scale);
  if (dim0 < kHeadDim) {
    acc0 = acc0 * old_scale + row_scale * v0;
  }
  if (dim1 < kHeadDim) {
    acc1 = acc1 * old_scale + row_scale * v1;
  }
}

__device__ __forceinline__ void consume_sink_broadcast(
    float sink,
    float& running_max,
    float& running_sum,
    float& acc0,
    float& acc1,
    int dim0,
    int dim1) {
  float old_scale = 0.0f;
  float sink_scale = 0.0f;
  warp_broadcast_softmax_update(sink, running_max, running_sum, old_scale, sink_scale);
  if (dim0 < kHeadDim) {
    acc0 *= old_scale;
  }
  if (dim1 < kHeadDim) {
    acc1 *= old_scale;
  }
}

__device__ __forceinline__ void stage_row_to_shared(
    const uint8_t* packed,
    float* kv_shared) {
  const int tid = threadIdx.x;
  const int dim0 = tid;
  const int dim1 = tid + kThreads;
  if (dim0 < kHeadDim) {
    kv_shared[dim0] = load_dsv4_packed_dim(packed, dim0);
  }
  if (dim1 < kHeadDim) {
    kv_shared[dim1] = load_dsv4_packed_dim(packed, dim1);
  }
  __syncthreads();
}

__device__ __forceinline__ float reduce_sum_head_group(
    float value,
    float* reduce_shared,
    int head_slot,
    int head_warp,
    int lane) {
  const int warp = threadIdx.x >> 5;
  value = warp_reduce_sum(value);
  if (lane == 0) {
    reduce_shared[warp] = value;
  }
  __syncthreads();

  const int first_warp = head_slot * kV4WarpsPerHead;
  float total = 0.0f;
  if (head_warp == 0) {
    total = lane < kV4WarpsPerHead ? reduce_shared[first_warp + lane] : 0.0f;
    total = warp_reduce_sum(total);
    if (lane == 0) {
      reduce_shared[first_warp] = total;
    }
  }
  __syncthreads();
  total = reduce_shared[first_warp];
  __syncthreads();
  return total;
}

__device__ __forceinline__ void consume_staged_row_warp_head(
    const float* kv_shared,
    float* reduce_shared,
    float softmax_scale,
    float& running_max,
    float& running_sum,
    float* acc,
    const float* q_vals,
    int head_slot,
    int head_warp,
    int head_thread,
    int lane) {
  float dot_part = 0.0f;
#pragma unroll
  for (int slot = 0; slot < kV4DimsPerThread; ++slot) {
    const int dim = head_thread + slot * kV4HeadThreads;
    const float value = kv_shared[dim];
    dot_part += q_vals[slot] * value;
  }

  dot_part = reduce_sum_head_group(dot_part, reduce_shared, head_slot, head_warp, lane);
  const float score = dot_part * softmax_scale;
  const float new_max = fmaxf(running_max, score);
  const float old_scale = running_sum == 0.0f ? 0.0f : expf(running_max - new_max);
  const float row_scale = expf(score - new_max);
  running_sum = running_sum * old_scale + row_scale;
  running_max = new_max;

#pragma unroll
  for (int slot = 0; slot < kV4DimsPerThread; ++slot) {
    const int dim = head_thread + slot * kV4HeadThreads;
    acc[slot] = acc[slot] * old_scale + row_scale * kv_shared[dim];
  }
}

__global__ void ds4_cuda_reference_attention_kernel(
    const __nv_bfloat16* __restrict__ q,
    const uint8_t* __restrict__ swa_cache,
    const int32_t* __restrict__ swa_indices,
    const int32_t* __restrict__ swa_lengths,
    int swa_width,
    int swa_page_size,
    int swa_row_stride,
    const float* __restrict__ attn_sink,
    bool has_sink,
    const uint8_t* __restrict__ extra_cache,
    const int32_t* __restrict__ extra_indices,
    const int32_t* __restrict__ extra_lengths,
    int extra_width,
    int extra_page_size,
    int extra_row_stride,
    bool has_extra,
    float softmax_scale,
    int batch_size,
    int num_heads,
    __nv_bfloat16* __restrict__ out) {
  const int batch = blockIdx.x;
  const int head = blockIdx.y;
  if (batch >= batch_size || head >= num_heads) {
    return;
  }

  __shared__ float shared[kThreads];

  const int tid = threadIdx.x;
  const int dim0 = tid;
  const int dim1 = tid + kThreads;
  const int64_t q_base = (static_cast<int64_t>(batch) * num_heads + head) * kHeadDim;

  const float q0 = dim0 < kHeadDim ? __bfloat162float(q[q_base + dim0]) : 0.0f;
  const float q1 = dim1 < kHeadDim ? __bfloat162float(q[q_base + dim1]) : 0.0f;

  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float acc0 = 0.0f;
  float acc1 = 0.0f;

  int swa_len = swa_lengths[batch];
  if (swa_len < 0) {
    swa_len = 0;
  }
  if (swa_len > swa_width) {
    swa_len = swa_width;
  }
  for (int row = 0; row < swa_len; ++row) {
    const uint8_t* packed = selected_row_ptr(
        swa_cache, swa_indices, swa_width, swa_page_size, swa_row_stride, batch, row);
    consume_row(
        packed,
        softmax_scale,
        running_max,
        running_sum,
        acc0,
        acc1,
        q0,
        q1,
        dim0,
        dim1,
        shared);
  }

  if (has_extra) {
    int extra_len = extra_lengths[batch];
    if (extra_len < 0) {
      extra_len = 0;
    }
    if (extra_len > extra_width) {
      extra_len = extra_width;
    }
    for (int row = 0; row < extra_len; ++row) {
      const uint8_t* packed = selected_row_ptr(
          extra_cache,
          extra_indices,
          extra_width,
          extra_page_size,
          extra_row_stride,
          batch,
          row);
      consume_row(
          packed,
          softmax_scale,
          running_max,
          running_sum,
          acc0,
          acc1,
          q0,
          q1,
          dim0,
          dim1,
          shared);
    }
  }

  if (has_sink) {
    const float sink = attn_sink[head];
    const float new_max = fmaxf(running_max, sink);
    const float old_scale = running_sum == 0.0f ? 0.0f : expf(running_max - new_max);
    const float sink_scale = expf(sink - new_max);
    running_sum = running_sum * old_scale + sink_scale;
    running_max = new_max;
    if (dim0 < kHeadDim) {
      acc0 *= old_scale;
    }
    if (dim1 < kHeadDim) {
      acc1 *= old_scale;
    }
  }

  const float inv_sum = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
  if (dim0 < kHeadDim) {
    out[q_base + dim0] = __float2bfloat16(acc0 * inv_sum);
  }
  if (dim1 < kHeadDim) {
    out[q_base + dim1] = __float2bfloat16(acc1 * inv_sum);
  }
}

template <bool SeparateTotalSlot>
__global__ void ds4_cuda_optimized_attention_kernel(
    const __nv_bfloat16* __restrict__ q,
    const uint8_t* __restrict__ swa_cache,
    const int32_t* __restrict__ swa_indices,
    const int32_t* __restrict__ swa_lengths,
    int swa_width,
    int swa_page_size,
    int swa_row_stride,
    const float* __restrict__ attn_sink,
    bool has_sink,
    const uint8_t* __restrict__ extra_cache,
    const int32_t* __restrict__ extra_indices,
    const int32_t* __restrict__ extra_lengths,
    int extra_width,
    int extra_page_size,
    int extra_row_stride,
    bool has_extra,
    float softmax_scale,
    int batch_size,
    int num_heads,
    __nv_bfloat16* __restrict__ out) {
  const int batch = blockIdx.x;
  const int head = blockIdx.y;
  if (batch >= batch_size || head >= num_heads) {
    return;
  }

  __shared__ float shared[SeparateTotalSlot ? kOptimizedV5SharedFloats : kWarps];

  const int tid = threadIdx.x;
  const int dim0 = tid;
  const int dim1 = tid + kThreads;
  const int64_t q_base = (static_cast<int64_t>(batch) * num_heads + head) * kHeadDim;

  const float q0 = dim0 < kHeadDim ? __bfloat162float(q[q_base + dim0]) : 0.0f;
  const float q1 = dim1 < kHeadDim ? __bfloat162float(q[q_base + dim1]) : 0.0f;

  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float acc0 = 0.0f;
  float acc1 = 0.0f;

  int swa_len = swa_lengths[batch];
  if (swa_len < 0) {
    swa_len = 0;
  }
  if (swa_len > swa_width) {
    swa_len = swa_width;
  }
  for (int row = 0; row < swa_len; ++row) {
    const uint8_t* packed = selected_row_ptr(
        swa_cache, swa_indices, swa_width, swa_page_size, swa_row_stride, batch, row);
    if constexpr (SeparateTotalSlot) {
      consume_row_optimized_v5(
          packed,
          softmax_scale,
          running_max,
          running_sum,
          acc0,
          acc1,
          q0,
          q1,
          dim0,
          dim1,
          shared);
    } else {
      consume_row_optimized(
          packed,
          softmax_scale,
          running_max,
          running_sum,
          acc0,
          acc1,
          q0,
          q1,
          dim0,
          dim1,
          shared);
    }
  }

  if (has_extra) {
    int extra_len = extra_lengths[batch];
    if (extra_len < 0) {
      extra_len = 0;
    }
    if (extra_len > extra_width) {
      extra_len = extra_width;
    }
    for (int row = 0; row < extra_len; ++row) {
      const uint8_t* packed = selected_row_ptr(
          extra_cache,
          extra_indices,
          extra_width,
          extra_page_size,
          extra_row_stride,
          batch,
          row);
      if constexpr (SeparateTotalSlot) {
        consume_row_optimized_v5(
            packed,
            softmax_scale,
            running_max,
            running_sum,
            acc0,
            acc1,
            q0,
            q1,
            dim0,
            dim1,
            shared);
      } else {
        consume_row_optimized(
            packed,
            softmax_scale,
            running_max,
            running_sum,
            acc0,
            acc1,
            q0,
            q1,
            dim0,
            dim1,
            shared);
      }
    }
  }

  if (has_sink) {
    const float sink = attn_sink[head];
    const float new_max = fmaxf(running_max, sink);
    const float old_scale = running_sum == 0.0f ? 0.0f : expf(running_max - new_max);
    const float sink_scale = expf(sink - new_max);
    running_sum = running_sum * old_scale + sink_scale;
    running_max = new_max;
    if (dim0 < kHeadDim) {
      acc0 *= old_scale;
    }
    if (dim1 < kHeadDim) {
      acc1 *= old_scale;
    }
  }

  const float inv_sum = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
  if (dim0 < kHeadDim) {
    out[q_base + dim0] = __float2bfloat16(acc0 * inv_sum);
  }
  if (dim1 < kHeadDim) {
    out[q_base + dim1] = __float2bfloat16(acc1 * inv_sum);
  }
}

template <bool CacheScales>
__global__ void ds4_cuda_optimized_v2_attention_kernel(
    const __nv_bfloat16* __restrict__ q,
    const uint8_t* __restrict__ swa_cache,
    const int32_t* __restrict__ swa_indices,
    const int32_t* __restrict__ swa_lengths,
    int swa_width,
    int swa_page_size,
    int swa_row_stride,
    const float* __restrict__ attn_sink,
    bool has_sink,
    const uint8_t* __restrict__ extra_cache,
    const int32_t* __restrict__ extra_indices,
    const int32_t* __restrict__ extra_lengths,
    int extra_width,
    int extra_page_size,
    int extra_row_stride,
    bool has_extra,
    float softmax_scale,
    int batch_size,
    int num_heads,
    __nv_bfloat16* __restrict__ out) {
  const int batch = blockIdx.x;
  const int head = blockIdx.y;
  if (batch >= batch_size || head >= num_heads) {
    return;
  }

  __shared__ float shared[CacheScales ? kOptimizedV3SharedFloats : kOptimizedV2SharedFloats];

  const int tid = threadIdx.x;
  const int dim0 = tid;
  const int dim1 = tid + kThreads;
  const int64_t q_base = (static_cast<int64_t>(batch) * num_heads + head) * kHeadDim;

  const float q0 = dim0 < kHeadDim ? __bfloat162float(q[q_base + dim0]) : 0.0f;
  const float q1 = dim1 < kHeadDim ? __bfloat162float(q[q_base + dim1]) : 0.0f;

  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float acc0 = 0.0f;
  float acc1 = 0.0f;

  int swa_len = swa_lengths[batch];
  if (swa_len < 0) {
    swa_len = 0;
  }
  if (swa_len > swa_width) {
    swa_len = swa_width;
  }
  for (int row = 0; row < swa_len; ++row) {
    const uint8_t* packed = selected_row_ptr(
        swa_cache, swa_indices, swa_width, swa_page_size, swa_row_stride, batch, row);
    if constexpr (CacheScales) {
      consume_row_optimized_v3(
          packed,
          softmax_scale,
          running_max,
          running_sum,
          acc0,
          acc1,
          q0,
          q1,
          dim0,
          dim1,
          shared);
    } else {
      consume_row_optimized_v2(
          packed,
          softmax_scale,
          running_max,
          running_sum,
          acc0,
          acc1,
          q0,
          q1,
          dim0,
          dim1,
          shared);
    }
  }

  if (has_extra) {
    int extra_len = extra_lengths[batch];
    if (extra_len < 0) {
      extra_len = 0;
    }
    if (extra_len > extra_width) {
      extra_len = extra_width;
    }
    for (int row = 0; row < extra_len; ++row) {
      const uint8_t* packed = selected_row_ptr(
          extra_cache,
          extra_indices,
          extra_width,
          extra_page_size,
          extra_row_stride,
          batch,
          row);
      if constexpr (CacheScales) {
        consume_row_optimized_v3(
            packed,
            softmax_scale,
            running_max,
            running_sum,
            acc0,
            acc1,
            q0,
            q1,
            dim0,
            dim1,
            shared);
      } else {
        consume_row_optimized_v2(
            packed,
            softmax_scale,
            running_max,
            running_sum,
            acc0,
            acc1,
            q0,
            q1,
            dim0,
            dim1,
            shared);
      }
    }
  }

  if (has_sink) {
    consume_sink_broadcast(
        attn_sink[head], running_max, running_sum, acc0, acc1, dim0, dim1);
  }

  const float inv_sum = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
  if (dim0 < kHeadDim) {
    out[q_base + dim0] = __float2bfloat16(acc0 * inv_sum);
  }
  if (dim1 < kHeadDim) {
    out[q_base + dim1] = __float2bfloat16(acc1 * inv_sum);
  }
}

__global__ void ds4_cuda_grouped_head_attention_kernel(
    const __nv_bfloat16* __restrict__ q,
    const uint8_t* __restrict__ swa_cache,
    const int32_t* __restrict__ swa_indices,
    const int32_t* __restrict__ swa_lengths,
    int swa_width,
    int swa_page_size,
    int swa_row_stride,
    const float* __restrict__ attn_sink,
    bool has_sink,
    const uint8_t* __restrict__ extra_cache,
    const int32_t* __restrict__ extra_indices,
    const int32_t* __restrict__ extra_lengths,
    int extra_width,
    int extra_page_size,
    int extra_row_stride,
    bool has_extra,
    float softmax_scale,
    int batch_size,
    int num_heads,
    __nv_bfloat16* __restrict__ out) {
  const int batch = blockIdx.x;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int head_slot = warp / kV4WarpsPerHead;
  const int head_warp = warp - head_slot * kV4WarpsPerHead;
  const int head_thread = head_warp * 32 + lane;
  if (batch >= batch_size) {
    return;
  }
  const int head = blockIdx.y * kV4HeadsPerBlock + head_slot;
  const bool active_head = head < num_heads;

  __shared__ float kv_shared[kHeadDim];
  __shared__ float reduce_shared[kWarps];

  const int64_t q_base = (static_cast<int64_t>(batch) * num_heads + head) * kHeadDim;
  float q_vals[kV4DimsPerThread];
  float acc[kV4DimsPerThread];
#pragma unroll
  for (int slot = 0; slot < kV4DimsPerThread; ++slot) {
    const int dim = head_thread + slot * kV4HeadThreads;
    q_vals[slot] = active_head ? __bfloat162float(q[q_base + dim]) : 0.0f;
    acc[slot] = 0.0f;
  }

  float running_max = -INFINITY;
  float running_sum = 0.0f;

  int swa_len = swa_lengths[batch];
  if (swa_len < 0) {
    swa_len = 0;
  }
  if (swa_len > swa_width) {
    swa_len = swa_width;
  }
  for (int row = 0; row < swa_len; ++row) {
    const uint8_t* packed = selected_row_ptr(
        swa_cache, swa_indices, swa_width, swa_page_size, swa_row_stride, batch, row);
    stage_row_to_shared(packed, kv_shared);
    consume_staged_row_warp_head(
        kv_shared,
        reduce_shared,
        softmax_scale,
        running_max,
        running_sum,
        acc,
        q_vals,
        head_slot,
        head_warp,
        head_thread,
        lane);
    __syncthreads();
  }

  if (has_extra) {
    int extra_len = extra_lengths[batch];
    if (extra_len < 0) {
      extra_len = 0;
    }
    if (extra_len > extra_width) {
      extra_len = extra_width;
    }
    for (int row = 0; row < extra_len; ++row) {
      const uint8_t* packed = selected_row_ptr(
          extra_cache,
          extra_indices,
          extra_width,
          extra_page_size,
          extra_row_stride,
          batch,
          row);
      stage_row_to_shared(packed, kv_shared);
      consume_staged_row_warp_head(
          kv_shared,
          reduce_shared,
          softmax_scale,
          running_max,
          running_sum,
          acc,
          q_vals,
          head_slot,
          head_warp,
          head_thread,
          lane);
      __syncthreads();
    }
  }

  if (active_head && has_sink) {
    const float sink = attn_sink[head];
    const float new_max = fmaxf(running_max, sink);
    const float old_scale = running_sum == 0.0f ? 0.0f : expf(running_max - new_max);
    const float sink_scale = expf(sink - new_max);
    running_sum = running_sum * old_scale + sink_scale;
    running_max = new_max;
#pragma unroll
    for (int slot = 0; slot < kV4DimsPerThread; ++slot) {
      acc[slot] *= old_scale;
    }
  }

  const float inv_sum = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
#pragma unroll
  for (int slot = 0; slot < kV4DimsPerThread; ++slot) {
    const int dim = head_thread + slot * kV4HeadThreads;
    if (active_head) {
      out[q_base + dim] = __float2bfloat16(acc[slot] * inv_sum);
    }
  }
}

int64_t flattened_batch(torch::Tensor q) {
  TORCH_CHECK(q.dim() >= 3, "q must have shape [..., heads, 512], got ", q.sizes());
  TORCH_CHECK(q.size(-1) == kHeadDim, "q head dim must be 512, got ", q.size(-1));
  const int64_t num_heads = q.size(-2);
  TORCH_CHECK(num_heads > 0, "q must have at least one head");
  const int64_t denom = num_heads * kHeadDim;
  TORCH_CHECK(q.numel() % denom == 0, "q shape is not flattenable as [B, H, 512]");
  return q.numel() / denom;
}

torch::Tensor launch_ds4_cuda_attention(
    torch::Tensor q,
    torch::Tensor swa_k_cache,
    torch::Tensor swa_indices,
    torch::Tensor swa_topk_lengths,
    int64_t swa_page_size,
    double softmax_scale,
    torch::Tensor attn_sink,
    torch::Tensor extra_k_cache,
    torch::Tensor extra_indices,
    torch::Tensor extra_topk_lengths,
    int64_t extra_page_size,
    AttentionVariant variant) {
  TORCH_CHECK(q.is_cuda(), "q must be CUDA");
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "q must be bfloat16");
  TORCH_CHECK(swa_k_cache.is_cuda(), "swa_k_cache must be CUDA");
  TORCH_CHECK(swa_k_cache.scalar_type() == torch::kUInt8, "swa_k_cache must be uint8");
  TORCH_CHECK(swa_indices.is_cuda(), "swa_indices must be CUDA");
  TORCH_CHECK(swa_topk_lengths.is_cuda(), "swa_topk_lengths must be CUDA");
  TORCH_CHECK(swa_indices.scalar_type() == torch::kInt32, "swa_indices must be int32");
  TORCH_CHECK(swa_topk_lengths.scalar_type() == torch::kInt32, "swa_topk_lengths must be int32");
  TORCH_CHECK(swa_k_cache.dim() == 4, "swa_k_cache must be [pages, page, 1, bytes]");
  TORCH_CHECK(swa_k_cache.size(2) == 1, "swa_k_cache singleton dim must be 1");
  TORCH_CHECK(swa_k_cache.size(3) >= kPackedBytes, "swa_k_cache packed dim is too small");
  TORCH_CHECK(swa_indices.dim() >= 2, "swa_indices must have at least 2 dims");

  q = q.contiguous();
  swa_k_cache = swa_k_cache.contiguous();
  swa_indices = swa_indices.contiguous();
  swa_topk_lengths = swa_topk_lengths.contiguous();
  attn_sink = attn_sink.contiguous();
  extra_k_cache = extra_k_cache.contiguous();
  extra_indices = extra_indices.contiguous();
  extra_topk_lengths = extra_topk_lengths.contiguous();

  const auto batch_size = flattened_batch(q);
  const auto num_heads = q.size(-2);
  const auto swa_width = swa_indices.size(-1);
  TORCH_CHECK(swa_topk_lengths.numel() == batch_size, "swa lengths must match flattened q batch");
  TORCH_CHECK(swa_indices.numel() / swa_width == batch_size, "swa indices must match flattened q batch");

  const bool has_sink = attn_sink.numel() > 0;
  if (has_sink) {
    TORCH_CHECK(attn_sink.is_cuda(), "attn_sink must be CUDA when provided");
    TORCH_CHECK(attn_sink.scalar_type() == torch::kFloat32, "attn_sink must be float32");
    TORCH_CHECK(attn_sink.numel() == num_heads, "attn_sink must have one value per head");
  }

  const bool has_extra = extra_k_cache.numel() > 0;
  int64_t extra_width = 0;
  if (has_extra) {
    TORCH_CHECK(extra_k_cache.is_cuda(), "extra_k_cache must be CUDA when provided");
    TORCH_CHECK(extra_k_cache.scalar_type() == torch::kUInt8, "extra_k_cache must be uint8");
    TORCH_CHECK(extra_indices.is_cuda(), "extra_indices must be CUDA when provided");
    TORCH_CHECK(extra_topk_lengths.is_cuda(), "extra_topk_lengths must be CUDA when provided");
    TORCH_CHECK(extra_indices.scalar_type() == torch::kInt32, "extra_indices must be int32");
    TORCH_CHECK(extra_topk_lengths.scalar_type() == torch::kInt32, "extra_topk_lengths must be int32");
    TORCH_CHECK(extra_k_cache.dim() == 4, "extra_k_cache must be [pages, page, 1, bytes]");
    TORCH_CHECK(extra_k_cache.size(2) == 1, "extra_k_cache singleton dim must be 1");
    TORCH_CHECK(extra_k_cache.size(3) >= kPackedBytes, "extra_k_cache packed dim is too small");
    TORCH_CHECK(extra_indices.dim() >= 2, "extra_indices must have at least 2 dims");
    extra_width = extra_indices.size(-1);
    TORCH_CHECK(extra_topk_lengths.numel() == batch_size, "extra lengths must match flattened q batch");
    TORCH_CHECK(extra_indices.numel() / extra_width == batch_size, "extra indices must match flattened q batch");
  }

  c10::cuda::CUDAGuard device_guard(q.device());
  auto out = torch::empty_like(q);

  dim3 grid(batch_size, num_heads);
  dim3 grouped_grid(
      static_cast<unsigned int>(batch_size),
      static_cast<unsigned int>((num_heads + kV4HeadsPerBlock - 1) / kV4HeadsPerBlock));
  dim3 block(kThreads);
  switch (variant) {
    case AttentionVariant::kReference:
      ds4_cuda_reference_attention_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
          swa_k_cache.data_ptr<uint8_t>(),
          swa_indices.data_ptr<int32_t>(),
          swa_topk_lengths.data_ptr<int32_t>(),
          static_cast<int>(swa_width),
          static_cast<int>(swa_page_size),
          static_cast<int>(swa_k_cache.size(3)),
          has_sink ? attn_sink.data_ptr<float>() : nullptr,
          has_sink,
          has_extra ? extra_k_cache.data_ptr<uint8_t>() : nullptr,
          has_extra ? extra_indices.data_ptr<int32_t>() : nullptr,
          has_extra ? extra_topk_lengths.data_ptr<int32_t>() : nullptr,
          static_cast<int>(extra_width),
          static_cast<int>(extra_page_size),
          has_extra ? static_cast<int>(extra_k_cache.size(3)) : 0,
          has_extra,
          static_cast<float>(softmax_scale),
          static_cast<int>(batch_size),
          static_cast<int>(num_heads),
          reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()));
      break;
    case AttentionVariant::kOptimizedV1:
      ds4_cuda_optimized_attention_kernel<false>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
              reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
              swa_k_cache.data_ptr<uint8_t>(),
              swa_indices.data_ptr<int32_t>(),
              swa_topk_lengths.data_ptr<int32_t>(),
              static_cast<int>(swa_width),
              static_cast<int>(swa_page_size),
              static_cast<int>(swa_k_cache.size(3)),
              has_sink ? attn_sink.data_ptr<float>() : nullptr,
              has_sink,
              has_extra ? extra_k_cache.data_ptr<uint8_t>() : nullptr,
              has_extra ? extra_indices.data_ptr<int32_t>() : nullptr,
              has_extra ? extra_topk_lengths.data_ptr<int32_t>() : nullptr,
              static_cast<int>(extra_width),
              static_cast<int>(extra_page_size),
              has_extra ? static_cast<int>(extra_k_cache.size(3)) : 0,
              has_extra,
              static_cast<float>(softmax_scale),
              static_cast<int>(batch_size),
              static_cast<int>(num_heads),
              reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()));
      break;
    case AttentionVariant::kOptimizedV2:
      ds4_cuda_optimized_v2_attention_kernel<false>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
              reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
              swa_k_cache.data_ptr<uint8_t>(),
              swa_indices.data_ptr<int32_t>(),
              swa_topk_lengths.data_ptr<int32_t>(),
              static_cast<int>(swa_width),
              static_cast<int>(swa_page_size),
              static_cast<int>(swa_k_cache.size(3)),
              has_sink ? attn_sink.data_ptr<float>() : nullptr,
              has_sink,
              has_extra ? extra_k_cache.data_ptr<uint8_t>() : nullptr,
              has_extra ? extra_indices.data_ptr<int32_t>() : nullptr,
              has_extra ? extra_topk_lengths.data_ptr<int32_t>() : nullptr,
              static_cast<int>(extra_width),
              static_cast<int>(extra_page_size),
              has_extra ? static_cast<int>(extra_k_cache.size(3)) : 0,
              has_extra,
              static_cast<float>(softmax_scale),
              static_cast<int>(batch_size),
              static_cast<int>(num_heads),
              reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()));
      break;
    case AttentionVariant::kOptimizedV3:
      ds4_cuda_optimized_v2_attention_kernel<true>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
              reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
              swa_k_cache.data_ptr<uint8_t>(),
              swa_indices.data_ptr<int32_t>(),
              swa_topk_lengths.data_ptr<int32_t>(),
              static_cast<int>(swa_width),
              static_cast<int>(swa_page_size),
              static_cast<int>(swa_k_cache.size(3)),
              has_sink ? attn_sink.data_ptr<float>() : nullptr,
              has_sink,
              has_extra ? extra_k_cache.data_ptr<uint8_t>() : nullptr,
              has_extra ? extra_indices.data_ptr<int32_t>() : nullptr,
              has_extra ? extra_topk_lengths.data_ptr<int32_t>() : nullptr,
              static_cast<int>(extra_width),
              static_cast<int>(extra_page_size),
              has_extra ? static_cast<int>(extra_k_cache.size(3)) : 0,
              has_extra,
              static_cast<float>(softmax_scale),
              static_cast<int>(batch_size),
              static_cast<int>(num_heads),
              reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()));
      break;
    case AttentionVariant::kOptimizedV4:
      ds4_cuda_grouped_head_attention_kernel
          <<<grouped_grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
              reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
              swa_k_cache.data_ptr<uint8_t>(),
              swa_indices.data_ptr<int32_t>(),
              swa_topk_lengths.data_ptr<int32_t>(),
              static_cast<int>(swa_width),
              static_cast<int>(swa_page_size),
              static_cast<int>(swa_k_cache.size(3)),
              has_sink ? attn_sink.data_ptr<float>() : nullptr,
              has_sink,
              has_extra ? extra_k_cache.data_ptr<uint8_t>() : nullptr,
              has_extra ? extra_indices.data_ptr<int32_t>() : nullptr,
              has_extra ? extra_topk_lengths.data_ptr<int32_t>() : nullptr,
              static_cast<int>(extra_width),
              static_cast<int>(extra_page_size),
              has_extra ? static_cast<int>(extra_k_cache.size(3)) : 0,
              has_extra,
              static_cast<float>(softmax_scale),
              static_cast<int>(batch_size),
              static_cast<int>(num_heads),
              reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()));
      break;
    case AttentionVariant::kOptimizedV5:
      ds4_cuda_optimized_attention_kernel<true>
          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
              reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
              swa_k_cache.data_ptr<uint8_t>(),
              swa_indices.data_ptr<int32_t>(),
              swa_topk_lengths.data_ptr<int32_t>(),
              static_cast<int>(swa_width),
              static_cast<int>(swa_page_size),
              static_cast<int>(swa_k_cache.size(3)),
              has_sink ? attn_sink.data_ptr<float>() : nullptr,
              has_sink,
              has_extra ? extra_k_cache.data_ptr<uint8_t>() : nullptr,
              has_extra ? extra_indices.data_ptr<int32_t>() : nullptr,
              has_extra ? extra_topk_lengths.data_ptr<int32_t>() : nullptr,
              static_cast<int>(extra_width),
              static_cast<int>(extra_page_size),
              has_extra ? static_cast<int>(extra_k_cache.size(3)) : 0,
              has_extra,
              static_cast<float>(softmax_scale),
              static_cast<int>(batch_size),
              static_cast<int>(num_heads),
              reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()));
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace

torch::Tensor ds4_cuda_reference_attention(
    torch::Tensor q,
    torch::Tensor swa_k_cache,
    torch::Tensor swa_indices,
    torch::Tensor swa_topk_lengths,
    int64_t swa_page_size,
    double softmax_scale,
    torch::Tensor attn_sink,
    torch::Tensor extra_k_cache,
    torch::Tensor extra_indices,
    torch::Tensor extra_topk_lengths,
    int64_t extra_page_size) {
  return launch_ds4_cuda_attention(
      q,
      swa_k_cache,
      swa_indices,
      swa_topk_lengths,
      swa_page_size,
      softmax_scale,
      attn_sink,
      extra_k_cache,
      extra_indices,
      extra_topk_lengths,
      extra_page_size,
      AttentionVariant::kReference);
}

torch::Tensor ds4_cuda_optimized_attention(
    torch::Tensor q,
    torch::Tensor swa_k_cache,
    torch::Tensor swa_indices,
    torch::Tensor swa_topk_lengths,
    int64_t swa_page_size,
    double softmax_scale,
    torch::Tensor attn_sink,
    torch::Tensor extra_k_cache,
    torch::Tensor extra_indices,
    torch::Tensor extra_topk_lengths,
    int64_t extra_page_size) {
  return launch_ds4_cuda_attention(
      q,
      swa_k_cache,
      swa_indices,
      swa_topk_lengths,
      swa_page_size,
      softmax_scale,
      attn_sink,
      extra_k_cache,
      extra_indices,
      extra_topk_lengths,
      extra_page_size,
      AttentionVariant::kOptimizedV1);
}

torch::Tensor ds4_cuda_optimized_v2_attention(
    torch::Tensor q,
    torch::Tensor swa_k_cache,
    torch::Tensor swa_indices,
    torch::Tensor swa_topk_lengths,
    int64_t swa_page_size,
    double softmax_scale,
    torch::Tensor attn_sink,
    torch::Tensor extra_k_cache,
    torch::Tensor extra_indices,
    torch::Tensor extra_topk_lengths,
    int64_t extra_page_size) {
  return launch_ds4_cuda_attention(
      q,
      swa_k_cache,
      swa_indices,
      swa_topk_lengths,
      swa_page_size,
      softmax_scale,
      attn_sink,
      extra_k_cache,
      extra_indices,
      extra_topk_lengths,
      extra_page_size,
      AttentionVariant::kOptimizedV2);
}

torch::Tensor ds4_cuda_optimized_v3_attention(
    torch::Tensor q,
    torch::Tensor swa_k_cache,
    torch::Tensor swa_indices,
    torch::Tensor swa_topk_lengths,
    int64_t swa_page_size,
    double softmax_scale,
    torch::Tensor attn_sink,
    torch::Tensor extra_k_cache,
    torch::Tensor extra_indices,
    torch::Tensor extra_topk_lengths,
    int64_t extra_page_size) {
  return launch_ds4_cuda_attention(
      q,
      swa_k_cache,
      swa_indices,
      swa_topk_lengths,
      swa_page_size,
      softmax_scale,
      attn_sink,
      extra_k_cache,
      extra_indices,
      extra_topk_lengths,
      extra_page_size,
      AttentionVariant::kOptimizedV3);
}

torch::Tensor ds4_cuda_optimized_v4_attention(
    torch::Tensor q,
    torch::Tensor swa_k_cache,
    torch::Tensor swa_indices,
    torch::Tensor swa_topk_lengths,
    int64_t swa_page_size,
    double softmax_scale,
    torch::Tensor attn_sink,
    torch::Tensor extra_k_cache,
    torch::Tensor extra_indices,
    torch::Tensor extra_topk_lengths,
    int64_t extra_page_size) {
  return launch_ds4_cuda_attention(
      q,
      swa_k_cache,
      swa_indices,
      swa_topk_lengths,
      swa_page_size,
      softmax_scale,
      attn_sink,
      extra_k_cache,
      extra_indices,
      extra_topk_lengths,
      extra_page_size,
      AttentionVariant::kOptimizedV4);
}

torch::Tensor ds4_cuda_optimized_v5_attention(
    torch::Tensor q,
    torch::Tensor swa_k_cache,
    torch::Tensor swa_indices,
    torch::Tensor swa_topk_lengths,
    int64_t swa_page_size,
    double softmax_scale,
    torch::Tensor attn_sink,
    torch::Tensor extra_k_cache,
    torch::Tensor extra_indices,
    torch::Tensor extra_topk_lengths,
    int64_t extra_page_size) {
  return launch_ds4_cuda_attention(
      q,
      swa_k_cache,
      swa_indices,
      swa_topk_lengths,
      swa_page_size,
      softmax_scale,
      attn_sink,
      extra_k_cache,
      extra_indices,
      extra_topk_lengths,
      extra_page_size,
      AttentionVariant::kOptimizedV5);
}
