#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
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
constexpr int kScoreTileM = 16;
constexpr int kScoreTileN = 16;
constexpr int kScoreTileK = 16;
constexpr int kScoreTileElems = kScoreTileM * kScoreTileN;
constexpr int kScoreKTileCount = kHeadDim / kScoreTileK;
static_assert(
    kHeadDim % kScoreTileK == 0,
    "DS4 score kernel expects the head dimension to divide into score-K tiles");
constexpr int kScoreWarpsPerBlock = 4;
constexpr int kScoreRowsPerBlock = kScoreTileN * kScoreWarpsPerBlock;
constexpr int kScoreThreads = 32 * kScoreWarpsPerBlock;
constexpr int kV7Threads = 512;
constexpr int kV7AccElems = kScoreTileM * kHeadDim;
constexpr int kV8Threads = kV7Threads;
constexpr int kV11DimGroups = 4;
constexpr int kV11WarpsPerBlock = kScoreWarpsPerBlock * kV11DimGroups;
constexpr int kV11DimRounds = kHeadDim / (kScoreTileN * kV11DimGroups);
constexpr int kV15DimRoundGroups = 4;
constexpr int kV15DimOuterRounds = kV11DimRounds / kV15DimRoundGroups;
static_assert(
    kV11DimRounds % kV15DimRoundGroups == 0,
    "v15 expects dimension rounds to divide evenly into round groups");
constexpr int kReduceScaleCacheMaxRowTiles = 16;
constexpr int kPartialProfileRowSetup = 0;
constexpr int kPartialProfileScaleCache = 1;
constexpr int kPartialProfileQk = 2;
constexpr int kPartialProfileSoftmax = 3;
constexpr int kPartialProfilePCache = 4;
constexpr int kPartialProfilePvMma = 5;
constexpr int kPartialProfilePartialStore = 6;
constexpr int kPartialProfileBlocks = 7;
constexpr int kPartialProfileQkStagingWall = 8;
constexpr int kPartialProfileQkMmaWall = 9;
constexpr int kPartialProfileQkScoreStoreWall = 10;
constexpr int kPartialProfileQkBlockBarrier = 11;
constexpr int kPartialProfileQkThreadQLoad = 12;
constexpr int kPartialProfileQkThreadKDecode = 13;
constexpr int kPartialProfileSlots = 14;

enum class AttentionVariant : int {
  kReference = 0,
  kOptimizedV1 = 1,
  kOptimizedV2 = 2,
  kOptimizedV3 = 3,
  kOptimizedV4 = 4,
  kOptimizedV5 = 5,
  kOptimizedV7 = 7,
  kOptimizedV8 = 8,
  kOptimizedV9 = 9,
  kOptimizedV10 = 10,
  kOptimizedV11 = 11,
  kOptimizedV12 = 12,
  kOptimizedV13 = 13,
  kOptimizedV14 = 14,
  kOptimizedV15 = 15,
  kOptimizedV16 = 16,
  kOptimizedV17 = 17,
  kOptimizedV18 = 18,
  kOptimizedV19 = 19,
  kOptimizedV20 = 20,
};

const char* attention_variant_name(AttentionVariant variant) {
  switch (variant) {
    case AttentionVariant::kReference:
      return "reference";
    case AttentionVariant::kOptimizedV1:
      return "v1";
    case AttentionVariant::kOptimizedV2:
      return "v2";
    case AttentionVariant::kOptimizedV3:
      return "v3";
    case AttentionVariant::kOptimizedV4:
      return "v4";
    case AttentionVariant::kOptimizedV5:
      return "v5";
    case AttentionVariant::kOptimizedV7:
      return "v7";
    case AttentionVariant::kOptimizedV8:
      return "v8";
    case AttentionVariant::kOptimizedV9:
      return "v9";
    case AttentionVariant::kOptimizedV10:
      return "v10";
    case AttentionVariant::kOptimizedV11:
      return "v11";
    case AttentionVariant::kOptimizedV12:
      return "v12";
    case AttentionVariant::kOptimizedV13:
      return "v13";
    case AttentionVariant::kOptimizedV14:
      return "v14";
    case AttentionVariant::kOptimizedV15:
      return "v15";
    case AttentionVariant::kOptimizedV16:
      return "v16";
    case AttentionVariant::kOptimizedV17:
      return "v17";
    case AttentionVariant::kOptimizedV18:
      return "v18";
    case AttentionVariant::kOptimizedV19:
      return "v19";
    case AttentionVariant::kOptimizedV20:
      return "v20";
  }
  return "unknown";
}

bool split_profile_enabled() {
  const char* value = std::getenv("DSV4_CUDA_REF_PROFILE_SPLIT");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool partial_stage_profile_enabled() {
  const char* value = std::getenv("DSV4_CUDA_REF_PROFILE_PARTIAL_STAGES");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

__device__ __forceinline__ float fp8_e4m3fn_to_float(uint8_t bits) {
  const int sign = bits & 0x80;
  const int exponent = (bits >> 3) & 0x0f;
  const int mantissa = bits & 0x07;

  if (exponent == 0) {
    const float value = static_cast<float>(mantissa) * 0.001953125f;
    return sign ? -value : value;
  } else {
    const uint32_t fp32_bits =
        (static_cast<uint32_t>(sign) << 24) |
        (static_cast<uint32_t>(exponent + 120) << 23) |
        (static_cast<uint32_t>(mantissa) << 20);
    return __uint_as_float(fp32_bits);
  }
}

__device__ __forceinline__ float bf16_bytes_to_float(const uint8_t* ptr) {
  const uint16_t low = static_cast<uint16_t>(ptr[0]);
  const uint16_t high = static_cast<uint16_t>(ptr[1]) << 8;
  const uint32_t bits = static_cast<uint32_t>(low | high) << 16;
  return __uint_as_float(bits);
}

__device__ __forceinline__ __nv_bfloat16 bf16_bytes_to_bfloat16(const uint8_t* ptr) {
  const uint16_t low = static_cast<uint16_t>(ptr[0]);
  const uint16_t high = static_cast<uint16_t>(ptr[1]) << 8;
  return __ushort_as_bfloat16(static_cast<unsigned short>(low | high));
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

__device__ __forceinline__ void store_partial_acc(
    float* partial_acc,
    int64_t offset,
    float value) {
  partial_acc[offset] = value;
}

__device__ __forceinline__ void store_partial_acc(
    __nv_bfloat16* partial_acc,
    int64_t offset,
    float value) {
  partial_acc[offset] = __float2bfloat16(value);
}

__device__ __forceinline__ float load_partial_acc(
    const float* partial_acc,
    int64_t offset) {
  return partial_acc[offset];
}

__device__ __forceinline__ float load_partial_acc(
    const __nv_bfloat16* partial_acc,
    int64_t offset) {
  return __bfloat162float(partial_acc[offset]);
}

__device__ __forceinline__ void add_partial_profile_cycles(
    unsigned long long* profile_cycles,
    int slot,
    unsigned long long cycles) {
  if (profile_cycles != nullptr && threadIdx.x == 0) {
    atomicAdd(profile_cycles + slot, cycles);
  }
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

__device__ __forceinline__ int clamped_length(
    const int32_t* lengths,
    int batch,
    int width) {
  int len = lengths[batch];
  if (len < 0) {
    len = 0;
  }
  if (len > width) {
    len = width;
  }
  return len;
}

__device__ __forceinline__ const uint8_t* selected_combined_row_ptr(
    const uint8_t* swa_cache,
    const int32_t* swa_indices,
    const int32_t* swa_lengths,
    int swa_width,
    int swa_page_size,
    int swa_row_stride,
    const uint8_t* extra_cache,
    const int32_t* extra_indices,
    const int32_t* extra_lengths,
    int extra_width,
    int extra_page_size,
    int extra_row_stride,
    bool has_extra,
    int batch,
    int row,
    bool& valid) {
  if (row < swa_width) {
    valid = row < clamped_length(swa_lengths, batch, swa_width);
    return valid ? selected_row_ptr(
                       swa_cache,
                       swa_indices,
                       swa_width,
                       swa_page_size,
                       swa_row_stride,
                       batch,
                       row)
                 : nullptr;
  }

  const int extra_row = row - swa_width;
  if (!has_extra || extra_row >= extra_width) {
    valid = false;
    return nullptr;
  }
  valid = extra_row < clamped_length(extra_lengths, batch, extra_width);
  return valid ? selected_row_ptr(
                     extra_cache,
                     extra_indices,
                     extra_width,
                     extra_page_size,
                     extra_row_stride,
                     batch,
                     extra_row)
               : nullptr;
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

__global__ void ds4_cuda_scores_reference_kernel(
    const __nv_bfloat16* __restrict__ q,
    const uint8_t* __restrict__ swa_cache,
    const int32_t* __restrict__ swa_indices,
    const int32_t* __restrict__ swa_lengths,
    int swa_width,
    int swa_page_size,
    int swa_row_stride,
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
    int total_width,
    float* __restrict__ out) {
  const int batch = blockIdx.x;
  const int head = blockIdx.y;
  const int row = blockIdx.z;
  if (batch >= batch_size || head >= num_heads || row >= total_width) {
    return;
  }

  bool valid = false;
  const uint8_t* packed = selected_combined_row_ptr(
      swa_cache,
      swa_indices,
      swa_lengths,
      swa_width,
      swa_page_size,
      swa_row_stride,
      extra_cache,
      extra_indices,
      extra_lengths,
      extra_width,
      extra_page_size,
      extra_row_stride,
      has_extra,
      batch,
      row,
      valid);

  if (!valid) {
    if (threadIdx.x == 0) {
      out[(static_cast<int64_t>(batch) * num_heads + head) * total_width + row] = 0.0f;
    }
    return;
  }

  __shared__ float shared[kOptimizedV5SharedFloats];

  const int tid = threadIdx.x;
  const int dim0 = tid;
  const int dim1 = tid + kThreads;
  const int64_t q_base = (static_cast<int64_t>(batch) * num_heads + head) * kHeadDim;

  float dot_part = 0.0f;
  if (dim0 < kHeadDim) {
    dot_part += __bfloat162float(q[q_base + dim0]) * load_dsv4_packed_dim(packed, dim0);
  }
  if (dim1 < kHeadDim) {
    dot_part += __bfloat162float(q[q_base + dim1]) * load_dsv4_packed_dim(packed, dim1);
  }

  reduce_sum_warp_block_total_slot(shared, dot_part);
  if (threadIdx.x == 0) {
    out[(static_cast<int64_t>(batch) * num_heads + head) * total_width + row] =
        dot_part * softmax_scale;
  }
}

__global__ void ds4_cuda_scores_v6_mma_kernel(
    const __nv_bfloat16* __restrict__ q,
    const uint8_t* __restrict__ swa_cache,
    const int32_t* __restrict__ swa_indices,
    const int32_t* __restrict__ swa_lengths,
    int swa_width,
    int swa_page_size,
    int swa_row_stride,
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
    int total_width,
    float* __restrict__ out) {
  namespace wmma = nvcuda::wmma;

  const int batch = blockIdx.x;
  const int head_base = blockIdx.y * kScoreTileM;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row_base = (blockIdx.z * kScoreWarpsPerBlock + warp) * kScoreTileN;
  if (batch >= batch_size) {
    return;
  }

  __shared__ __align__(16) __nv_bfloat16
      q_shared[kScoreWarpsPerBlock][kScoreTileM * kScoreTileK];
  __shared__ __align__(16) __nv_bfloat16
      k_shared[kScoreWarpsPerBlock][kScoreTileK * kScoreTileN];
  __shared__ __align__(16) float score_shared[kScoreWarpsPerBlock][kScoreTileElems];

  wmma::fragment<wmma::matrix_a, kScoreTileM, kScoreTileN, kScoreTileK, __nv_bfloat16, wmma::row_major>
      q_frag;
  wmma::fragment<wmma::matrix_b, kScoreTileM, kScoreTileN, kScoreTileK, __nv_bfloat16, wmma::row_major>
      k_frag;
  wmma::fragment<wmma::accumulator, kScoreTileM, kScoreTileN, kScoreTileK, float>
      acc_frag;
  wmma::fill_fragment(acc_frag, 0.0f);

  for (int dim_base = 0; dim_base < kHeadDim; dim_base += kScoreTileK) {
    for (int idx = lane; idx < kScoreTileElems; idx += 32) {
      const int q_head_slot = idx / kScoreTileK;
      const int q_dim_slot = idx - q_head_slot * kScoreTileK;
      const int head = head_base + q_head_slot;
      const int dim = dim_base + q_dim_slot;
      const int64_t q_offset =
          (static_cast<int64_t>(batch) * num_heads + head) * kHeadDim + dim;
      q_shared[warp][idx] = head < num_heads ? q[q_offset] : __float2bfloat16(0.0f);

      const int k_dim_slot = idx / kScoreTileN;
      const int k_row_slot = idx - k_dim_slot * kScoreTileN;
      const int row = row_base + k_row_slot;
      bool valid = false;
      const uint8_t* packed = selected_combined_row_ptr(
          swa_cache,
          swa_indices,
          swa_lengths,
          swa_width,
          swa_page_size,
          swa_row_stride,
          extra_cache,
          extra_indices,
          extra_lengths,
          extra_width,
          extra_page_size,
          extra_row_stride,
          has_extra,
          batch,
          row,
          valid);
      const float value =
          valid && row < total_width ? load_dsv4_packed_dim(packed, dim_base + k_dim_slot) : 0.0f;
      k_shared[warp][idx] = __float2bfloat16(value);
    }
    __syncwarp();

    wmma::load_matrix_sync(q_frag, q_shared[warp], kScoreTileK);
    wmma::load_matrix_sync(k_frag, k_shared[warp], kScoreTileN);
    wmma::mma_sync(acc_frag, q_frag, k_frag, acc_frag);
    __syncwarp();
  }

  wmma::store_matrix_sync(score_shared[warp], acc_frag, kScoreTileN, wmma::mem_row_major);
  __syncwarp();

  for (int idx = lane; idx < kScoreTileElems; idx += 32) {
    const int head_slot = idx / kScoreTileN;
    const int row_slot = idx - head_slot * kScoreTileN;
    const int head = head_base + head_slot;
    const int row = row_base + row_slot;
    if (head < num_heads && row < total_width) {
      bool valid = false;
      selected_combined_row_ptr(
          swa_cache,
          swa_indices,
          swa_lengths,
          swa_width,
          swa_page_size,
          swa_row_stride,
          extra_cache,
          extra_indices,
          extra_lengths,
          extra_width,
          extra_page_size,
          extra_row_stride,
          has_extra,
          batch,
          row,
          valid);
      out[(static_cast<int64_t>(batch) * num_heads + head) * total_width + row] =
          valid ? score_shared[warp][idx] * softmax_scale : 0.0f;
    }
  }
}

__global__ void ds4_cuda_fused_v7_mma_attention_kernel(
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
    int total_width,
    __nv_bfloat16* __restrict__ out) {
  namespace wmma = nvcuda::wmma;

  const int batch = blockIdx.x;
  const int head_base = blockIdx.y * kScoreTileM;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  if (batch >= batch_size) {
    return;
  }

  __shared__ __align__(16) __nv_bfloat16
      q_shared[kScoreWarpsPerBlock][kScoreTileM * kScoreTileK];
  __shared__ __align__(16) __nv_bfloat16
      k_shared[kScoreWarpsPerBlock][kScoreTileK * kScoreTileN];
  __shared__ __align__(16) float score_shared[kScoreWarpsPerBlock][kScoreTileElems];
  __shared__ __align__(16) float weight_shared[kScoreWarpsPerBlock][kScoreTileElems];
  __shared__ __align__(16) float acc_shared[kV7AccElems];
  __shared__ const uint8_t* row_ptrs[kScoreRowsPerBlock];
  __shared__ uint8_t row_valid[kScoreRowsPerBlock];
  __shared__ float running_max[kScoreTileM];
  __shared__ float running_sum[kScoreTileM];
  __shared__ float old_scale_shared[kScoreTileM];
  __shared__ float inv_sum_shared[kScoreTileM];

  for (int idx = threadIdx.x; idx < kV7AccElems; idx += blockDim.x) {
    acc_shared[idx] = 0.0f;
  }
  if (threadIdx.x < kScoreTileM) {
    running_max[threadIdx.x] = -INFINITY;
    running_sum[threadIdx.x] = 0.0f;
    old_scale_shared[threadIdx.x] = 1.0f;
    inv_sum_shared[threadIdx.x] = 0.0f;
  }
  __syncthreads();

  for (int tile_row_base = 0; tile_row_base < total_width; tile_row_base += kScoreRowsPerBlock) {
    if (threadIdx.x < kScoreRowsPerBlock) {
      const int row = tile_row_base + threadIdx.x;
      bool valid = false;
      const uint8_t* packed = nullptr;
      if (row < total_width) {
        packed = selected_combined_row_ptr(
            swa_cache,
            swa_indices,
            swa_lengths,
            swa_width,
            swa_page_size,
            swa_row_stride,
            extra_cache,
            extra_indices,
            extra_lengths,
            extra_width,
            extra_page_size,
            extra_row_stride,
            has_extra,
            batch,
            row,
            valid);
      }
      row_ptrs[threadIdx.x] = packed;
      row_valid[threadIdx.x] = valid ? 1 : 0;
    }
    __syncthreads();

    if (warp < kScoreWarpsPerBlock) {
      wmma::fragment<wmma::matrix_a, kScoreTileM, kScoreTileN, kScoreTileK, __nv_bfloat16, wmma::row_major>
          q_frag;
      wmma::fragment<wmma::matrix_b, kScoreTileM, kScoreTileN, kScoreTileK, __nv_bfloat16, wmma::row_major>
          k_frag;
      wmma::fragment<wmma::accumulator, kScoreTileM, kScoreTileN, kScoreTileK, float>
          acc_frag;
      wmma::fill_fragment(acc_frag, 0.0f);

      const int warp_row_base = warp * kScoreTileN;
      for (int dim_base = 0; dim_base < kHeadDim; dim_base += kScoreTileK) {
        for (int idx = lane; idx < kScoreTileElems; idx += 32) {
          const int q_head_slot = idx / kScoreTileK;
          const int q_dim_slot = idx - q_head_slot * kScoreTileK;
          const int head = head_base + q_head_slot;
          const int dim = dim_base + q_dim_slot;
          const int64_t q_offset =
              (static_cast<int64_t>(batch) * num_heads + head) * kHeadDim + dim;
          q_shared[warp][idx] = head < num_heads ? q[q_offset] : __float2bfloat16(0.0f);

          const int k_dim_slot = idx / kScoreTileN;
          const int k_row_slot = idx - k_dim_slot * kScoreTileN;
          const int local_row = warp_row_base + k_row_slot;
          const bool valid = row_valid[local_row] != 0;
          const uint8_t* packed = row_ptrs[local_row];
          const float value =
              valid ? load_dsv4_packed_dim(packed, dim_base + k_dim_slot) : 0.0f;
          k_shared[warp][idx] = __float2bfloat16(value);
        }
        __syncwarp();

        wmma::load_matrix_sync(q_frag, q_shared[warp], kScoreTileK);
        wmma::load_matrix_sync(k_frag, k_shared[warp], kScoreTileN);
        wmma::mma_sync(acc_frag, q_frag, k_frag, acc_frag);
        __syncwarp();
      }

      wmma::store_matrix_sync(score_shared[warp], acc_frag, kScoreTileN, wmma::mem_row_major);
    }
    __syncthreads();

    if (threadIdx.x < kScoreTileM) {
      const int head_slot = threadIdx.x;
      const int head = head_base + head_slot;
      float tile_max = -INFINITY;
#pragma unroll
      for (int row_slot = 0; row_slot < kScoreRowsPerBlock; ++row_slot) {
        if (row_valid[row_slot]) {
          const int row_warp = row_slot / kScoreTileN;
          const int row_lane = row_slot - row_warp * kScoreTileN;
          const float score =
              score_shared[row_warp][head_slot * kScoreTileN + row_lane] * softmax_scale;
          tile_max = fmaxf(tile_max, score);
        }
      }

      const float prev_max = running_max[head_slot];
      const float prev_sum = running_sum[head_slot];
      const float new_max = fmaxf(prev_max, tile_max);
      const float old_scale = prev_sum == 0.0f ? 0.0f : expf(prev_max - new_max);
      float tile_sum = 0.0f;
#pragma unroll
      for (int row_slot = 0; row_slot < kScoreRowsPerBlock; ++row_slot) {
        float weight = 0.0f;
        if (head < num_heads && row_valid[row_slot]) {
          const int row_warp = row_slot / kScoreTileN;
          const int row_lane = row_slot - row_warp * kScoreTileN;
          const float score =
              score_shared[row_warp][head_slot * kScoreTileN + row_lane] * softmax_scale;
          weight = expf(score - new_max);
          tile_sum += weight;
        }
        const int row_warp = row_slot / kScoreTileN;
        const int row_lane = row_slot - row_warp * kScoreTileN;
        weight_shared[row_warp][head_slot * kScoreTileN + row_lane] = weight;
      }
      running_max[head_slot] = new_max;
      running_sum[head_slot] = prev_sum * old_scale + tile_sum;
      old_scale_shared[head_slot] = old_scale;
    }
    __syncthreads();

    for (int dim = threadIdx.x; dim < kHeadDim; dim += blockDim.x) {
      float values[kScoreTileM];
#pragma unroll
      for (int head_slot = 0; head_slot < kScoreTileM; ++head_slot) {
        const int head = head_base + head_slot;
        const int acc_idx = head_slot * kHeadDim + dim;
        values[head_slot] =
            head < num_heads ? acc_shared[acc_idx] * old_scale_shared[head_slot] : 0.0f;
      }

#pragma unroll
      for (int row_slot = 0; row_slot < kScoreRowsPerBlock; ++row_slot) {
        if (row_valid[row_slot]) {
          const int row_warp = row_slot / kScoreTileN;
          const int row_lane = row_slot - row_warp * kScoreTileN;
          const float v = load_dsv4_packed_dim(row_ptrs[row_slot], dim);
#pragma unroll
          for (int head_slot = 0; head_slot < kScoreTileM; ++head_slot) {
            const int head = head_base + head_slot;
            if (head < num_heads) {
              const float weight = weight_shared[row_warp][head_slot * kScoreTileN + row_lane];
              values[head_slot] += weight * v;
            }
          }
        }
      }

#pragma unroll
      for (int head_slot = 0; head_slot < kScoreTileM; ++head_slot) {
        const int head = head_base + head_slot;
        const int acc_idx = head_slot * kHeadDim + dim;
        acc_shared[acc_idx] = head < num_heads ? values[head_slot] : 0.0f;
      }
    }
    __syncthreads();
  }

  if (has_sink && threadIdx.x < kScoreTileM) {
    const int head_slot = threadIdx.x;
    const int head = head_base + head_slot;
    float old_scale = 1.0f;
    if (head < num_heads) {
      const float sink = attn_sink[head];
      const float new_max = fmaxf(running_max[head_slot], sink);
      old_scale = running_sum[head_slot] == 0.0f
          ? 0.0f
          : expf(running_max[head_slot] - new_max);
      const float sink_scale = expf(sink - new_max);
      running_sum[head_slot] = running_sum[head_slot] * old_scale + sink_scale;
      running_max[head_slot] = new_max;
    }
    old_scale_shared[head_slot] = old_scale;
  }
  __syncthreads();

  if (has_sink) {
    for (int idx = threadIdx.x; idx < kV7AccElems; idx += blockDim.x) {
      const int head_slot = idx / kHeadDim;
      acc_shared[idx] *= old_scale_shared[head_slot];
    }
    __syncthreads();
  }

  if (threadIdx.x < kScoreTileM) {
    inv_sum_shared[threadIdx.x] =
        running_sum[threadIdx.x] > 0.0f ? 1.0f / running_sum[threadIdx.x] : 0.0f;
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < kV7AccElems; idx += blockDim.x) {
    const int head_slot = idx / kHeadDim;
    const int dim = idx - head_slot * kHeadDim;
    const int head = head_base + head_slot;
    if (head < num_heads) {
      const int64_t out_offset =
          (static_cast<int64_t>(batch) * num_heads + head) * kHeadDim + dim;
      out[out_offset] = __float2bfloat16(acc_shared[idx] * inv_sum_shared[head_slot]);
    }
  }
}

template <
    bool CacheScales,
    int PVMmaMode,
    typename PartialAccT = float,
    bool ProfileStages = false>
__global__ void ds4_cuda_fused_v8_mma_partial_kernel(
    const __nv_bfloat16* __restrict__ q,
    const uint8_t* __restrict__ swa_cache,
    const int32_t* __restrict__ swa_indices,
    const int32_t* __restrict__ swa_lengths,
    int swa_width,
    int swa_page_size,
    int swa_row_stride,
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
    int total_width,
    int head_tiles,
    int row_tiles,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    PartialAccT* __restrict__ partial_acc,
    unsigned long long* __restrict__ profile_cycles) {
  namespace wmma = nvcuda::wmma;

  const int batch = blockIdx.x;
  const int head_tile = blockIdx.y;
  const int row_tile = blockIdx.z;
  const int head_base = head_tile * kScoreTileM;
  const int tile_row_base = row_tile * kScoreRowsPerBlock;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  if (batch >= batch_size) {
    return;
  }
  constexpr bool profile_this_block =
      ProfileStages && (PVMmaMode == 5 || PVMmaMode == 6 || PVMmaMode == 7);
  unsigned long long profile_t0 = 0;
  if (profile_this_block && threadIdx.x == 0) {
    profile_t0 = clock64();
    atomicAdd(profile_cycles + kPartialProfileBlocks, 1ULL);
  }

  constexpr int kPartialSharedWarps =
      PVMmaMode >= 2 ? kV11WarpsPerBlock : kScoreWarpsPerBlock;
  __shared__ __align__(16) __nv_bfloat16
      q_shared[kPartialSharedWarps][kScoreTileM * kScoreTileK];
  __shared__ __align__(16) __nv_bfloat16
      q_reuse_shared[PVMmaMode == 7 ? kScoreKTileCount : 1][kScoreTileElems];
  __shared__ __align__(16) __nv_bfloat16
      k_shared[kPartialSharedWarps][kScoreTileK * kScoreTileN];
  __shared__ __align__(16) float score_shared[kPartialSharedWarps][kScoreTileElems];
  __shared__ __align__(16) float weight_shared[kScoreWarpsPerBlock][kScoreTileElems];
  __shared__ float row_scales[CacheScales ? kScoreRowsPerBlock : 1][kScaleCount];
  __shared__ const uint8_t* row_ptrs[kScoreRowsPerBlock];
  __shared__ uint8_t row_valid[kScoreRowsPerBlock];
  __shared__ float tile_max_shared[kScoreTileM];
  __shared__ float tile_sum_shared[kScoreTileM];

  if (threadIdx.x < kScoreTileM) {
    tile_max_shared[threadIdx.x] = -INFINITY;
    tile_sum_shared[threadIdx.x] = 0.0f;
  }
  if (threadIdx.x < kScoreRowsPerBlock) {
    const int row = tile_row_base + threadIdx.x;
    bool valid = false;
    const uint8_t* packed = nullptr;
    if (row < total_width) {
      packed = selected_combined_row_ptr(
          swa_cache,
          swa_indices,
          swa_lengths,
          swa_width,
          swa_page_size,
          swa_row_stride,
          extra_cache,
          extra_indices,
          extra_lengths,
          extra_width,
          extra_page_size,
          extra_row_stride,
          has_extra,
          batch,
          row,
          valid);
    }
    row_ptrs[threadIdx.x] = packed;
    row_valid[threadIdx.x] = valid ? 1 : 0;
  }
  __syncthreads();
  if (profile_this_block && threadIdx.x == 0) {
    const unsigned long long now = clock64();
    add_partial_profile_cycles(
        profile_cycles, kPartialProfileRowSetup, now - profile_t0);
    profile_t0 = now;
  }
  if constexpr (CacheScales) {
    if (threadIdx.x < kScoreRowsPerBlock * kScaleCount) {
      const int row_slot = threadIdx.x / kScaleCount;
      const int scale_slot = threadIdx.x - row_slot * kScaleCount;
      const uint8_t* packed = row_ptrs[row_slot];
      row_scales[row_slot][scale_slot] =
          row_valid[row_slot]
              ? exp2f(static_cast<float>(packed[kScaleOffset + scale_slot]) - 127.0f)
              : 0.0f;
    }
  }
  __syncthreads();
  if (profile_this_block && threadIdx.x == 0) {
    const unsigned long long now = clock64();
    add_partial_profile_cycles(
        profile_cycles, kPartialProfileScaleCache, now - profile_t0);
    profile_t0 = now;
  }

  if constexpr (PVMmaMode == 7) {
    unsigned long long q_reuse_t0 = 0;
    if (profile_this_block && threadIdx.x == 0) {
      q_reuse_t0 = clock64();
    }
    for (int idx = threadIdx.x; idx < kScoreKTileCount * kScoreTileElems;
         idx += blockDim.x) {
      const int dim_tile = idx / kScoreTileElems;
      const int tile_idx = idx - dim_tile * kScoreTileElems;
      const int q_head_slot = tile_idx / kScoreTileK;
      const int q_dim_slot = tile_idx - q_head_slot * kScoreTileK;
      const int head = head_base + q_head_slot;
      const int dim = dim_tile * kScoreTileK + q_dim_slot;
      const int64_t q_offset =
          (static_cast<int64_t>(batch) * num_heads + head) * kHeadDim + dim;
      q_reuse_shared[dim_tile][tile_idx] =
          head < num_heads ? q[q_offset] : __float2bfloat16(0.0f);
    }
    __syncthreads();
    if (profile_this_block && threadIdx.x == 0) {
      const unsigned long long now = clock64();
      add_partial_profile_cycles(
          profile_cycles, kPartialProfileQkStagingWall, now - q_reuse_t0);
      add_partial_profile_cycles(
          profile_cycles, kPartialProfileQkThreadQLoad, now - q_reuse_t0);
    }
  }

  unsigned long long qk_barrier_t0 = profile_t0;
  if (warp < kScoreWarpsPerBlock) {
    wmma::fragment<wmma::matrix_a, kScoreTileM, kScoreTileN, kScoreTileK, __nv_bfloat16, wmma::row_major>
        q_frag;
    wmma::fragment<wmma::matrix_b, kScoreTileM, kScoreTileN, kScoreTileK, __nv_bfloat16, wmma::row_major>
        k_frag;
    wmma::fragment<wmma::accumulator, kScoreTileM, kScoreTileN, kScoreTileK, float>
        acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    const int warp_row_base = warp * kScoreTileN;
    unsigned long long qk_thread_q_load_cycles = 0;
    unsigned long long qk_thread_k_decode_cycles = 0;
    for (int dim_base = 0; dim_base < kHeadDim; dim_base += kScoreTileK) {
      unsigned long long qk_detail_t0 = 0;
      if (profile_this_block && threadIdx.x == 0) {
        qk_detail_t0 = clock64();
      }
      for (int idx = lane; idx < kScoreTileElems; idx += 32) {
        if constexpr (PVMmaMode != 7) {
          unsigned long long q_thread_t0 = 0;
          if (profile_this_block && threadIdx.x == 0) {
            q_thread_t0 = clock64();
          }
          const int q_head_slot = idx / kScoreTileK;
          const int q_dim_slot = idx - q_head_slot * kScoreTileK;
          const int head = head_base + q_head_slot;
          const int dim = dim_base + q_dim_slot;
          const int64_t q_offset =
              (static_cast<int64_t>(batch) * num_heads + head) * kHeadDim + dim;
          q_shared[warp][idx] = head < num_heads ? q[q_offset] : __float2bfloat16(0.0f);
          if (profile_this_block && threadIdx.x == 0) {
            const unsigned long long now = clock64();
            qk_thread_q_load_cycles += now - q_thread_t0;
          }
        }

        unsigned long long k_thread_t0 = 0;
        if (profile_this_block && threadIdx.x == 0) {
          k_thread_t0 = clock64();
        }
        if constexpr (PVMmaMode == 6) {
          const int load_row_lane = idx / kScoreTileK;
          const int load_dim_lane = idx - load_row_lane * kScoreTileK;
          const int local_row = warp_row_base + load_row_lane;
          const int dim = dim_base + load_dim_lane;
          const int shared_idx = load_dim_lane * kScoreTileN + load_row_lane;
          __nv_bfloat16 value = __float2bfloat16(0.0f);
          if (row_valid[local_row]) {
            const uint8_t* packed = row_ptrs[local_row];
            if constexpr (CacheScales) {
              if (dim_base < kNopeDim) {
                const int scale_slot = dim_base / kScaleGroup;
                const float decoded =
                    fp8_e4m3fn_to_float(packed[dim]) * row_scales[local_row][scale_slot];
                value = __float2bfloat16(decoded);
              } else {
                value = bf16_bytes_to_bfloat16(
                    packed + kNopeDim + (dim - kNopeDim) * 2);
              }
            } else {
              value = __float2bfloat16(load_dsv4_packed_dim(packed, dim));
            }
          }
          k_shared[warp][shared_idx] = value;
        } else {
          const int k_dim_slot = idx / kScoreTileN;
          const int k_row_slot = idx - k_dim_slot * kScoreTileN;
          const int local_row = warp_row_base + k_row_slot;
          const bool valid = row_valid[local_row] != 0;
          const uint8_t* packed = row_ptrs[local_row];
          float value = 0.0f;
          if (valid) {
            const int dim = dim_base + k_dim_slot;
            if constexpr (CacheScales) {
              value = load_dsv4_packed_dim_with_scales(packed, dim, row_scales[local_row]);
            } else {
              value = load_dsv4_packed_dim(packed, dim);
            }
          }
          k_shared[warp][idx] = __float2bfloat16(value);
        }
        if (profile_this_block && threadIdx.x == 0) {
          const unsigned long long now = clock64();
          qk_thread_k_decode_cycles += now - k_thread_t0;
        }
      }
      __syncwarp();
      if (profile_this_block && threadIdx.x == 0) {
        const unsigned long long now = clock64();
        add_partial_profile_cycles(
            profile_cycles, kPartialProfileQkStagingWall, now - qk_detail_t0);
        qk_detail_t0 = now;
      }

      if constexpr (PVMmaMode == 7) {
        wmma::load_matrix_sync(
            q_frag, q_reuse_shared[dim_base / kScoreTileK], kScoreTileK);
      } else {
        wmma::load_matrix_sync(q_frag, q_shared[warp], kScoreTileK);
      }
      wmma::load_matrix_sync(k_frag, k_shared[warp], kScoreTileN);
      wmma::mma_sync(acc_frag, q_frag, k_frag, acc_frag);
      __syncwarp();
      if (profile_this_block && threadIdx.x == 0) {
        const unsigned long long now = clock64();
        add_partial_profile_cycles(
            profile_cycles, kPartialProfileQkMmaWall, now - qk_detail_t0);
      }
    }

    unsigned long long qk_store_t0 = 0;
    if (profile_this_block && threadIdx.x == 0) {
      qk_store_t0 = clock64();
    }
    wmma::store_matrix_sync(score_shared[warp], acc_frag, kScoreTileN, wmma::mem_row_major);
    if (profile_this_block && threadIdx.x == 0) {
      const unsigned long long now = clock64();
      add_partial_profile_cycles(
          profile_cycles, kPartialProfileQkScoreStoreWall, now - qk_store_t0);
      add_partial_profile_cycles(
          profile_cycles, kPartialProfileQkThreadQLoad, qk_thread_q_load_cycles);
      add_partial_profile_cycles(
          profile_cycles, kPartialProfileQkThreadKDecode, qk_thread_k_decode_cycles);
      qk_barrier_t0 = now;
    }
  }
  __syncthreads();
  if (profile_this_block && threadIdx.x == 0) {
    const unsigned long long now = clock64();
    add_partial_profile_cycles(
        profile_cycles, kPartialProfileQkBlockBarrier, now - qk_barrier_t0);
    add_partial_profile_cycles(profile_cycles, kPartialProfileQk, now - profile_t0);
    profile_t0 = now;
  }

  if (threadIdx.x < kScoreTileM) {
    const int head_slot = threadIdx.x;
    const int head = head_base + head_slot;
    float tile_max = -INFINITY;
#pragma unroll
    for (int row_slot = 0; row_slot < kScoreRowsPerBlock; ++row_slot) {
      if (head < num_heads && row_valid[row_slot]) {
        const int row_warp = row_slot / kScoreTileN;
        const int row_lane = row_slot - row_warp * kScoreTileN;
        const float score =
            score_shared[row_warp][head_slot * kScoreTileN + row_lane] * softmax_scale;
        tile_max = fmaxf(tile_max, score);
      }
    }

    float tile_sum = 0.0f;
#pragma unroll
    for (int row_slot = 0; row_slot < kScoreRowsPerBlock; ++row_slot) {
      float weight = 0.0f;
      if (head < num_heads && row_valid[row_slot]) {
        const int row_warp = row_slot / kScoreTileN;
        const int row_lane = row_slot - row_warp * kScoreTileN;
        const float score =
            score_shared[row_warp][head_slot * kScoreTileN + row_lane] * softmax_scale;
        weight = expf(score - tile_max);
        tile_sum += weight;
      }
      const int row_warp = row_slot / kScoreTileN;
      const int row_lane = row_slot - row_warp * kScoreTileN;
      weight_shared[row_warp][head_slot * kScoreTileN + row_lane] = weight;
    }
    tile_max_shared[head_slot] = tile_max;
    tile_sum_shared[head_slot] = tile_sum;

    const int64_t state_offset =
        (((static_cast<int64_t>(batch) * head_tiles + head_tile) * row_tiles + row_tile) *
             kScoreTileM +
         head_slot);
    partial_max[state_offset] = tile_max;
    partial_sum[state_offset] = tile_sum;
  }
  __syncthreads();
  if (profile_this_block && threadIdx.x == 0) {
    const unsigned long long now = clock64();
    add_partial_profile_cycles(
        profile_cycles, kPartialProfileSoftmax, now - profile_t0);
    profile_t0 = now;
  }

  if constexpr (PVMmaMode == 1) {
    // Experimental v10 path: treat the softmax weights as P and the decoded
    // value tile as V, then use BF16 MMA to form a partial P @ V tile.
    wmma::fragment<wmma::matrix_a, kScoreTileM, kScoreTileN, kScoreTileK, __nv_bfloat16, wmma::row_major>
        p_frag;
    wmma::fragment<wmma::matrix_b, kScoreTileM, kScoreTileN, kScoreTileK, __nv_bfloat16, wmma::row_major>
        v_frag;
    wmma::fragment<wmma::accumulator, kScoreTileM, kScoreTileN, kScoreTileK, float>
        pv_acc_frag;

    for (int dim_base = 0; dim_base < kHeadDim; dim_base += kScoreTileN) {
      if (warp < kScoreWarpsPerBlock) {
        for (int idx = lane; idx < kScoreTileElems; idx += 32) {
          q_shared[warp][idx] = __float2bfloat16(weight_shared[warp][idx]);

          const int k_slot = idx / kScoreTileN;
          const int dim_slot = idx - k_slot * kScoreTileN;
          const int row_slot = warp * kScoreTileN + k_slot;
          const int dim = dim_base + dim_slot;
          float v = 0.0f;
          if (row_valid[row_slot]) {
            if constexpr (CacheScales) {
              v = load_dsv4_packed_dim_with_scales(
                  row_ptrs[row_slot], dim, row_scales[row_slot]);
            } else {
              v = load_dsv4_packed_dim(row_ptrs[row_slot], dim);
            }
          }
          k_shared[warp][idx] = __float2bfloat16(v);
        }
        __syncwarp();

        wmma::load_matrix_sync(p_frag, q_shared[warp], kScoreTileN);
        wmma::load_matrix_sync(v_frag, k_shared[warp], kScoreTileN);
        wmma::fill_fragment(pv_acc_frag, 0.0f);
        wmma::mma_sync(pv_acc_frag, p_frag, v_frag, pv_acc_frag);
        wmma::store_matrix_sync(score_shared[warp], pv_acc_frag, kScoreTileN, wmma::mem_row_major);
      }
      __syncthreads();

      for (int idx = threadIdx.x; idx < kScoreTileElems; idx += blockDim.x) {
        const int head_slot = idx / kScoreTileN;
        const int dim_slot = idx - head_slot * kScoreTileN;
        const int head = head_base + head_slot;
        const int dim = dim_base + dim_slot;
        float value = 0.0f;
#pragma unroll
        for (int row_warp = 0; row_warp < kScoreWarpsPerBlock; ++row_warp) {
          value += score_shared[row_warp][idx];
        }
        const int64_t acc_offset =
            ((((static_cast<int64_t>(batch) * head_tiles + head_tile) * row_tiles + row_tile) *
                  kScoreTileM +
              head_slot) *
                 kHeadDim +
             dim);
        store_partial_acc(partial_acc, acc_offset, head < num_heads ? value : 0.0f);
      }
      __syncthreads();
    }
    return;
  }

  if constexpr (PVMmaMode == 2) {
    // Experimental v11 path: map 16 warps as 4 row groups x 4 dim groups so
    // four V dimension tiles are processed per round.
    wmma::fragment<wmma::matrix_a, kScoreTileM, kScoreTileN, kScoreTileK, __nv_bfloat16, wmma::row_major>
        p_frag;
    wmma::fragment<wmma::matrix_b, kScoreTileM, kScoreTileN, kScoreTileK, __nv_bfloat16, wmma::row_major>
        v_frag;
    wmma::fragment<wmma::accumulator, kScoreTileM, kScoreTileN, kScoreTileK, float>
        pv_acc_frag;

    for (int dim_round = 0; dim_round < kV11DimRounds; ++dim_round) {
      if (warp < kV11WarpsPerBlock) {
        const int row_group = warp / kV11DimGroups;
        const int dim_group = warp - row_group * kV11DimGroups;
        const int dim_base = (dim_round * kV11DimGroups + dim_group) * kScoreTileN;

        for (int idx = lane; idx < kScoreTileElems; idx += 32) {
          q_shared[warp][idx] = __float2bfloat16(weight_shared[row_group][idx]);

          const int k_slot = idx / kScoreTileN;
          const int dim_slot = idx - k_slot * kScoreTileN;
          const int row_slot = row_group * kScoreTileN + k_slot;
          const int dim = dim_base + dim_slot;
          float v = 0.0f;
          if (row_valid[row_slot]) {
            if constexpr (CacheScales) {
              v = load_dsv4_packed_dim_with_scales(
                  row_ptrs[row_slot], dim, row_scales[row_slot]);
            } else {
              v = load_dsv4_packed_dim(row_ptrs[row_slot], dim);
            }
          }
          k_shared[warp][idx] = __float2bfloat16(v);
        }
        __syncwarp();

        wmma::load_matrix_sync(p_frag, q_shared[warp], kScoreTileN);
        wmma::load_matrix_sync(v_frag, k_shared[warp], kScoreTileN);
        wmma::fill_fragment(pv_acc_frag, 0.0f);
        wmma::mma_sync(pv_acc_frag, p_frag, v_frag, pv_acc_frag);
        wmma::store_matrix_sync(score_shared[warp], pv_acc_frag, kScoreTileN, wmma::mem_row_major);
      }
      __syncthreads();

      for (int idx = threadIdx.x; idx < kV11DimGroups * kScoreTileElems; idx += blockDim.x) {
        const int dim_group = idx / kScoreTileElems;
        const int tile_idx = idx - dim_group * kScoreTileElems;
        const int head_slot = tile_idx / kScoreTileN;
        const int dim_slot = tile_idx - head_slot * kScoreTileN;
        const int head = head_base + head_slot;
        const int dim = (dim_round * kV11DimGroups + dim_group) * kScoreTileN + dim_slot;
        float value = 0.0f;
#pragma unroll
        for (int row_group = 0; row_group < kScoreWarpsPerBlock; ++row_group) {
          const int result_warp = row_group * kV11DimGroups + dim_group;
          value += score_shared[result_warp][tile_idx];
        }
        const int64_t acc_offset =
            ((((static_cast<int64_t>(batch) * head_tiles + head_tile) * row_tiles + row_tile) *
                  kScoreTileM +
              head_slot) *
                 kHeadDim +
             dim);
        store_partial_acc(partial_acc, acc_offset, head < num_heads ? value : 0.0f);
      }
      __syncthreads();
    }
    return;
  }

  if constexpr (PVMmaMode == 3 || PVMmaMode == 4) {
    // Experimental v12/v14 path: keep the v11 2D warp schedule, but cache each
    // row group's BF16 P tile once and reuse it across all dim-group warps and
    // dimension rounds. v14 additionally specializes V decode by 16-dim tile.
    wmma::fragment<wmma::matrix_a, kScoreTileM, kScoreTileN, kScoreTileK, __nv_bfloat16, wmma::row_major>
        p_frag;
    wmma::fragment<wmma::matrix_b, kScoreTileM, kScoreTileN, kScoreTileK, __nv_bfloat16, wmma::row_major>
        v_frag;
    wmma::fragment<wmma::accumulator, kScoreTileM, kScoreTileN, kScoreTileK, float>
        pv_acc_frag;

    if (warp < kScoreWarpsPerBlock) {
      for (int idx = lane; idx < kScoreTileElems; idx += 32) {
        q_shared[warp][idx] = __float2bfloat16(weight_shared[warp][idx]);
      }
    }
    __syncthreads();

    for (int dim_round = 0; dim_round < kV11DimRounds; ++dim_round) {
      if (warp < kV11WarpsPerBlock) {
        const int row_group = warp / kV11DimGroups;
        const int dim_group = warp - row_group * kV11DimGroups;
        const int dim_base = (dim_round * kV11DimGroups + dim_group) * kScoreTileN;
        const bool nope_tile = dim_base < kNopeDim;
        const int scale_slot = dim_base / kScaleGroup;

        for (int idx = lane; idx < kScoreTileElems; idx += 32) {
          const int k_slot = idx / kScoreTileN;
          const int dim_slot = idx - k_slot * kScoreTileN;
          const int row_slot = row_group * kScoreTileN + k_slot;
          const int dim = dim_base + dim_slot;
          float v = 0.0f;
          if (row_valid[row_slot]) {
            if constexpr (PVMmaMode == 4 && CacheScales) {
              const uint8_t* row = row_ptrs[row_slot];
              if (nope_tile) {
                v = fp8_e4m3fn_to_float(row[dim]) * row_scales[row_slot][scale_slot];
              } else {
                v = bf16_bytes_to_float(row + kNopeDim + (dim - kNopeDim) * 2);
              }
            } else if constexpr (CacheScales) {
              v = load_dsv4_packed_dim_with_scales(
                  row_ptrs[row_slot], dim, row_scales[row_slot]);
            } else {
              v = load_dsv4_packed_dim(row_ptrs[row_slot], dim);
            }
          }
          k_shared[warp][idx] = __float2bfloat16(v);
        }
        __syncwarp();

        wmma::load_matrix_sync(p_frag, q_shared[row_group], kScoreTileN);
        wmma::load_matrix_sync(v_frag, k_shared[warp], kScoreTileN);
        wmma::fill_fragment(pv_acc_frag, 0.0f);
        wmma::mma_sync(pv_acc_frag, p_frag, v_frag, pv_acc_frag);
        wmma::store_matrix_sync(score_shared[warp], pv_acc_frag, kScoreTileN, wmma::mem_row_major);
      }
      __syncthreads();

      for (int idx = threadIdx.x; idx < kV11DimGroups * kScoreTileElems; idx += blockDim.x) {
        const int dim_group = idx / kScoreTileElems;
        const int tile_idx = idx - dim_group * kScoreTileElems;
        const int head_slot = tile_idx / kScoreTileN;
        const int dim_slot = tile_idx - head_slot * kScoreTileN;
        const int head = head_base + head_slot;
        const int dim = (dim_round * kV11DimGroups + dim_group) * kScoreTileN + dim_slot;
        float value = 0.0f;
#pragma unroll
        for (int row_group = 0; row_group < kScoreWarpsPerBlock; ++row_group) {
          const int result_warp = row_group * kV11DimGroups + dim_group;
          value += score_shared[result_warp][tile_idx];
        }
        const int64_t acc_offset =
            ((((static_cast<int64_t>(batch) * head_tiles + head_tile) * row_tiles + row_tile) *
                  kScoreTileM +
              head_slot) *
                 kHeadDim +
             dim);
        store_partial_acc(partial_acc, acc_offset, head < num_heads ? value : 0.0f);
      }
      __syncthreads();
    }
    return;
  }

  if constexpr (PVMmaMode == 5 || PVMmaMode == 6 || PVMmaMode == 7) {
    // Experimental v15/v16/v17 path: keep v14's cached-P, BF16 partial accumulator,
    // and specialized V decode, but remove the explicit shared-memory
    // row-group reduction. Each warp accumulates all four 16-row groups into
    // one WMMA accumulator for one output dim tile, while the 16 warps cover
    // four dimension rounds x four dim groups concurrently. v16 differs in the
    // preceding QK K-staging path; v17 reuses a block-level Q tile for QK.
    wmma::fragment<wmma::matrix_a, kScoreTileM, kScoreTileN, kScoreTileK, __nv_bfloat16, wmma::row_major>
        p_frag;
    wmma::fragment<wmma::matrix_b, kScoreTileM, kScoreTileN, kScoreTileK, __nv_bfloat16, wmma::row_major>
        v_frag;
    wmma::fragment<wmma::accumulator, kScoreTileM, kScoreTileN, kScoreTileK, float>
        pv_acc_frag;

    if (warp < kScoreWarpsPerBlock) {
      for (int idx = lane; idx < kScoreTileElems; idx += 32) {
        q_shared[warp][idx] = __float2bfloat16(weight_shared[warp][idx]);
      }
    }
    __syncthreads();
    if (profile_this_block && threadIdx.x == 0) {
      const unsigned long long now = clock64();
      add_partial_profile_cycles(
          profile_cycles, kPartialProfilePCache, now - profile_t0);
      profile_t0 = now;
    }

    for (int outer_round = 0; outer_round < kV15DimOuterRounds; ++outer_round) {
      if (warp < kV11WarpsPerBlock) {
        const int dim_round_group = warp / kV11DimGroups;
        const int dim_group = warp - dim_round_group * kV11DimGroups;
        const int dim_round = outer_round * kV15DimRoundGroups + dim_round_group;
        const int dim_base = (dim_round * kV11DimGroups + dim_group) * kScoreTileN;
        const bool nope_tile = dim_base < kNopeDim;
        const int scale_slot = dim_base / kScaleGroup;

        wmma::fill_fragment(pv_acc_frag, 0.0f);
#pragma unroll
        for (int row_group = 0; row_group < kScoreWarpsPerBlock; ++row_group) {
          for (int idx = lane; idx < kScoreTileElems; idx += 32) {
            const int k_slot = idx / kScoreTileN;
            const int dim_slot = idx - k_slot * kScoreTileN;
            const int row_slot = row_group * kScoreTileN + k_slot;
            const int dim = dim_base + dim_slot;
            float v = 0.0f;
            if (row_valid[row_slot]) {
              const uint8_t* row = row_ptrs[row_slot];
              if constexpr (CacheScales) {
                if (nope_tile) {
                  v = fp8_e4m3fn_to_float(row[dim]) * row_scales[row_slot][scale_slot];
                } else {
                  v = bf16_bytes_to_float(row + kNopeDim + (dim - kNopeDim) * 2);
                }
              } else {
                v = load_dsv4_packed_dim(row, dim);
              }
            }
            k_shared[warp][idx] = __float2bfloat16(v);
          }
          __syncwarp();

          wmma::load_matrix_sync(p_frag, q_shared[row_group], kScoreTileN);
          wmma::load_matrix_sync(v_frag, k_shared[warp], kScoreTileN);
          wmma::mma_sync(pv_acc_frag, p_frag, v_frag, pv_acc_frag);
          __syncwarp();
        }

        wmma::store_matrix_sync(score_shared[warp], pv_acc_frag, kScoreTileN, wmma::mem_row_major);
      }
      __syncthreads();
      if (profile_this_block && threadIdx.x == 0) {
        const unsigned long long now = clock64();
        add_partial_profile_cycles(
            profile_cycles, kPartialProfilePvMma, now - profile_t0);
        profile_t0 = now;
      }

      for (int idx = threadIdx.x;
           idx < kV15DimRoundGroups * kV11DimGroups * kScoreTileElems;
           idx += blockDim.x) {
        const int dim_round_group = idx / (kV11DimGroups * kScoreTileElems);
        const int rem = idx - dim_round_group * kV11DimGroups * kScoreTileElems;
        const int dim_group = rem / kScoreTileElems;
        const int tile_idx = rem - dim_group * kScoreTileElems;
        const int head_slot = tile_idx / kScoreTileN;
        const int dim_slot = tile_idx - head_slot * kScoreTileN;
        const int head = head_base + head_slot;
        const int dim_round = outer_round * kV15DimRoundGroups + dim_round_group;
        const int dim = (dim_round * kV11DimGroups + dim_group) * kScoreTileN + dim_slot;
        const int result_warp = dim_round_group * kV11DimGroups + dim_group;
        const float value = score_shared[result_warp][tile_idx];
        const int64_t acc_offset =
            ((((static_cast<int64_t>(batch) * head_tiles + head_tile) * row_tiles + row_tile) *
                  kScoreTileM +
              head_slot) *
                 kHeadDim +
             dim);
        store_partial_acc(partial_acc, acc_offset, head < num_heads ? value : 0.0f);
      }
      __syncthreads();
      if (profile_this_block && threadIdx.x == 0) {
        const unsigned long long now = clock64();
        add_partial_profile_cycles(
            profile_cycles, kPartialProfilePartialStore, now - profile_t0);
        profile_t0 = now;
      }
    }
    return;
  }

  for (int dim = threadIdx.x; dim < kHeadDim; dim += blockDim.x) {
    float values[kScoreTileM];
#pragma unroll
    for (int head_slot = 0; head_slot < kScoreTileM; ++head_slot) {
      values[head_slot] = 0.0f;
    }

#pragma unroll
    for (int row_slot = 0; row_slot < kScoreRowsPerBlock; ++row_slot) {
      if (row_valid[row_slot]) {
        const int row_warp = row_slot / kScoreTileN;
        const int row_lane = row_slot - row_warp * kScoreTileN;
        float v = 0.0f;
        if constexpr (CacheScales) {
          v = load_dsv4_packed_dim_with_scales(row_ptrs[row_slot], dim, row_scales[row_slot]);
        } else {
          v = load_dsv4_packed_dim(row_ptrs[row_slot], dim);
        }
#pragma unroll
        for (int head_slot = 0; head_slot < kScoreTileM; ++head_slot) {
          const int head = head_base + head_slot;
          if (head < num_heads) {
            const float weight = weight_shared[row_warp][head_slot * kScoreTileN + row_lane];
            values[head_slot] += weight * v;
          }
        }
      }
    }

#pragma unroll
    for (int head_slot = 0; head_slot < kScoreTileM; ++head_slot) {
      const int head = head_base + head_slot;
      const int64_t acc_offset =
          ((((static_cast<int64_t>(batch) * head_tiles + head_tile) * row_tiles + row_tile) *
                kScoreTileM +
            head_slot) *
               kHeadDim +
           dim);
      store_partial_acc(
          partial_acc, acc_offset, head < num_heads ? values[head_slot] : 0.0f);
    }
  }
}

template <typename PartialAccT = float, bool CacheReduceScales = false>
__global__ void ds4_cuda_fused_v8_reduce_kernel(
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    const PartialAccT* __restrict__ partial_acc,
    const float* __restrict__ attn_sink,
    bool has_sink,
    int batch_size,
    int num_heads,
    int head_tiles,
    int row_tiles,
    __nv_bfloat16* __restrict__ out) {
  const int batch = blockIdx.x;
  const int head_tile = blockIdx.y;
  const int head_base = head_tile * kScoreTileM;
  if (batch >= batch_size) {
    return;
  }

  __shared__ float final_max[kScoreTileM];
  __shared__ float final_sum[kScoreTileM];
  __shared__ float inv_sum[kScoreTileM];
  __shared__ float reduce_scale_shared
      [kScoreTileM][CacheReduceScales ? kReduceScaleCacheMaxRowTiles : 1];
  const bool cache_reduce_scales =
      CacheReduceScales && row_tiles <= kReduceScaleCacheMaxRowTiles;

  if (threadIdx.x < kScoreTileM) {
    const int head_slot = threadIdx.x;
    const int head = head_base + head_slot;
    float m = -INFINITY;
    if (head < num_heads) {
      for (int row_tile = 0; row_tile < row_tiles; ++row_tile) {
        const int64_t state_offset =
            (((static_cast<int64_t>(batch) * head_tiles + head_tile) * row_tiles + row_tile) *
                 kScoreTileM +
             head_slot);
        const float l = partial_sum[state_offset];
        if (l > 0.0f) {
          m = fmaxf(m, partial_max[state_offset]);
        }
      }
      if (has_sink) {
        m = fmaxf(m, attn_sink[head]);
      }
    }

    float l_total = 0.0f;
    if (head < num_heads) {
      for (int row_tile = 0; row_tile < row_tiles; ++row_tile) {
        const int64_t state_offset =
            (((static_cast<int64_t>(batch) * head_tiles + head_tile) * row_tiles + row_tile) *
                 kScoreTileM +
             head_slot);
        const float l = partial_sum[state_offset];
        if (l > 0.0f) {
          const float scale = expf(partial_max[state_offset] - m);
          l_total += l * scale;
          if constexpr (CacheReduceScales) {
            if (cache_reduce_scales) {
              reduce_scale_shared[head_slot][row_tile] = scale;
            }
          }
        } else if constexpr (CacheReduceScales) {
          if (cache_reduce_scales) {
            reduce_scale_shared[head_slot][row_tile] = 0.0f;
          }
        }
      }
      if (has_sink) {
        l_total += expf(attn_sink[head] - m);
      }
    } else if constexpr (CacheReduceScales) {
      if (cache_reduce_scales) {
        for (int row_tile = 0; row_tile < row_tiles; ++row_tile) {
          reduce_scale_shared[head_slot][row_tile] = 0.0f;
        }
      }
    }

    final_max[head_slot] = m;
    final_sum[head_slot] = l_total;
    inv_sum[head_slot] = l_total > 0.0f ? 1.0f / l_total : 0.0f;
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < kV7AccElems; idx += blockDim.x) {
    const int head_slot = idx / kHeadDim;
    const int dim = idx - head_slot * kHeadDim;
    const int head = head_base + head_slot;
    if (head < num_heads) {
      float acc = 0.0f;
      for (int row_tile = 0; row_tile < row_tiles; ++row_tile) {
        const int64_t state_offset =
            (((static_cast<int64_t>(batch) * head_tiles + head_tile) * row_tiles + row_tile) *
                 kScoreTileM +
             head_slot);
        float scale = 0.0f;
        if constexpr (CacheReduceScales) {
          if (cache_reduce_scales) {
            scale = reduce_scale_shared[head_slot][row_tile];
          } else {
            const float l = partial_sum[state_offset];
            scale =
                l > 0.0f ? expf(partial_max[state_offset] - final_max[head_slot]) : 0.0f;
          }
        } else {
          const float l = partial_sum[state_offset];
          scale =
              l > 0.0f ? expf(partial_max[state_offset] - final_max[head_slot]) : 0.0f;
        }
        if (scale != 0.0f) {
          const int64_t acc_offset =
              ((((static_cast<int64_t>(batch) * head_tiles + head_tile) * row_tiles + row_tile) *
                    kScoreTileM +
                head_slot) *
                   kHeadDim +
               dim);
          acc += load_partial_acc(partial_acc, acc_offset) * scale;
        }
      }
      const int64_t out_offset =
          (static_cast<int64_t>(batch) * num_heads + head) * kHeadDim + dim;
      out[out_offset] = __float2bfloat16(acc * inv_sum[head_slot]);
    }
  }
}

__global__ void ds4_cuda_fused_v19_score_kernel(
    const __nv_bfloat16* __restrict__ q,
    const uint8_t* __restrict__ swa_cache,
    const int32_t* __restrict__ swa_indices,
    const int32_t* __restrict__ swa_lengths,
    int swa_width,
    int swa_page_size,
    int swa_row_stride,
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
    int total_width,
    int head_tiles,
    int row_tiles,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    float* __restrict__ score_state) {
  namespace wmma = nvcuda::wmma;

  const int batch = blockIdx.x;
  const int head_tile = blockIdx.y;
  const int row_tile = blockIdx.z;
  const int head_base = head_tile * kScoreTileM;
  const int tile_row_base = row_tile * kScoreRowsPerBlock;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  if (batch >= batch_size) {
    return;
  }

  __shared__ __align__(16) __nv_bfloat16 q_reuse_shared[kScoreKTileCount][kScoreTileElems];
  __shared__ __align__(16) __nv_bfloat16 k_shared[kScoreWarpsPerBlock][kScoreTileElems];
  __shared__ __align__(16) float score_shared[kScoreWarpsPerBlock][kScoreTileElems];
  __shared__ float row_scales[kScoreRowsPerBlock][kScaleCount];
  __shared__ const uint8_t* row_ptrs[kScoreRowsPerBlock];
  __shared__ uint8_t row_valid[kScoreRowsPerBlock];

  if (threadIdx.x < kScoreRowsPerBlock) {
    const int row = tile_row_base + threadIdx.x;
    bool valid = false;
    const uint8_t* packed = nullptr;
    if (row < total_width) {
      packed = selected_combined_row_ptr(
          swa_cache,
          swa_indices,
          swa_lengths,
          swa_width,
          swa_page_size,
          swa_row_stride,
          extra_cache,
          extra_indices,
          extra_lengths,
          extra_width,
          extra_page_size,
          extra_row_stride,
          has_extra,
          batch,
          row,
          valid);
    }
    row_ptrs[threadIdx.x] = packed;
    row_valid[threadIdx.x] = valid ? 1 : 0;
  }
  __syncthreads();

  if (threadIdx.x < kScoreRowsPerBlock * kScaleCount) {
    const int row_slot = threadIdx.x / kScaleCount;
    const int scale_slot = threadIdx.x - row_slot * kScaleCount;
    const uint8_t* packed = row_ptrs[row_slot];
    row_scales[row_slot][scale_slot] =
        row_valid[row_slot]
            ? exp2f(static_cast<float>(packed[kScaleOffset + scale_slot]) - 127.0f)
            : 0.0f;
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < kScoreKTileCount * kScoreTileElems;
       idx += blockDim.x) {
    const int dim_tile = idx / kScoreTileElems;
    const int tile_idx = idx - dim_tile * kScoreTileElems;
    const int q_head_slot = tile_idx / kScoreTileK;
    const int q_dim_slot = tile_idx - q_head_slot * kScoreTileK;
    const int head = head_base + q_head_slot;
    const int dim = dim_tile * kScoreTileK + q_dim_slot;
    const int64_t q_offset =
        (static_cast<int64_t>(batch) * num_heads + head) * kHeadDim + dim;
    q_reuse_shared[dim_tile][tile_idx] =
        head < num_heads ? q[q_offset] : __float2bfloat16(0.0f);
  }
  __syncthreads();

  if (warp < kScoreWarpsPerBlock) {
    wmma::fragment<wmma::matrix_a, kScoreTileM, kScoreTileN, kScoreTileK, __nv_bfloat16, wmma::row_major>
        q_frag;
    wmma::fragment<wmma::matrix_b, kScoreTileM, kScoreTileN, kScoreTileK, __nv_bfloat16, wmma::row_major>
        k_frag;
    wmma::fragment<wmma::accumulator, kScoreTileM, kScoreTileN, kScoreTileK, float>
        acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    const int warp_row_base = warp * kScoreTileN;
    for (int dim_base = 0; dim_base < kHeadDim; dim_base += kScoreTileK) {
      for (int idx = lane; idx < kScoreTileElems; idx += 32) {
        const int k_dim_slot = idx / kScoreTileN;
        const int k_row_slot = idx - k_dim_slot * kScoreTileN;
        const int local_row = warp_row_base + k_row_slot;
        const bool valid = row_valid[local_row] != 0;
        const uint8_t* packed = row_ptrs[local_row];
        float value = 0.0f;
        if (valid) {
          const int dim = dim_base + k_dim_slot;
          value = load_dsv4_packed_dim_with_scales(packed, dim, row_scales[local_row]);
        }
        k_shared[warp][idx] = __float2bfloat16(value);
      }
      __syncwarp();

      wmma::load_matrix_sync(
          q_frag, q_reuse_shared[dim_base / kScoreTileK], kScoreTileK);
      wmma::load_matrix_sync(k_frag, k_shared[warp], kScoreTileN);
      wmma::mma_sync(acc_frag, q_frag, k_frag, acc_frag);
      __syncwarp();
    }

    wmma::store_matrix_sync(score_shared[warp], acc_frag, kScoreTileN, wmma::mem_row_major);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < kScoreRowsPerBlock * kScoreTileM; idx += blockDim.x) {
    const int row_slot = idx / kScoreTileM;
    const int head_slot = idx - row_slot * kScoreTileM;
    const int row = tile_row_base + row_slot;
    const int head = head_base + head_slot;
    if (row < total_width && head < num_heads) {
      const int row_warp = row_slot / kScoreTileN;
      const int row_lane = row_slot - row_warp * kScoreTileN;
      const float score =
          row_valid[row_slot]
              ? score_shared[row_warp][head_slot * kScoreTileN + row_lane] * softmax_scale
              : 0.0f;
      score_state[(static_cast<int64_t>(batch) * num_heads + head) * total_width + row] =
          score;
    }
  }

  if (threadIdx.x < kScoreTileM) {
    const int head_slot = threadIdx.x;
    const int head = head_base + head_slot;
    float tile_max = -INFINITY;
#pragma unroll
    for (int row_slot = 0; row_slot < kScoreRowsPerBlock; ++row_slot) {
      if (head < num_heads && row_valid[row_slot]) {
        const int row_warp = row_slot / kScoreTileN;
        const int row_lane = row_slot - row_warp * kScoreTileN;
        const float score =
            score_shared[row_warp][head_slot * kScoreTileN + row_lane] * softmax_scale;
        tile_max = fmaxf(tile_max, score);
      }
    }

    float tile_sum = 0.0f;
#pragma unroll
    for (int row_slot = 0; row_slot < kScoreRowsPerBlock; ++row_slot) {
      if (head < num_heads && row_valid[row_slot]) {
        const int row_warp = row_slot / kScoreTileN;
        const int row_lane = row_slot - row_warp * kScoreTileN;
        const float score =
            score_shared[row_warp][head_slot * kScoreTileN + row_lane] * softmax_scale;
        tile_sum += expf(score - tile_max);
      }
    }

    const int64_t state_offset =
        (((static_cast<int64_t>(batch) * head_tiles + head_tile) * row_tiles + row_tile) *
             kScoreTileM +
         head_slot);
    partial_max[state_offset] = tile_max;
    partial_sum[state_offset] = tile_sum;
  }
}

__global__ void ds4_cuda_fused_v19_finalize_kernel(
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    const float* __restrict__ score_state,
    const float* __restrict__ attn_sink,
    bool has_sink,
    const uint8_t* __restrict__ swa_cache,
    const int32_t* __restrict__ swa_indices,
    const int32_t* __restrict__ swa_lengths,
    int swa_width,
    int swa_page_size,
    int swa_row_stride,
    const uint8_t* __restrict__ extra_cache,
    const int32_t* __restrict__ extra_indices,
    const int32_t* __restrict__ extra_lengths,
    int extra_width,
    int extra_page_size,
    int extra_row_stride,
    bool has_extra,
    int batch_size,
    int num_heads,
    int total_width,
    int head_tiles,
    int row_tiles,
    __nv_bfloat16* __restrict__ out) {
  namespace wmma = nvcuda::wmma;

  const int batch = blockIdx.x;
  const int head_tile = blockIdx.y;
  const int head_base = head_tile * kScoreTileM;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  if (batch >= batch_size) {
    return;
  }

  __shared__ float final_max[kScoreTileM];
  __shared__ float final_sum[kScoreTileM];
  __shared__ float inv_sum[kScoreTileM];
  __shared__ const uint8_t* row_ptrs[kScoreRowsPerBlock];
  __shared__ uint8_t row_valid[kScoreRowsPerBlock];
  __shared__ float row_scales[kScoreRowsPerBlock][kScaleCount];
  __shared__ __align__(16) __nv_bfloat16 p_shared[kScoreWarpsPerBlock][kScoreTileElems];
  __shared__ __align__(16) __nv_bfloat16 v_shared[kV11WarpsPerBlock][kScoreTileElems];
  __shared__ __align__(16) float pv_shared[kV11WarpsPerBlock][kScoreTileElems];

  if (threadIdx.x < kScoreTileM) {
    const int head_slot = threadIdx.x;
    const int head = head_base + head_slot;
    float m = -INFINITY;
    if (head < num_heads) {
      for (int row_tile = 0; row_tile < row_tiles; ++row_tile) {
        const int64_t state_offset =
            (((static_cast<int64_t>(batch) * head_tiles + head_tile) * row_tiles + row_tile) *
                 kScoreTileM +
             head_slot);
        const float l = partial_sum[state_offset];
        if (l > 0.0f) {
          m = fmaxf(m, partial_max[state_offset]);
        }
      }
      if (has_sink) {
        m = fmaxf(m, attn_sink[head]);
      }
    }

    float l_total = 0.0f;
    if (head < num_heads) {
      for (int row_tile = 0; row_tile < row_tiles; ++row_tile) {
        const int64_t state_offset =
            (((static_cast<int64_t>(batch) * head_tiles + head_tile) * row_tiles + row_tile) *
                 kScoreTileM +
             head_slot);
        const float l = partial_sum[state_offset];
        if (l > 0.0f) {
          l_total += l * expf(partial_max[state_offset] - m);
        }
      }
      if (has_sink) {
        l_total += expf(attn_sink[head] - m);
      }
    }
    final_max[head_slot] = m;
    final_sum[head_slot] = l_total;
    inv_sum[head_slot] = l_total > 0.0f ? 1.0f / l_total : 0.0f;
  }
  __syncthreads();

  for (int outer_round = 0; outer_round < kV15DimOuterRounds; ++outer_round) {
    if (warp < kV11WarpsPerBlock) {
      const int dim_round_group = warp / kV11DimGroups;
      const int dim_group = warp - dim_round_group * kV11DimGroups;
      const int dim_round = outer_round * kV15DimRoundGroups + dim_round_group;
      const int dim_base = (dim_round * kV11DimGroups + dim_group) * kScoreTileN;
      const bool nope_tile = dim_base < kNopeDim;
      const int scale_slot = dim_base / kScaleGroup;

      wmma::fragment<wmma::matrix_a, kScoreTileM, kScoreTileN, kScoreTileK, __nv_bfloat16, wmma::row_major>
          p_frag;
      wmma::fragment<wmma::matrix_b, kScoreTileM, kScoreTileN, kScoreTileK, __nv_bfloat16, wmma::row_major>
          v_frag;
      wmma::fragment<wmma::accumulator, kScoreTileM, kScoreTileN, kScoreTileK, float>
          pv_acc_frag;
      wmma::fill_fragment(pv_acc_frag, 0.0f);

      for (int row_tile = 0; row_tile < row_tiles; ++row_tile) {
        if (threadIdx.x < kScoreRowsPerBlock) {
          const int row = row_tile * kScoreRowsPerBlock + threadIdx.x;
          bool valid = false;
          const uint8_t* packed = nullptr;
          if (row < total_width) {
            packed = selected_combined_row_ptr(
                swa_cache,
                swa_indices,
                swa_lengths,
                swa_width,
                swa_page_size,
                swa_row_stride,
                extra_cache,
                extra_indices,
                extra_lengths,
                extra_width,
                extra_page_size,
                extra_row_stride,
                has_extra,
                batch,
                row,
                valid);
          }
          row_ptrs[threadIdx.x] = packed;
          row_valid[threadIdx.x] = valid ? 1 : 0;
        }
        __syncthreads();
        if (threadIdx.x < kScoreRowsPerBlock * kScaleCount) {
          const int row_slot = threadIdx.x / kScaleCount;
          const int scale_slot = threadIdx.x - row_slot * kScaleCount;
          const uint8_t* packed = row_ptrs[row_slot];
          row_scales[row_slot][scale_slot] =
              row_valid[row_slot]
                  ? exp2f(static_cast<float>(packed[kScaleOffset + scale_slot]) - 127.0f)
                  : 0.0f;
        }
        __syncthreads();

        if (warp < kScoreWarpsPerBlock) {
          const int row_group = warp;
          for (int idx = lane; idx < kScoreTileElems; idx += 32) {
            const int head_slot = idx / kScoreTileN;
            const int row_lane = idx - head_slot * kScoreTileN;
            const int row_slot = row_group * kScoreTileN + row_lane;
            const int row = row_tile * kScoreRowsPerBlock + row_slot;
            const int head = head_base + head_slot;
            float p = 0.0f;
            if (head < num_heads && row < total_width && row_valid[row_slot]) {
              const float score =
                  score_state[(static_cast<int64_t>(batch) * num_heads + head) * total_width +
                              row];
              p = expf(score - final_max[head_slot]);
            }
            p_shared[row_group][idx] = __float2bfloat16(p);
          }
        }
        __syncthreads();

#pragma unroll
        for (int row_group = 0; row_group < kScoreWarpsPerBlock; ++row_group) {
          for (int idx = lane; idx < kScoreTileElems; idx += 32) {
            const int k_slot = idx / kScoreTileN;
            const int dim_slot = idx - k_slot * kScoreTileN;
            const int row_slot = row_group * kScoreTileN + k_slot;
            const int dim = dim_base + dim_slot;
            float v = 0.0f;
            if (row_valid[row_slot]) {
              const uint8_t* row = row_ptrs[row_slot];
              if (nope_tile) {
                v = fp8_e4m3fn_to_float(row[dim]) *
                    row_scales[row_slot][scale_slot];
              } else {
                v = bf16_bytes_to_float(row + kNopeDim + (dim - kNopeDim) * 2);
              }
            }
            v_shared[warp][idx] = __float2bfloat16(v);
          }
          __syncwarp();

          wmma::load_matrix_sync(p_frag, p_shared[row_group], kScoreTileN);
          wmma::load_matrix_sync(v_frag, v_shared[warp], kScoreTileN);
          wmma::mma_sync(pv_acc_frag, p_frag, v_frag, pv_acc_frag);
          __syncwarp();
        }
        __syncthreads();
      }

      wmma::store_matrix_sync(pv_shared[warp], pv_acc_frag, kScoreTileN, wmma::mem_row_major);
    }
    __syncthreads();

    for (int idx = threadIdx.x;
         idx < kV15DimRoundGroups * kV11DimGroups * kScoreTileElems;
         idx += blockDim.x) {
      const int dim_round_group = idx / (kV11DimGroups * kScoreTileElems);
      const int rem = idx - dim_round_group * kV11DimGroups * kScoreTileElems;
      const int dim_group = rem / kScoreTileElems;
      const int tile_idx = rem - dim_group * kScoreTileElems;
      const int head_slot = tile_idx / kScoreTileN;
      const int dim_slot = tile_idx - head_slot * kScoreTileN;
      const int head = head_base + head_slot;
      const int dim_round = outer_round * kV15DimRoundGroups + dim_round_group;
      const int dim = (dim_round * kV11DimGroups + dim_group) * kScoreTileN + dim_slot;
      const int result_warp = dim_round_group * kV11DimGroups + dim_group;
      if (head < num_heads) {
        const float value = pv_shared[result_warp][tile_idx] * inv_sum[head_slot];
        const int64_t out_offset =
            (static_cast<int64_t>(batch) * num_heads + head) * kHeadDim + dim;
        out[out_offset] = __float2bfloat16(value);
      }
    }
    __syncthreads();
  }
}

__global__ void ds4_cuda_fused_v20_finalize_kernel(
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    const float* __restrict__ score_state,
    const float* __restrict__ attn_sink,
    bool has_sink,
    const uint8_t* __restrict__ swa_cache,
    const int32_t* __restrict__ swa_indices,
    const int32_t* __restrict__ swa_lengths,
    int swa_width,
    int swa_page_size,
    int swa_row_stride,
    const uint8_t* __restrict__ extra_cache,
    const int32_t* __restrict__ extra_indices,
    const int32_t* __restrict__ extra_lengths,
    int extra_width,
    int extra_page_size,
    int extra_row_stride,
    bool has_extra,
    int batch_size,
    int num_heads,
    int total_width,
    int head_tiles,
    int row_tiles,
    __nv_bfloat16* __restrict__ out) {
  namespace wmma = nvcuda::wmma;

  const int batch = blockIdx.x;
  const int head_tile = blockIdx.y;
  const int outer_round = blockIdx.z;
  const int head_base = head_tile * kScoreTileM;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  if (batch >= batch_size || outer_round >= kV15DimOuterRounds) {
    return;
  }

  __shared__ float final_max[kScoreTileM];
  __shared__ float final_sum[kScoreTileM];
  __shared__ float inv_sum[kScoreTileM];
  __shared__ const uint8_t* row_ptrs[kScoreRowsPerBlock];
  __shared__ uint8_t row_valid[kScoreRowsPerBlock];
  __shared__ float row_scales[kScoreRowsPerBlock][kScaleCount];
  __shared__ __align__(16) __nv_bfloat16 p_shared[kScoreWarpsPerBlock][kScoreTileElems];
  __shared__ __align__(16) __nv_bfloat16 v_shared[kV11WarpsPerBlock][kScoreTileElems];
  __shared__ __align__(16) float pv_shared[kV11WarpsPerBlock][kScoreTileElems];

  if (threadIdx.x < kScoreTileM) {
    const int head_slot = threadIdx.x;
    const int head = head_base + head_slot;
    float m = -INFINITY;
    if (head < num_heads) {
      for (int row_tile = 0; row_tile < row_tiles; ++row_tile) {
        const int64_t state_offset =
            (((static_cast<int64_t>(batch) * head_tiles + head_tile) * row_tiles + row_tile) *
                 kScoreTileM +
             head_slot);
        const float l = partial_sum[state_offset];
        if (l > 0.0f) {
          m = fmaxf(m, partial_max[state_offset]);
        }
      }
      if (has_sink) {
        m = fmaxf(m, attn_sink[head]);
      }
    }

    float l_total = 0.0f;
    if (head < num_heads) {
      for (int row_tile = 0; row_tile < row_tiles; ++row_tile) {
        const int64_t state_offset =
            (((static_cast<int64_t>(batch) * head_tiles + head_tile) * row_tiles + row_tile) *
                 kScoreTileM +
             head_slot);
        const float l = partial_sum[state_offset];
        if (l > 0.0f) {
          l_total += l * expf(partial_max[state_offset] - m);
        }
      }
      if (has_sink) {
        l_total += expf(attn_sink[head] - m);
      }
    }
    final_max[head_slot] = m;
    final_sum[head_slot] = l_total;
    inv_sum[head_slot] = l_total > 0.0f ? 1.0f / l_total : 0.0f;
  }
  __syncthreads();

  if (warp < kV11WarpsPerBlock) {
    const int dim_round_group = warp / kV11DimGroups;
    const int dim_group = warp - dim_round_group * kV11DimGroups;
    const int dim_round = outer_round * kV15DimRoundGroups + dim_round_group;
    const int dim_base = (dim_round * kV11DimGroups + dim_group) * kScoreTileN;
    const bool nope_tile = dim_base < kNopeDim;
    const int scale_slot = dim_base / kScaleGroup;

    wmma::fragment<wmma::matrix_a, kScoreTileM, kScoreTileN, kScoreTileK, __nv_bfloat16, wmma::row_major>
        p_frag;
    wmma::fragment<wmma::matrix_b, kScoreTileM, kScoreTileN, kScoreTileK, __nv_bfloat16, wmma::row_major>
        v_frag;
    wmma::fragment<wmma::accumulator, kScoreTileM, kScoreTileN, kScoreTileK, float>
        pv_acc_frag;
    wmma::fill_fragment(pv_acc_frag, 0.0f);

    for (int row_tile = 0; row_tile < row_tiles; ++row_tile) {
      if (threadIdx.x < kScoreRowsPerBlock) {
        const int row = row_tile * kScoreRowsPerBlock + threadIdx.x;
        bool valid = false;
        const uint8_t* packed = nullptr;
        if (row < total_width) {
          packed = selected_combined_row_ptr(
              swa_cache,
              swa_indices,
              swa_lengths,
              swa_width,
              swa_page_size,
              swa_row_stride,
              extra_cache,
              extra_indices,
              extra_lengths,
              extra_width,
              extra_page_size,
              extra_row_stride,
              has_extra,
              batch,
              row,
              valid);
        }
        row_ptrs[threadIdx.x] = packed;
        row_valid[threadIdx.x] = valid ? 1 : 0;
      }
      __syncthreads();
      if (threadIdx.x < kScoreRowsPerBlock * kScaleCount) {
        const int row_slot = threadIdx.x / kScaleCount;
        const int local_scale_slot = threadIdx.x - row_slot * kScaleCount;
        const uint8_t* packed = row_ptrs[row_slot];
        row_scales[row_slot][local_scale_slot] =
            row_valid[row_slot]
                ? exp2f(static_cast<float>(packed[kScaleOffset + local_scale_slot]) - 127.0f)
                : 0.0f;
      }
      __syncthreads();

      if (warp < kScoreWarpsPerBlock) {
        const int row_group = warp;
        for (int idx = lane; idx < kScoreTileElems; idx += 32) {
          const int head_slot = idx / kScoreTileN;
          const int row_lane = idx - head_slot * kScoreTileN;
          const int row_slot = row_group * kScoreTileN + row_lane;
          const int row = row_tile * kScoreRowsPerBlock + row_slot;
          const int head = head_base + head_slot;
          float p = 0.0f;
          if (head < num_heads && row < total_width && row_valid[row_slot]) {
            const float score =
                score_state[(static_cast<int64_t>(batch) * num_heads + head) * total_width +
                            row];
            p = expf(score - final_max[head_slot]);
          }
          p_shared[row_group][idx] = __float2bfloat16(p);
        }
      }
      __syncthreads();

#pragma unroll
      for (int row_group = 0; row_group < kScoreWarpsPerBlock; ++row_group) {
        for (int idx = lane; idx < kScoreTileElems; idx += 32) {
          const int k_slot = idx / kScoreTileN;
          const int dim_slot = idx - k_slot * kScoreTileN;
          const int row_slot = row_group * kScoreTileN + k_slot;
          const int dim = dim_base + dim_slot;
          float v = 0.0f;
          if (row_valid[row_slot]) {
            const uint8_t* row = row_ptrs[row_slot];
            if (nope_tile) {
              v = fp8_e4m3fn_to_float(row[dim]) * row_scales[row_slot][scale_slot];
            } else {
              v = bf16_bytes_to_float(row + kNopeDim + (dim - kNopeDim) * 2);
            }
          }
          v_shared[warp][idx] = __float2bfloat16(v);
        }
        __syncwarp();

        wmma::load_matrix_sync(p_frag, p_shared[row_group], kScoreTileN);
        wmma::load_matrix_sync(v_frag, v_shared[warp], kScoreTileN);
        wmma::mma_sync(pv_acc_frag, p_frag, v_frag, pv_acc_frag);
        __syncwarp();
      }
      __syncthreads();
    }

    wmma::store_matrix_sync(pv_shared[warp], pv_acc_frag, kScoreTileN, wmma::mem_row_major);
  }
  __syncthreads();

  for (int idx = threadIdx.x;
       idx < kV15DimRoundGroups * kV11DimGroups * kScoreTileElems;
       idx += blockDim.x) {
    const int dim_round_group = idx / (kV11DimGroups * kScoreTileElems);
    const int rem = idx - dim_round_group * kV11DimGroups * kScoreTileElems;
    const int dim_group = rem / kScoreTileElems;
    const int tile_idx = rem - dim_group * kScoreTileElems;
    const int head_slot = tile_idx / kScoreTileN;
    const int dim_slot = tile_idx - head_slot * kScoreTileN;
    const int head = head_base + head_slot;
    const int dim_round = outer_round * kV15DimRoundGroups + dim_round_group;
    const int dim = (dim_round * kV11DimGroups + dim_group) * kScoreTileN + dim_slot;
    const int result_warp = dim_round_group * kV11DimGroups + dim_group;
    if (head < num_heads) {
      const float value = pv_shared[result_warp][tile_idx] * inv_sum[head_slot];
      const int64_t out_offset =
          (static_cast<int64_t>(batch) * num_heads + head) * kHeadDim + dim;
      out[out_offset] = __float2bfloat16(value);
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

  const int64_t total_width = swa_width + extra_width;
  c10::cuda::CUDAGuard device_guard(q.device());
  auto out = torch::empty_like(q);

  dim3 grid(batch_size, num_heads);
  dim3 grouped_grid(
      static_cast<unsigned int>(batch_size),
      static_cast<unsigned int>((num_heads + kV4HeadsPerBlock - 1) / kV4HeadsPerBlock));
  dim3 fused_grid(
      static_cast<unsigned int>(batch_size),
      static_cast<unsigned int>((num_heads + kScoreTileM - 1) / kScoreTileM));
  dim3 block(kThreads);
  dim3 score_block(kScoreThreads);
  dim3 fused_block(kV7Threads);
  dim3 split_block(kV8Threads);
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
    case AttentionVariant::kOptimizedV7:
      ds4_cuda_fused_v7_mma_attention_kernel
          <<<fused_grid, fused_block, 0, at::cuda::getCurrentCUDAStream()>>>(
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
              static_cast<int>(total_width),
              reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()));
      break;
    case AttentionVariant::kOptimizedV8:
    case AttentionVariant::kOptimizedV9:
    case AttentionVariant::kOptimizedV10:
    case AttentionVariant::kOptimizedV11:
    case AttentionVariant::kOptimizedV12:
    case AttentionVariant::kOptimizedV13:
    case AttentionVariant::kOptimizedV14:
    case AttentionVariant::kOptimizedV15:
    case AttentionVariant::kOptimizedV16:
    case AttentionVariant::kOptimizedV17:
    case AttentionVariant::kOptimizedV18: {
      const bool cache_scales = variant != AttentionVariant::kOptimizedV8;
      const bool tensor_core_pv = variant == AttentionVariant::kOptimizedV10;
      const bool tensor_core_pv_parallel = variant == AttentionVariant::kOptimizedV11;
      const bool tensor_core_pv_cached = variant == AttentionVariant::kOptimizedV12;
      const bool tensor_core_pv_cached_bf16_partial =
          variant == AttentionVariant::kOptimizedV13;
      const bool tensor_core_pv_specialized_v_decode =
          variant == AttentionVariant::kOptimizedV14;
      const bool tensor_core_pv_rowgroup_accum =
          variant == AttentionVariant::kOptimizedV15 ||
          variant == AttentionVariant::kOptimizedV16 ||
          variant == AttentionVariant::kOptimizedV17 ||
          variant == AttentionVariant::kOptimizedV18;
      const bool tensor_core_pv_coalesced_k_staging =
          variant == AttentionVariant::kOptimizedV16;
      const bool tensor_core_pv_q_tile_reuse =
          variant == AttentionVariant::kOptimizedV17 ||
          variant == AttentionVariant::kOptimizedV18;
      const bool tensor_core_pv_reduce_scale_reuse =
          variant == AttentionVariant::kOptimizedV18;
      const int64_t head_tiles = (num_heads + kScoreTileM - 1) / kScoreTileM;
      const int64_t row_tiles = (total_width + kScoreRowsPerBlock - 1) / kScoreRowsPerBlock;
      auto partial_max = torch::empty(
          {batch_size, head_tiles, row_tiles, static_cast<int64_t>(kScoreTileM)},
          q.options().dtype(torch::kFloat32));
      auto partial_sum = torch::empty_like(partial_max);
      const auto partial_acc_dtype =
          (tensor_core_pv_cached_bf16_partial ||
           tensor_core_pv_specialized_v_decode ||
           tensor_core_pv_rowgroup_accum)
          ? torch::kBFloat16
          : torch::kFloat32;
      auto partial_acc = torch::empty(
          {batch_size,
           head_tiles,
           row_tiles,
           static_cast<int64_t>(kScoreTileM),
           static_cast<int64_t>(kHeadDim)},
          q.options().dtype(partial_acc_dtype));
      const bool profile_partial_stages =
          partial_stage_profile_enabled() && tensor_core_pv_rowgroup_accum;
      torch::Tensor partial_stage_profile;
      unsigned long long* partial_stage_profile_ptr = nullptr;
      if (profile_partial_stages) {
        partial_stage_profile = torch::zeros(
            {static_cast<int64_t>(kPartialProfileSlots)},
            q.options().dtype(torch::kInt64));
        partial_stage_profile_ptr = reinterpret_cast<unsigned long long*>(
            partial_stage_profile.data_ptr<int64_t>());
      }
      dim3 split_grid(
          static_cast<unsigned int>(batch_size),
          static_cast<unsigned int>(head_tiles),
          static_cast<unsigned int>(row_tiles));
      const auto stream = at::cuda::getCurrentCUDAStream();
      const bool profile_split = split_profile_enabled();
      cudaEvent_t partial_start = nullptr;
      cudaEvent_t partial_stop = nullptr;
      cudaEvent_t reduce_stop = nullptr;
      if (profile_split) {
        C10_CUDA_CHECK(cudaEventCreate(&partial_start));
        C10_CUDA_CHECK(cudaEventCreate(&partial_stop));
        C10_CUDA_CHECK(cudaEventCreate(&reduce_stop));
        C10_CUDA_CHECK(cudaEventRecord(partial_start, stream));
      }
      if (tensor_core_pv_rowgroup_accum) {
        if (tensor_core_pv_q_tile_reuse) {
          if (profile_partial_stages) {
            ds4_cuda_fused_v8_mma_partial_kernel<true, 7, __nv_bfloat16, true>
                <<<split_grid, split_block, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
                    swa_k_cache.data_ptr<uint8_t>(),
                    swa_indices.data_ptr<int32_t>(),
                    swa_topk_lengths.data_ptr<int32_t>(),
                    static_cast<int>(swa_width),
                    static_cast<int>(swa_page_size),
                    static_cast<int>(swa_k_cache.size(3)),
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
                    static_cast<int>(total_width),
                    static_cast<int>(head_tiles),
                    static_cast<int>(row_tiles),
                    partial_max.data_ptr<float>(),
                    partial_sum.data_ptr<float>(),
                    reinterpret_cast<__nv_bfloat16*>(partial_acc.data_ptr<at::BFloat16>()),
                    partial_stage_profile_ptr);
          } else {
            ds4_cuda_fused_v8_mma_partial_kernel<true, 7, __nv_bfloat16>
                <<<split_grid, split_block, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
                    swa_k_cache.data_ptr<uint8_t>(),
                    swa_indices.data_ptr<int32_t>(),
                    swa_topk_lengths.data_ptr<int32_t>(),
                    static_cast<int>(swa_width),
                    static_cast<int>(swa_page_size),
                    static_cast<int>(swa_k_cache.size(3)),
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
                    static_cast<int>(total_width),
                    static_cast<int>(head_tiles),
                    static_cast<int>(row_tiles),
                    partial_max.data_ptr<float>(),
                    partial_sum.data_ptr<float>(),
                    reinterpret_cast<__nv_bfloat16*>(partial_acc.data_ptr<at::BFloat16>()),
                    nullptr);
          }
        } else if (tensor_core_pv_coalesced_k_staging) {
          if (profile_partial_stages) {
            ds4_cuda_fused_v8_mma_partial_kernel<true, 6, __nv_bfloat16, true>
                <<<split_grid, split_block, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
                    swa_k_cache.data_ptr<uint8_t>(),
                    swa_indices.data_ptr<int32_t>(),
                    swa_topk_lengths.data_ptr<int32_t>(),
                    static_cast<int>(swa_width),
                    static_cast<int>(swa_page_size),
                    static_cast<int>(swa_k_cache.size(3)),
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
                    static_cast<int>(total_width),
                    static_cast<int>(head_tiles),
                    static_cast<int>(row_tiles),
                    partial_max.data_ptr<float>(),
                    partial_sum.data_ptr<float>(),
                    reinterpret_cast<__nv_bfloat16*>(partial_acc.data_ptr<at::BFloat16>()),
                    partial_stage_profile_ptr);
          } else {
            ds4_cuda_fused_v8_mma_partial_kernel<true, 6, __nv_bfloat16>
                <<<split_grid, split_block, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
                    swa_k_cache.data_ptr<uint8_t>(),
                    swa_indices.data_ptr<int32_t>(),
                    swa_topk_lengths.data_ptr<int32_t>(),
                    static_cast<int>(swa_width),
                    static_cast<int>(swa_page_size),
                    static_cast<int>(swa_k_cache.size(3)),
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
                    static_cast<int>(total_width),
                    static_cast<int>(head_tiles),
                    static_cast<int>(row_tiles),
                    partial_max.data_ptr<float>(),
                    partial_sum.data_ptr<float>(),
                    reinterpret_cast<__nv_bfloat16*>(partial_acc.data_ptr<at::BFloat16>()),
                    nullptr);
          }
        } else if (profile_partial_stages) {
          ds4_cuda_fused_v8_mma_partial_kernel<true, 5, __nv_bfloat16, true>
              <<<split_grid, split_block, 0, stream>>>(
                  reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
                  swa_k_cache.data_ptr<uint8_t>(),
                  swa_indices.data_ptr<int32_t>(),
                  swa_topk_lengths.data_ptr<int32_t>(),
                  static_cast<int>(swa_width),
                  static_cast<int>(swa_page_size),
                  static_cast<int>(swa_k_cache.size(3)),
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
                  static_cast<int>(total_width),
                  static_cast<int>(head_tiles),
                  static_cast<int>(row_tiles),
                  partial_max.data_ptr<float>(),
                  partial_sum.data_ptr<float>(),
                  reinterpret_cast<__nv_bfloat16*>(partial_acc.data_ptr<at::BFloat16>()),
                  partial_stage_profile_ptr);
        } else {
          ds4_cuda_fused_v8_mma_partial_kernel<true, 5, __nv_bfloat16>
              <<<split_grid, split_block, 0, stream>>>(
                  reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
                  swa_k_cache.data_ptr<uint8_t>(),
                  swa_indices.data_ptr<int32_t>(),
                  swa_topk_lengths.data_ptr<int32_t>(),
                  static_cast<int>(swa_width),
                  static_cast<int>(swa_page_size),
                  static_cast<int>(swa_k_cache.size(3)),
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
                  static_cast<int>(total_width),
                  static_cast<int>(head_tiles),
                  static_cast<int>(row_tiles),
                  partial_max.data_ptr<float>(),
                  partial_sum.data_ptr<float>(),
                  reinterpret_cast<__nv_bfloat16*>(partial_acc.data_ptr<at::BFloat16>()),
                  nullptr);
        }
      } else if (tensor_core_pv_specialized_v_decode) {
        ds4_cuda_fused_v8_mma_partial_kernel<true, 4, __nv_bfloat16>
            <<<split_grid, split_block, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
                swa_k_cache.data_ptr<uint8_t>(),
                swa_indices.data_ptr<int32_t>(),
                swa_topk_lengths.data_ptr<int32_t>(),
                static_cast<int>(swa_width),
                static_cast<int>(swa_page_size),
                static_cast<int>(swa_k_cache.size(3)),
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
                static_cast<int>(total_width),
                static_cast<int>(head_tiles),
                static_cast<int>(row_tiles),
                partial_max.data_ptr<float>(),
                partial_sum.data_ptr<float>(),
                reinterpret_cast<__nv_bfloat16*>(partial_acc.data_ptr<at::BFloat16>()),
                nullptr);
      } else if (tensor_core_pv_cached_bf16_partial) {
        ds4_cuda_fused_v8_mma_partial_kernel<true, 3, __nv_bfloat16>
            <<<split_grid, split_block, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
                swa_k_cache.data_ptr<uint8_t>(),
                swa_indices.data_ptr<int32_t>(),
                swa_topk_lengths.data_ptr<int32_t>(),
                static_cast<int>(swa_width),
                static_cast<int>(swa_page_size),
                static_cast<int>(swa_k_cache.size(3)),
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
                static_cast<int>(total_width),
                static_cast<int>(head_tiles),
                static_cast<int>(row_tiles),
                partial_max.data_ptr<float>(),
                partial_sum.data_ptr<float>(),
                reinterpret_cast<__nv_bfloat16*>(partial_acc.data_ptr<at::BFloat16>()),
                nullptr);
      } else if (tensor_core_pv_cached) {
        ds4_cuda_fused_v8_mma_partial_kernel<true, 3>
            <<<split_grid, split_block, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
                swa_k_cache.data_ptr<uint8_t>(),
                swa_indices.data_ptr<int32_t>(),
                swa_topk_lengths.data_ptr<int32_t>(),
                static_cast<int>(swa_width),
                static_cast<int>(swa_page_size),
                static_cast<int>(swa_k_cache.size(3)),
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
                static_cast<int>(total_width),
                static_cast<int>(head_tiles),
                static_cast<int>(row_tiles),
                partial_max.data_ptr<float>(),
                partial_sum.data_ptr<float>(),
                partial_acc.data_ptr<float>(),
                nullptr);
      } else if (tensor_core_pv_parallel) {
        ds4_cuda_fused_v8_mma_partial_kernel<true, 2>
            <<<split_grid, split_block, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
                swa_k_cache.data_ptr<uint8_t>(),
                swa_indices.data_ptr<int32_t>(),
                swa_topk_lengths.data_ptr<int32_t>(),
                static_cast<int>(swa_width),
                static_cast<int>(swa_page_size),
                static_cast<int>(swa_k_cache.size(3)),
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
                static_cast<int>(total_width),
                static_cast<int>(head_tiles),
                static_cast<int>(row_tiles),
                partial_max.data_ptr<float>(),
                partial_sum.data_ptr<float>(),
                partial_acc.data_ptr<float>(),
                nullptr);
      } else if (tensor_core_pv) {
        ds4_cuda_fused_v8_mma_partial_kernel<true, 1>
            <<<split_grid, split_block, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
                swa_k_cache.data_ptr<uint8_t>(),
                swa_indices.data_ptr<int32_t>(),
                swa_topk_lengths.data_ptr<int32_t>(),
                static_cast<int>(swa_width),
                static_cast<int>(swa_page_size),
                static_cast<int>(swa_k_cache.size(3)),
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
                static_cast<int>(total_width),
                static_cast<int>(head_tiles),
                static_cast<int>(row_tiles),
                partial_max.data_ptr<float>(),
                partial_sum.data_ptr<float>(),
                partial_acc.data_ptr<float>(),
                nullptr);
      } else if (cache_scales) {
        ds4_cuda_fused_v8_mma_partial_kernel<true, 0>
            <<<split_grid, split_block, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
                swa_k_cache.data_ptr<uint8_t>(),
                swa_indices.data_ptr<int32_t>(),
                swa_topk_lengths.data_ptr<int32_t>(),
                static_cast<int>(swa_width),
                static_cast<int>(swa_page_size),
                static_cast<int>(swa_k_cache.size(3)),
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
                static_cast<int>(total_width),
                static_cast<int>(head_tiles),
                static_cast<int>(row_tiles),
                partial_max.data_ptr<float>(),
                partial_sum.data_ptr<float>(),
                partial_acc.data_ptr<float>(),
                nullptr);
      } else {
        ds4_cuda_fused_v8_mma_partial_kernel<false, 0>
            <<<split_grid, split_block, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
                swa_k_cache.data_ptr<uint8_t>(),
                swa_indices.data_ptr<int32_t>(),
                swa_topk_lengths.data_ptr<int32_t>(),
                static_cast<int>(swa_width),
                static_cast<int>(swa_page_size),
                static_cast<int>(swa_k_cache.size(3)),
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
                static_cast<int>(total_width),
                static_cast<int>(head_tiles),
                static_cast<int>(row_tiles),
                partial_max.data_ptr<float>(),
                partial_sum.data_ptr<float>(),
                partial_acc.data_ptr<float>(),
                nullptr);
      }
      if (profile_split) {
        C10_CUDA_CHECK(cudaEventRecord(partial_stop, stream));
      }
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      if (tensor_core_pv_cached_bf16_partial ||
          tensor_core_pv_specialized_v_decode ||
          tensor_core_pv_rowgroup_accum) {
        if (tensor_core_pv_reduce_scale_reuse) {
          ds4_cuda_fused_v8_reduce_kernel<__nv_bfloat16, true>
              <<<fused_grid, split_block, 0, stream>>>(
                  partial_max.data_ptr<float>(),
                  partial_sum.data_ptr<float>(),
                  reinterpret_cast<const __nv_bfloat16*>(partial_acc.data_ptr<at::BFloat16>()),
                  has_sink ? attn_sink.data_ptr<float>() : nullptr,
                  has_sink,
                  static_cast<int>(batch_size),
                  static_cast<int>(num_heads),
                  static_cast<int>(head_tiles),
                  static_cast<int>(row_tiles),
                  reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()));
        } else {
          ds4_cuda_fused_v8_reduce_kernel<__nv_bfloat16>
              <<<fused_grid, split_block, 0, stream>>>(
                  partial_max.data_ptr<float>(),
                  partial_sum.data_ptr<float>(),
                  reinterpret_cast<const __nv_bfloat16*>(partial_acc.data_ptr<at::BFloat16>()),
                  has_sink ? attn_sink.data_ptr<float>() : nullptr,
                  has_sink,
                  static_cast<int>(batch_size),
                  static_cast<int>(num_heads),
                  static_cast<int>(head_tiles),
                  static_cast<int>(row_tiles),
                  reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()));
        }
      } else {
        ds4_cuda_fused_v8_reduce_kernel<float>
            <<<fused_grid, split_block, 0, stream>>>(
                partial_max.data_ptr<float>(),
                partial_sum.data_ptr<float>(),
                partial_acc.data_ptr<float>(),
                has_sink ? attn_sink.data_ptr<float>() : nullptr,
                has_sink,
                static_cast<int>(batch_size),
                static_cast<int>(num_heads),
                static_cast<int>(head_tiles),
                static_cast<int>(row_tiles),
                reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()));
      }
      if (profile_split) {
        C10_CUDA_CHECK(cudaEventRecord(reduce_stop, stream));
      }
      if (profile_partial_stages) {
        if (profile_split) {
          C10_CUDA_CHECK(cudaEventSynchronize(reduce_stop));
        } else {
          C10_CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        auto partial_stage_profile_cpu = partial_stage_profile.cpu();
        const int64_t* profile_values = partial_stage_profile_cpu.data_ptr<int64_t>();
        const double profile_total = static_cast<double>(
            profile_values[kPartialProfileRowSetup] +
            profile_values[kPartialProfileScaleCache] +
            profile_values[kPartialProfileQk] +
            profile_values[kPartialProfileSoftmax] +
            profile_values[kPartialProfilePCache] +
            profile_values[kPartialProfilePvMma] +
            profile_values[kPartialProfilePartialStore]);
        const double inv_total = profile_total > 0.0 ? 100.0 / profile_total : 0.0;
        const double qk_total = static_cast<double>(profile_values[kPartialProfileQk]);
        const double inv_qk = qk_total > 0.0 ? 100.0 / qk_total : 0.0;
        std::fprintf(
            stderr,
            "DSV4_CUDA_PARTIAL_STAGE_PROFILE variant=%s batch=%lld heads=%lld "
            "total_width=%lld head_tiles=%lld row_tiles=%lld blocks=%lld "
            "row_setup_cycles=%lld scale_cache_cycles=%lld qk_cycles=%lld "
            "softmax_cycles=%lld p_cache_cycles=%lld pv_mma_cycles=%lld "
            "partial_store_cycles=%lld row_setup_pct=%.2f scale_cache_pct=%.2f "
            "qk_pct=%.2f softmax_pct=%.2f p_cache_pct=%.2f pv_mma_pct=%.2f "
            "partial_store_pct=%.2f qk_staging_wall_cycles=%lld "
            "qk_mma_wall_cycles=%lld qk_score_store_wall_cycles=%lld "
            "qk_block_barrier_cycles=%lld qk_thread_q_load_cycles=%lld "
            "qk_thread_k_decode_cycles=%lld qk_staging_wall_pct_of_qk=%.2f "
            "qk_mma_wall_pct_of_qk=%.2f qk_score_store_wall_pct_of_qk=%.2f "
            "qk_block_barrier_pct_of_qk=%.2f\n",
            attention_variant_name(variant),
            static_cast<long long>(batch_size),
            static_cast<long long>(num_heads),
            static_cast<long long>(total_width),
            static_cast<long long>(head_tiles),
            static_cast<long long>(row_tiles),
            static_cast<long long>(profile_values[kPartialProfileBlocks]),
            static_cast<long long>(profile_values[kPartialProfileRowSetup]),
            static_cast<long long>(profile_values[kPartialProfileScaleCache]),
            static_cast<long long>(profile_values[kPartialProfileQk]),
            static_cast<long long>(profile_values[kPartialProfileSoftmax]),
            static_cast<long long>(profile_values[kPartialProfilePCache]),
            static_cast<long long>(profile_values[kPartialProfilePvMma]),
            static_cast<long long>(profile_values[kPartialProfilePartialStore]),
            static_cast<double>(profile_values[kPartialProfileRowSetup]) * inv_total,
            static_cast<double>(profile_values[kPartialProfileScaleCache]) * inv_total,
            static_cast<double>(profile_values[kPartialProfileQk]) * inv_total,
            static_cast<double>(profile_values[kPartialProfileSoftmax]) * inv_total,
            static_cast<double>(profile_values[kPartialProfilePCache]) * inv_total,
            static_cast<double>(profile_values[kPartialProfilePvMma]) * inv_total,
            static_cast<double>(profile_values[kPartialProfilePartialStore]) * inv_total,
            static_cast<long long>(profile_values[kPartialProfileQkStagingWall]),
            static_cast<long long>(profile_values[kPartialProfileQkMmaWall]),
            static_cast<long long>(profile_values[kPartialProfileQkScoreStoreWall]),
            static_cast<long long>(profile_values[kPartialProfileQkBlockBarrier]),
            static_cast<long long>(profile_values[kPartialProfileQkThreadQLoad]),
            static_cast<long long>(profile_values[kPartialProfileQkThreadKDecode]),
            static_cast<double>(profile_values[kPartialProfileQkStagingWall]) * inv_qk,
            static_cast<double>(profile_values[kPartialProfileQkMmaWall]) * inv_qk,
            static_cast<double>(profile_values[kPartialProfileQkScoreStoreWall]) * inv_qk,
            static_cast<double>(profile_values[kPartialProfileQkBlockBarrier]) * inv_qk);
      }
      if (profile_split) {
        if (!profile_partial_stages) {
          C10_CUDA_CHECK(cudaEventSynchronize(reduce_stop));
        }
        float partial_ms = 0.0f;
        float reduce_ms = 0.0f;
        C10_CUDA_CHECK(cudaEventElapsedTime(&partial_ms, partial_start, partial_stop));
        C10_CUDA_CHECK(cudaEventElapsedTime(&reduce_ms, partial_stop, reduce_stop));
        const char* partial_acc_dtype_name =
            partial_acc_dtype == torch::kBFloat16 ? "bf16" : "fp32";
        std::fprintf(
            stderr,
            "DSV4_CUDA_SPLIT_PROFILE variant=%s batch=%lld heads=%lld total_width=%lld "
            "head_tiles=%lld row_tiles=%lld partial_acc_dtype=%s partial_ms=%.6f "
            "reduce_ms=%.6f kernel_ms=%.6f\n",
            attention_variant_name(variant),
            static_cast<long long>(batch_size),
            static_cast<long long>(num_heads),
            static_cast<long long>(total_width),
            static_cast<long long>(head_tiles),
            static_cast<long long>(row_tiles),
            partial_acc_dtype_name,
            partial_ms,
            reduce_ms,
            partial_ms + reduce_ms);
        C10_CUDA_CHECK(cudaEventDestroy(partial_start));
        C10_CUDA_CHECK(cudaEventDestroy(partial_stop));
        C10_CUDA_CHECK(cudaEventDestroy(reduce_stop));
      }
      break;
    }
    case AttentionVariant::kOptimizedV19:
    case AttentionVariant::kOptimizedV20: {
      const int64_t head_tiles = (num_heads + kScoreTileM - 1) / kScoreTileM;
      const int64_t row_tiles = (total_width + kScoreRowsPerBlock - 1) / kScoreRowsPerBlock;
      TORCH_CHECK(total_width > 0, "score-state split requires selected rows");
      auto partial_max = torch::empty(
          {batch_size, head_tiles, row_tiles, static_cast<int64_t>(kScoreTileM)},
          q.options().dtype(torch::kFloat32));
      auto partial_sum = torch::empty_like(partial_max);
      auto score_state = torch::empty(
          {batch_size, num_heads, total_width},
          q.options().dtype(torch::kFloat32));

      dim3 split_grid(
          static_cast<unsigned int>(batch_size),
          static_cast<unsigned int>(head_tiles),
          static_cast<unsigned int>(row_tiles));
      const auto stream = at::cuda::getCurrentCUDAStream();
      const bool profile_split = split_profile_enabled();
      cudaEvent_t score_start = nullptr;
      cudaEvent_t score_stop = nullptr;
      cudaEvent_t finalize_stop = nullptr;
      if (profile_split) {
        C10_CUDA_CHECK(cudaEventCreate(&score_start));
        C10_CUDA_CHECK(cudaEventCreate(&score_stop));
        C10_CUDA_CHECK(cudaEventCreate(&finalize_stop));
        C10_CUDA_CHECK(cudaEventRecord(score_start, stream));
      }

      ds4_cuda_fused_v19_score_kernel<<<split_grid, split_block, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
          swa_k_cache.data_ptr<uint8_t>(),
          swa_indices.data_ptr<int32_t>(),
          swa_topk_lengths.data_ptr<int32_t>(),
          static_cast<int>(swa_width),
          static_cast<int>(swa_page_size),
          static_cast<int>(swa_k_cache.size(3)),
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
          static_cast<int>(total_width),
          static_cast<int>(head_tiles),
          static_cast<int>(row_tiles),
          partial_max.data_ptr<float>(),
          partial_sum.data_ptr<float>(),
          score_state.data_ptr<float>());
      if (profile_split) {
        C10_CUDA_CHECK(cudaEventRecord(score_stop, stream));
      }
      C10_CUDA_KERNEL_LAUNCH_CHECK();

      if (variant == AttentionVariant::kOptimizedV20) {
        dim3 finalize_grid(
            static_cast<unsigned int>(batch_size),
            static_cast<unsigned int>(head_tiles),
            static_cast<unsigned int>(kV15DimOuterRounds));
        ds4_cuda_fused_v20_finalize_kernel<<<finalize_grid, split_block, 0, stream>>>(
            partial_max.data_ptr<float>(),
            partial_sum.data_ptr<float>(),
            score_state.data_ptr<float>(),
            has_sink ? attn_sink.data_ptr<float>() : nullptr,
            has_sink,
            swa_k_cache.data_ptr<uint8_t>(),
            swa_indices.data_ptr<int32_t>(),
            swa_topk_lengths.data_ptr<int32_t>(),
            static_cast<int>(swa_width),
            static_cast<int>(swa_page_size),
            static_cast<int>(swa_k_cache.size(3)),
            has_extra ? extra_k_cache.data_ptr<uint8_t>() : nullptr,
            has_extra ? extra_indices.data_ptr<int32_t>() : nullptr,
            has_extra ? extra_topk_lengths.data_ptr<int32_t>() : nullptr,
            static_cast<int>(extra_width),
            static_cast<int>(extra_page_size),
            has_extra ? static_cast<int>(extra_k_cache.size(3)) : 0,
            has_extra,
            static_cast<int>(batch_size),
            static_cast<int>(num_heads),
            static_cast<int>(total_width),
            static_cast<int>(head_tiles),
            static_cast<int>(row_tiles),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()));
      } else {
        ds4_cuda_fused_v19_finalize_kernel<<<fused_grid, split_block, 0, stream>>>(
            partial_max.data_ptr<float>(),
            partial_sum.data_ptr<float>(),
            score_state.data_ptr<float>(),
            has_sink ? attn_sink.data_ptr<float>() : nullptr,
            has_sink,
            swa_k_cache.data_ptr<uint8_t>(),
            swa_indices.data_ptr<int32_t>(),
            swa_topk_lengths.data_ptr<int32_t>(),
            static_cast<int>(swa_width),
            static_cast<int>(swa_page_size),
            static_cast<int>(swa_k_cache.size(3)),
            has_extra ? extra_k_cache.data_ptr<uint8_t>() : nullptr,
            has_extra ? extra_indices.data_ptr<int32_t>() : nullptr,
            has_extra ? extra_topk_lengths.data_ptr<int32_t>() : nullptr,
            static_cast<int>(extra_width),
            static_cast<int>(extra_page_size),
            has_extra ? static_cast<int>(extra_k_cache.size(3)) : 0,
            has_extra,
            static_cast<int>(batch_size),
            static_cast<int>(num_heads),
            static_cast<int>(total_width),
            static_cast<int>(head_tiles),
            static_cast<int>(row_tiles),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()));
      }
      if (profile_split) {
        C10_CUDA_CHECK(cudaEventRecord(finalize_stop, stream));
        C10_CUDA_CHECK(cudaEventSynchronize(finalize_stop));
        float score_ms = 0.0f;
        float finalize_ms = 0.0f;
        C10_CUDA_CHECK(cudaEventElapsedTime(&score_ms, score_start, score_stop));
        C10_CUDA_CHECK(cudaEventElapsedTime(&finalize_ms, score_stop, finalize_stop));
        std::fprintf(
            stderr,
            "DSV4_CUDA_SCORE_SPLIT_PROFILE variant=%s batch=%lld heads=%lld "
            "total_width=%lld head_tiles=%lld row_tiles=%lld score_state_dtype=fp32 "
            "score_ms=%.6f finalize_ms=%.6f kernel_ms=%.6f\n",
            attention_variant_name(variant),
            static_cast<long long>(batch_size),
            static_cast<long long>(num_heads),
            static_cast<long long>(total_width),
            static_cast<long long>(head_tiles),
            static_cast<long long>(row_tiles),
            score_ms,
            finalize_ms,
            score_ms + finalize_ms);
        C10_CUDA_CHECK(cudaEventDestroy(score_start));
        C10_CUDA_CHECK(cudaEventDestroy(score_stop));
        C10_CUDA_CHECK(cudaEventDestroy(finalize_stop));
      }
      break;
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor launch_ds4_cuda_scores(
    torch::Tensor q,
    torch::Tensor swa_k_cache,
    torch::Tensor swa_indices,
    torch::Tensor swa_topk_lengths,
    int64_t swa_page_size,
    double softmax_scale,
    torch::Tensor extra_k_cache,
    torch::Tensor extra_indices,
    torch::Tensor extra_topk_lengths,
    int64_t extra_page_size,
    bool tensor_core) {
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
  extra_k_cache = extra_k_cache.contiguous();
  extra_indices = extra_indices.contiguous();
  extra_topk_lengths = extra_topk_lengths.contiguous();

  const auto batch_size = flattened_batch(q);
  const auto num_heads = q.size(-2);
  const auto swa_width = swa_indices.size(-1);
  TORCH_CHECK(swa_topk_lengths.numel() == batch_size, "swa lengths must match flattened q batch");
  TORCH_CHECK(swa_indices.numel() / swa_width == batch_size, "swa indices must match flattened q batch");

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

  const int64_t total_width = swa_width + extra_width;
  auto out = torch::empty(
      {batch_size, num_heads, total_width},
      q.options().dtype(torch::kFloat32));

  c10::cuda::CUDAGuard device_guard(q.device());
  if (tensor_core) {
    dim3 grid(
        static_cast<unsigned int>(batch_size),
        static_cast<unsigned int>((num_heads + kScoreTileM - 1) / kScoreTileM),
        static_cast<unsigned int>((total_width + kScoreRowsPerBlock - 1) / kScoreRowsPerBlock));
    dim3 block(kScoreThreads);
    ds4_cuda_scores_v6_mma_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
        swa_k_cache.data_ptr<uint8_t>(),
        swa_indices.data_ptr<int32_t>(),
        swa_topk_lengths.data_ptr<int32_t>(),
        static_cast<int>(swa_width),
        static_cast<int>(swa_page_size),
        static_cast<int>(swa_k_cache.size(3)),
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
        static_cast<int>(total_width),
        out.data_ptr<float>());
  } else {
    dim3 grid(
        static_cast<unsigned int>(batch_size),
        static_cast<unsigned int>(num_heads),
        static_cast<unsigned int>(total_width));
    dim3 block(kThreads);
    ds4_cuda_scores_reference_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
        swa_k_cache.data_ptr<uint8_t>(),
        swa_indices.data_ptr<int32_t>(),
        swa_topk_lengths.data_ptr<int32_t>(),
        static_cast<int>(swa_width),
        static_cast<int>(swa_page_size),
        static_cast<int>(swa_k_cache.size(3)),
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
        static_cast<int>(total_width),
        out.data_ptr<float>());
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

torch::Tensor ds4_cuda_optimized_v7_attention(
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
      AttentionVariant::kOptimizedV7);
}

torch::Tensor ds4_cuda_optimized_v8_attention(
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
      AttentionVariant::kOptimizedV8);
}

torch::Tensor ds4_cuda_optimized_v9_attention(
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
      AttentionVariant::kOptimizedV9);
}

torch::Tensor ds4_cuda_optimized_v10_attention(
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
      AttentionVariant::kOptimizedV10);
}

torch::Tensor ds4_cuda_optimized_v11_attention(
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
      AttentionVariant::kOptimizedV11);
}

torch::Tensor ds4_cuda_optimized_v12_attention(
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
      AttentionVariant::kOptimizedV12);
}

torch::Tensor ds4_cuda_optimized_v13_attention(
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
      AttentionVariant::kOptimizedV13);
}

torch::Tensor ds4_cuda_optimized_v14_attention(
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
      AttentionVariant::kOptimizedV14);
}

torch::Tensor ds4_cuda_optimized_v15_attention(
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
      AttentionVariant::kOptimizedV15);
}

torch::Tensor ds4_cuda_optimized_v16_attention(
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
      AttentionVariant::kOptimizedV16);
}

torch::Tensor ds4_cuda_optimized_v17_attention(
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
      AttentionVariant::kOptimizedV17);
}

torch::Tensor ds4_cuda_optimized_v18_attention(
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
      AttentionVariant::kOptimizedV18);
}

torch::Tensor ds4_cuda_optimized_v19_attention(
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
      AttentionVariant::kOptimizedV19);
}

torch::Tensor ds4_cuda_optimized_v20_attention(
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
      AttentionVariant::kOptimizedV20);
}

torch::Tensor ds4_cuda_reference_scores(
    torch::Tensor q,
    torch::Tensor swa_k_cache,
    torch::Tensor swa_indices,
    torch::Tensor swa_topk_lengths,
    int64_t swa_page_size,
    double softmax_scale,
    torch::Tensor extra_k_cache,
    torch::Tensor extra_indices,
    torch::Tensor extra_topk_lengths,
    int64_t extra_page_size) {
  return launch_ds4_cuda_scores(
      q,
      swa_k_cache,
      swa_indices,
      swa_topk_lengths,
      swa_page_size,
      softmax_scale,
      extra_k_cache,
      extra_indices,
      extra_topk_lengths,
      extra_page_size,
      false);
}

torch::Tensor ds4_cuda_v6_mma_scores(
    torch::Tensor q,
    torch::Tensor swa_k_cache,
    torch::Tensor swa_indices,
    torch::Tensor swa_topk_lengths,
    int64_t swa_page_size,
    double softmax_scale,
    torch::Tensor extra_k_cache,
    torch::Tensor extra_indices,
    torch::Tensor extra_topk_lengths,
    int64_t extra_page_size) {
  return launch_ds4_cuda_scores(
      q,
      swa_k_cache,
      swa_indices,
      swa_topk_lengths,
      swa_page_size,
      softmax_scale,
      extra_k_cache,
      extra_indices,
      extra_topk_lengths,
      extra_page_size,
      true);
}
