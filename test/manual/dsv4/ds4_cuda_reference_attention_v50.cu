#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr int kNopeDim = 448;
constexpr int kRopeDim = 64;
constexpr int kHeadDim = kNopeDim + kRopeDim;
constexpr int kScoreTileM = 16;
constexpr int kScoreTileN = 16;
constexpr int kScoreTileK = 16;
constexpr int kScoreTileElems = kScoreTileM * kScoreTileN;
constexpr int kScoreKTileCount = kHeadDim / kScoreTileK;
constexpr int kScoreWarpsPerBlock = 4;
constexpr int kScoreRowsPerBlock = kScoreTileN * kScoreWarpsPerBlock;
constexpr int kPreparedKTileElems = kScoreTileK * kScoreRowsPerBlock;
constexpr int kPreparedVTileElems = kScoreRowsPerBlock * kScoreTileK;
constexpr int kV7Threads = 512;
constexpr int kV8Threads = kV7Threads;
constexpr int kV11DimGroups = 4;
constexpr int kV11WarpsPerBlock = kScoreWarpsPerBlock * kV11DimGroups;
constexpr int kV11DimRounds = kHeadDim / (kScoreTileN * kV11DimGroups);
constexpr int kV15DimRoundGroups = 4;
constexpr int kV15DimOuterRounds = kV11DimRounds / kV15DimRoundGroups;
static_assert(
    kHeadDim % kScoreTileK == 0,
    "DS4 v50 expects the head dimension to divide into score-K tiles");
static_assert(
    kV11DimRounds % kV15DimRoundGroups == 0,
    "DS4 v50 expects P@V dim rounds to divide evenly into outer rounds");

constexpr int kPartialProfileRowSetup = 0;
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
constexpr int kPartialProfileQkStagingNopeWall = 14;
constexpr int kPartialProfileQkStagingRopeWall = 15;

__device__ __forceinline__ void store_partial_acc(
    __nv_bfloat16* partial_acc,
    int64_t offset,
    float value) {
  partial_acc[offset] = __float2bfloat16(value);
}

__device__ __forceinline__ void add_partial_profile_cycles(
    unsigned long long* profile_cycles,
    int slot,
    unsigned long long cycles) {
  if (profile_cycles != nullptr && threadIdx.x == 0) {
    atomicAdd(profile_cycles + slot, cycles);
  }
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

// v50 is the v49 hot path without the old PVMmaMode dispatch ladder.
//
// Prepared-KV producer contract:
//   prepared_k[batch, row_tile, dim_tile, dim_slot, row_slot]
//   prepared_v[batch, row_tile, dim_tile, row_slot, dim_slot]
//
// QK consumes prepared_k directly as WMMA matrix_b row-major with leading
// dimension 64. P@V consumes prepared_v directly as WMMA matrix_b row-major
// with leading dimension 16. The only cache-local state here is Q reuse,
// softmax probabilities, and the BF16 partial accumulator consumed by the
// existing DS4 reduction kernel.
template <bool ProfileStages = false>
__global__ void __launch_bounds__(kV8Threads, 1)
ds4_cuda_fused_v50_direct_prepared_wmma_partial_kernel(
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
    __nv_bfloat16* __restrict__ partial_acc,
    unsigned long long* __restrict__ profile_cycles,
    const __nv_bfloat16* __restrict__ prepared_k,
    const __nv_bfloat16* __restrict__ prepared_v) {
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

  constexpr bool profile_this_block = ProfileStages;
  unsigned long long profile_t0 = 0;
  if (profile_this_block && threadIdx.x == 0) {
    profile_t0 = clock64();
    atomicAdd(profile_cycles + kPartialProfileBlocks, 1ULL);
  }

  __shared__ __align__(16) __nv_bfloat16
      q_reuse_shared[kScoreKTileCount][kScoreTileElems];
  __shared__ __align__(16) float score_shared[kV11WarpsPerBlock][kScoreTileElems];
  __shared__ __align__(16) __nv_bfloat16 p_shared[kScoreWarpsPerBlock][kScoreTileElems];
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
    if (row < total_width) {
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
    }
    row_valid[threadIdx.x] = valid ? 1 : 0;
  }
  __syncthreads();
  if (profile_this_block && threadIdx.x == 0) {
    const unsigned long long now = clock64();
    add_partial_profile_cycles(
        profile_cycles, kPartialProfileRowSetup, now - profile_t0);
    profile_t0 = now;
  }

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

  unsigned long long qk_barrier_t0 = profile_t0;
  if (warp < kScoreWarpsPerBlock) {
    wmma::fragment<
        wmma::matrix_a,
        kScoreTileM,
        kScoreTileN,
        kScoreTileK,
        __nv_bfloat16,
        wmma::row_major>
        q_frag;
    wmma::fragment<
        wmma::matrix_b,
        kScoreTileM,
        kScoreTileN,
        kScoreTileK,
        __nv_bfloat16,
        wmma::row_major>
        k_frag;
    wmma::fragment<wmma::accumulator, kScoreTileM, kScoreTileN, kScoreTileK, float>
        acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    const int warp_row_base = warp * kScoreTileN;
    for (int dim_base = 0; dim_base < kHeadDim; dim_base += kScoreTileK) {
      unsigned long long qk_detail_t0 = 0;
      if (profile_this_block && threadIdx.x == 0) {
        qk_detail_t0 = clock64();
      }
      __syncwarp();
      if (profile_this_block && threadIdx.x == 0) {
        const unsigned long long now = clock64();
        const unsigned long long staging_cycles = now - qk_detail_t0;
        add_partial_profile_cycles(
            profile_cycles, kPartialProfileQkStagingWall, staging_cycles);
        add_partial_profile_cycles(
            profile_cycles,
            dim_base < kNopeDim ? kPartialProfileQkStagingNopeWall
                                : kPartialProfileQkStagingRopeWall,
            staging_cycles);
        qk_detail_t0 = now;
      }

      wmma::load_matrix_sync(
          q_frag, q_reuse_shared[dim_base / kScoreTileK], kScoreTileK);
      const int64_t prepared_base =
          (((static_cast<int64_t>(batch) * row_tiles + row_tile) *
                kScoreKTileCount +
            (dim_base / kScoreTileK)) *
           kPreparedKTileElems);
      wmma::load_matrix_sync(
          k_frag, prepared_k + prepared_base + warp_row_base, kScoreRowsPerBlock);
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
      p_shared[row_warp][head_slot * kScoreTileN + row_lane] =
          __float2bfloat16(weight);
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

  __syncthreads();
  if (profile_this_block && threadIdx.x == 0) {
    const unsigned long long now = clock64();
    add_partial_profile_cycles(
        profile_cycles, kPartialProfilePCache, now - profile_t0);
    profile_t0 = now;
  }

  wmma::fragment<
      wmma::matrix_a,
      kScoreTileM,
      kScoreTileN,
      kScoreTileK,
      __nv_bfloat16,
      wmma::row_major>
      p_frag;
  wmma::fragment<
      wmma::matrix_b,
      kScoreTileM,
      kScoreTileN,
      kScoreTileK,
      __nv_bfloat16,
      wmma::row_major>
      v_frag;
  wmma::fragment<wmma::accumulator, kScoreTileM, kScoreTileN, kScoreTileK, float>
      pv_acc_frag;

  for (int outer_round = 0; outer_round < kV15DimOuterRounds; ++outer_round) {
    if (warp < kV11WarpsPerBlock) {
      const int dim_round_group = warp / kV11DimGroups;
      const int dim_group = warp - dim_round_group * kV11DimGroups;
      const int dim_round = outer_round * kV15DimRoundGroups + dim_round_group;
      const int dim_base = (dim_round * kV11DimGroups + dim_group) * kScoreTileN;

      wmma::fill_fragment(pv_acc_frag, 0.0f);
#pragma unroll
      for (int row_group = 0; row_group < kScoreWarpsPerBlock; ++row_group) {
        wmma::load_matrix_sync(p_frag, p_shared[row_group], kScoreTileN);
        const int64_t prepared_base =
            (((static_cast<int64_t>(batch) * row_tiles + row_tile) *
                  kScoreKTileCount +
              (dim_base / kScoreTileK)) *
             kPreparedVTileElems);
        wmma::load_matrix_sync(
            v_frag,
            prepared_v + prepared_base + row_group * kScoreTileN * kScoreTileK,
            kScoreTileK);
        wmma::mma_sync(pv_acc_frag, p_frag, v_frag, pv_acc_frag);
      }
      wmma::store_matrix_sync(
          score_shared[warp], pv_acc_frag, kScoreTileN, wmma::mem_row_major);
      __syncwarp();

      for (int tile_idx = lane; tile_idx < kScoreTileElems; tile_idx += 32) {
        const int head_slot = tile_idx / kScoreTileN;
        const int dim_slot = tile_idx - head_slot * kScoreTileN;
        const int head = head_base + head_slot;
        const int dim = dim_base + dim_slot;
        const int64_t acc_offset =
            ((((static_cast<int64_t>(batch) * head_tiles + head_tile) * row_tiles + row_tile) *
                  kScoreTileM +
              head_slot) *
                 kHeadDim +
             dim);
        store_partial_acc(
            partial_acc,
            acc_offset,
            head < num_heads ? score_shared[warp][tile_idx] : 0.0f);
      }
    }
    __syncthreads();
    if (profile_this_block && threadIdx.x == 0) {
      const unsigned long long now = clock64();
      add_partial_profile_cycles(
          profile_cycles, kPartialProfilePvMma, now - profile_t0);
      profile_t0 = now;
    }

    __syncthreads();
    if (profile_this_block && threadIdx.x == 0) {
      const unsigned long long now = clock64();
      add_partial_profile_cycles(
          profile_cycles, kPartialProfilePartialStore, now - profile_t0);
      profile_t0 = now;
    }
  }
}

}  // namespace

void ds4_cuda_launch_v50_direct_prepared_wmma_partial(
    dim3 grid,
    dim3 block,
    cudaStream_t stream,
    bool profile_stages,
    const __nv_bfloat16* q,
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
    float softmax_scale,
    int batch_size,
    int num_heads,
    int total_width,
    int head_tiles,
    int row_tiles,
    float* partial_max,
    float* partial_sum,
    __nv_bfloat16* partial_acc,
    unsigned long long* profile_cycles,
    const __nv_bfloat16* prepared_k,
    const __nv_bfloat16* prepared_v) {
  if (profile_stages) {
    ds4_cuda_fused_v50_direct_prepared_wmma_partial_kernel<true>
        <<<grid, block, 0, stream>>>(
            q,
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
            softmax_scale,
            batch_size,
            num_heads,
            total_width,
            head_tiles,
            row_tiles,
            partial_max,
            partial_sum,
            partial_acc,
            profile_cycles,
            prepared_k,
            prepared_v);
  } else {
    ds4_cuda_fused_v50_direct_prepared_wmma_partial_kernel<false>
        <<<grid, block, 0, stream>>>(
            q,
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
            softmax_scale,
            batch_size,
            num_heads,
            total_width,
            head_tiles,
            row_tiles,
            partial_max,
            partial_sum,
            partial_acc,
            nullptr,
            prepared_k,
            prepared_v);
  }
}
