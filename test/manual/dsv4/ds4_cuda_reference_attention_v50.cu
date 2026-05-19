#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstdint>
#include <type_traits>

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
constexpr int kPreparedKRowGroupTileElems = kScoreWarpsPerBlock * kScoreTileElems;
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
static_assert(
    kPreparedKRowGroupTileElems == kPreparedKTileElems,
    "DS4 v52 row-group K layout preserves the existing prepared-K allocation size");

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

__device__ __forceinline__ float warp_reduce_sum(float value) {
  value += __shfl_down_sync(0xffffffff, value, 16);
  value += __shfl_down_sync(0xffffffff, value, 8);
  value += __shfl_down_sync(0xffffffff, value, 4);
  value += __shfl_down_sync(0xffffffff, value, 2);
  value += __shfl_down_sync(0xffffffff, value, 1);
  return value;
}

__device__ __forceinline__ float warp_reduce_max(float value) {
  value = fmaxf(value, __shfl_down_sync(0xffffffff, value, 16));
  value = fmaxf(value, __shfl_down_sync(0xffffffff, value, 8));
  value = fmaxf(value, __shfl_down_sync(0xffffffff, value, 4));
  value = fmaxf(value, __shfl_down_sync(0xffffffff, value, 2));
  value = fmaxf(value, __shfl_down_sync(0xffffffff, value, 1));
  return value;
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
// v58-v61 optionally flip these prepared/shared fragment contracts:
//   K as matrix_b col-major: [row_group, row_lane, dim_slot]
//   V as matrix_b col-major: [row_group, dim_slot, row_lane]
//   Q/P as matrix_a col-major in shared: [dim/row, head]
//   v61 stages each V row-group tile into shared before P@V WMMA.
//
// QK consumes prepared_k directly as WMMA matrix_b row-major, or stages one
// row-group K tile into shared memory for the v54 layout experiment. P@V
// consumes prepared_v directly as WMMA matrix_b row-major with leading
// dimension 16. The only cache-local state here is Q reuse, optional staged K,
// softmax probabilities, and the BF16 partial accumulator consumed by the
// existing DS4 reduction kernel.
template <
    bool ProfileStages = false,
    bool WarpSoftmax = false,
    bool RowGroupPreparedK = false,
    bool StagedPreparedK = false,
    bool DimSplitQK = false,
    bool KColMajorB = false,
    bool VColMajorB = false,
    bool AColMajor = false,
    bool StagedPreparedV = false>
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
  using MatrixALayout =
      std::conditional_t<AColMajor, wmma::col_major, wmma::row_major>;
  using KMatrixBLayout =
      std::conditional_t<KColMajorB, wmma::col_major, wmma::row_major>;
  using VMatrixBLayout =
      std::conditional_t<VColMajorB, wmma::col_major, wmma::row_major>;
  static_assert(
      !StagedPreparedK || !KColMajorB,
      "staged prepared-K expects the row-major B prepared-K contract");
  static_assert(
      !StagedPreparedV || VColMajorB,
      "staged prepared-V expects the V matrix-B col-major contract");

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
  __shared__ __align__(16) __nv_bfloat16
      k_stage_shared[StagedPreparedK ? kPreparedKTileElems : 1];
  __shared__ __align__(16) __nv_bfloat16
      v_stage_shared[StagedPreparedV ? kV11WarpsPerBlock * kScoreTileElems : 1];
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
    const int q_store_idx =
        AColMajor ? q_dim_slot * kScoreTileM + q_head_slot : tile_idx;
    q_reuse_shared[dim_tile][q_store_idx] =
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
  if constexpr (DimSplitQK) {
    static_assert(
        WarpSoftmax && RowGroupPreparedK && !StagedPreparedK,
        "DS4 v55 dim-split QK expects the v53 warp-softmax row-group layout");
    static_assert(
        kScoreKTileCount % kV11DimGroups == 0,
        "DS4 v55 expects the head dimension K tiles to split across dim groups");

    if (warp < kV11WarpsPerBlock) {
      wmma::fragment<
          wmma::matrix_a,
          kScoreTileM,
          kScoreTileN,
          kScoreTileK,
          __nv_bfloat16,
          MatrixALayout>
          q_frag;
      wmma::fragment<
          wmma::matrix_b,
          kScoreTileM,
          kScoreTileN,
          kScoreTileK,
          __nv_bfloat16,
          KMatrixBLayout>
          k_frag;
      wmma::fragment<wmma::accumulator, kScoreTileM, kScoreTileN, kScoreTileK, float>
          acc_frag;
      wmma::fill_fragment(acc_frag, 0.0f);

      const int row_group = warp / kV11DimGroups;
      const int dim_group = warp - row_group * kV11DimGroups;
#pragma unroll
      for (int dim_iter = 0; dim_iter < kScoreKTileCount / kV11DimGroups;
           ++dim_iter) {
        const int dim_tile = dim_iter * kV11DimGroups + dim_group;
        const int dim_base = dim_tile * kScoreTileK;
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

        wmma::load_matrix_sync(q_frag, q_reuse_shared[dim_tile], kScoreTileK);
        const int64_t prepared_base =
            (((static_cast<int64_t>(batch) * row_tiles + row_tile) *
                  kScoreKTileCount +
              dim_tile) *
             kPreparedKTileElems);
        wmma::load_matrix_sync(
            k_frag,
          prepared_k + prepared_base + row_group * kScoreTileElems,
          kScoreTileK);
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
      wmma::store_matrix_sync(
          score_shared[warp], acc_frag, kScoreTileN, wmma::mem_row_major);
      if (profile_this_block && threadIdx.x == 0) {
        const unsigned long long now = clock64();
        add_partial_profile_cycles(
            profile_cycles, kPartialProfileQkScoreStoreWall, now - qk_store_t0);
        qk_barrier_t0 = now;
      }
    }
    __syncthreads();

    if (warp < kScoreWarpsPerBlock) {
      const int row_group = warp;
      const int base_slot = row_group * kV11DimGroups;
      unsigned long long qk_reduce_t0 = 0;
      if (profile_this_block && threadIdx.x == 0) {
        qk_reduce_t0 = clock64();
      }
      for (int tile_idx = lane; tile_idx < kScoreTileElems; tile_idx += 32) {
        const float score =
            score_shared[base_slot][tile_idx] +
            score_shared[base_slot + 1][tile_idx] +
            score_shared[base_slot + 2][tile_idx] +
            score_shared[base_slot + 3][tile_idx];
        score_shared[base_slot][tile_idx] = score;
      }
      if (profile_this_block && threadIdx.x == 0) {
        const unsigned long long now = clock64();
        add_partial_profile_cycles(
            profile_cycles, kPartialProfileQkScoreStoreWall, now - qk_reduce_t0);
        qk_barrier_t0 = now;
      }
    }
  } else if (warp < kScoreWarpsPerBlock) {
    wmma::fragment<
        wmma::matrix_a,
        kScoreTileM,
        kScoreTileN,
        kScoreTileK,
        __nv_bfloat16,
        MatrixALayout>
        q_frag;
    wmma::fragment<
        wmma::matrix_b,
        kScoreTileM,
        kScoreTileN,
        kScoreTileK,
        __nv_bfloat16,
        KMatrixBLayout>
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
      if constexpr (StagedPreparedK) {
        static_assert(
            RowGroupPreparedK,
            "DS4 v54 staged-K path expects row-group prepared-K layout");
        constexpr int kRowGroupVecCount =
            (kScoreTileElems * static_cast<int>(sizeof(__nv_bfloat16))) /
            static_cast<int>(sizeof(uint4));
        static_assert(
            kScoreTileElems * static_cast<int>(sizeof(__nv_bfloat16)) ==
                kRowGroupVecCount * static_cast<int>(sizeof(uint4)),
            "DS4 v54 staged-K row-group tile should divide into uint4 copies");

        auto* staged_vec =
            reinterpret_cast<uint4*>(k_stage_shared + warp * kScoreTileElems);
        const auto* prepared_vec = reinterpret_cast<const uint4*>(
            prepared_k + prepared_base + warp * kScoreTileElems);
        if (lane < kRowGroupVecCount) {
          staged_vec[lane] = prepared_vec[lane];
        }
        __syncwarp();
        wmma::load_matrix_sync(
            k_frag, k_stage_shared + warp * kScoreTileElems, kScoreTileK);
      } else if constexpr (RowGroupPreparedK) {
        wmma::load_matrix_sync(
            k_frag, prepared_k + prepared_base + warp * kScoreTileElems, kScoreTileK);
      } else {
        wmma::load_matrix_sync(
            k_frag, prepared_k + prepared_base + warp_row_base, kScoreRowsPerBlock);
      }
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

  if constexpr (WarpSoftmax) {
    if (warp < kScoreTileM) {
      const int head_slot = warp;
      const int head = head_base + head_slot;
      float local_max = -INFINITY;
#pragma unroll
      for (int row_slot = lane; row_slot < kScoreRowsPerBlock; row_slot += 32) {
        if (head < num_heads && row_valid[row_slot]) {
          const int row_warp = row_slot / kScoreTileN;
          const int row_lane = row_slot - row_warp * kScoreTileN;
          const int score_warp = DimSplitQK ? row_warp * kV11DimGroups : row_warp;
          const float score =
              score_shared[score_warp][head_slot * kScoreTileN + row_lane] *
              softmax_scale;
          local_max = fmaxf(local_max, score);
        }
      }
      float tile_max = warp_reduce_max(local_max);
      tile_max = __shfl_sync(0xffffffff, tile_max, 0);

      float local_sum = 0.0f;
#pragma unroll
      for (int row_slot = lane; row_slot < kScoreRowsPerBlock; row_slot += 32) {
        float weight = 0.0f;
        if (head < num_heads && row_valid[row_slot]) {
          const int row_warp = row_slot / kScoreTileN;
          const int row_lane = row_slot - row_warp * kScoreTileN;
          const int score_warp = DimSplitQK ? row_warp * kV11DimGroups : row_warp;
          const float score =
              score_shared[score_warp][head_slot * kScoreTileN + row_lane] *
              softmax_scale;
          weight = expf(score - tile_max);
          local_sum += weight;
        }
        const int row_warp = row_slot / kScoreTileN;
        const int row_lane = row_slot - row_warp * kScoreTileN;
        const int p_idx =
            AColMajor ? row_lane * kScoreTileM + head_slot
                      : head_slot * kScoreTileN + row_lane;
        p_shared[row_warp][p_idx] =
            __float2bfloat16(weight);
      }
      float tile_sum = warp_reduce_sum(local_sum);
      tile_sum = __shfl_sync(0xffffffff, tile_sum, 0);

      if (lane == 0) {
        tile_max_shared[head_slot] = tile_max;
        tile_sum_shared[head_slot] = tile_sum;
        const int64_t state_offset =
            (((static_cast<int64_t>(batch) * head_tiles + head_tile) * row_tiles +
              row_tile) *
                 kScoreTileM +
             head_slot);
        partial_max[state_offset] = tile_max;
        partial_sum[state_offset] = tile_sum;
      }
    }
  } else if (threadIdx.x < kScoreTileM) {
    const int head_slot = threadIdx.x;
    const int head = head_base + head_slot;
    float tile_max = -INFINITY;
#pragma unroll
    for (int row_slot = 0; row_slot < kScoreRowsPerBlock; ++row_slot) {
      if (head < num_heads && row_valid[row_slot]) {
        const int row_warp = row_slot / kScoreTileN;
        const int row_lane = row_slot - row_warp * kScoreTileN;
        const int score_warp = DimSplitQK ? row_warp * kV11DimGroups : row_warp;
        const float score =
            score_shared[score_warp][head_slot * kScoreTileN + row_lane] * softmax_scale;
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
        const int score_warp = DimSplitQK ? row_warp * kV11DimGroups : row_warp;
        const float score =
            score_shared[score_warp][head_slot * kScoreTileN + row_lane] * softmax_scale;
        weight = expf(score - tile_max);
        tile_sum += weight;
      }
      const int row_warp = row_slot / kScoreTileN;
      const int row_lane = row_slot - row_warp * kScoreTileN;
      const int p_idx =
          AColMajor ? row_lane * kScoreTileM + head_slot
                    : head_slot * kScoreTileN + row_lane;
      p_shared[row_warp][p_idx] =
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
      MatrixALayout>
      p_frag;
  wmma::fragment<
      wmma::matrix_b,
      kScoreTileM,
      kScoreTileN,
      kScoreTileK,
      __nv_bfloat16,
      VMatrixBLayout>
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
        if constexpr (StagedPreparedV) {
          constexpr int kRowGroupVecCount =
              (kScoreTileElems * static_cast<int>(sizeof(__nv_bfloat16))) /
              static_cast<int>(sizeof(uint4));
          static_assert(
              kScoreTileElems * static_cast<int>(sizeof(__nv_bfloat16)) ==
                  kRowGroupVecCount * static_cast<int>(sizeof(uint4)),
              "DS4 v61 staged-V tile should divide into uint4 copies");

          auto* staged_vec = reinterpret_cast<uint4*>(
              v_stage_shared + warp * kScoreTileElems);
          const auto* prepared_vec = reinterpret_cast<const uint4*>(
              prepared_v + prepared_base + row_group * kScoreTileElems);
          if (lane < kRowGroupVecCount) {
            staged_vec[lane] = prepared_vec[lane];
          }
          __syncwarp();
          wmma::load_matrix_sync(
              v_frag, v_stage_shared + warp * kScoreTileElems, kScoreTileK);
        } else {
          wmma::load_matrix_sync(
              v_frag,
              prepared_v + prepared_base + row_group * kScoreTileN * kScoreTileK,
              kScoreTileK);
        }
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

template <
    bool WarpSoftmax,
    bool RowGroupPreparedK,
    bool StagedPreparedK,
    bool DimSplitQK,
    bool KColMajorB,
    bool VColMajorB,
    bool AColMajor,
    bool StagedPreparedV = false>
void ds4_cuda_launch_direct_prepared_wmma_partial_variant(
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
    ds4_cuda_fused_v50_direct_prepared_wmma_partial_kernel<
        true,
        WarpSoftmax,
        RowGroupPreparedK,
        StagedPreparedK,
        DimSplitQK,
        KColMajorB,
        VColMajorB,
        AColMajor,
        StagedPreparedV>
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
    ds4_cuda_fused_v50_direct_prepared_wmma_partial_kernel<
        false,
        WarpSoftmax,
        RowGroupPreparedK,
        StagedPreparedK,
        DimSplitQK,
        KColMajorB,
        VColMajorB,
        AColMajor,
        StagedPreparedV>
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

#define DSV4_DEFINE_DIRECT_PREPARED_LAUNCH(                              \
    NAME, WARP_SOFTMAX, ROWGROUP_K, STAGED_K, DIM_SPLIT_QK, K_COL_B,     \
    V_COL_B, A_COL, STAGED_V)                                             \
  void NAME(                                                              \
      dim3 grid,                                                          \
      dim3 block,                                                         \
      cudaStream_t stream,                                                \
      bool profile_stages,                                                \
      const __nv_bfloat16* q,                                             \
      const uint8_t* swa_cache,                                           \
      const int32_t* swa_indices,                                         \
      const int32_t* swa_lengths,                                         \
      int swa_width,                                                      \
      int swa_page_size,                                                  \
      int swa_row_stride,                                                 \
      const uint8_t* extra_cache,                                         \
      const int32_t* extra_indices,                                       \
      const int32_t* extra_lengths,                                       \
      int extra_width,                                                    \
      int extra_page_size,                                                \
      int extra_row_stride,                                               \
      bool has_extra,                                                     \
      float softmax_scale,                                                \
      int batch_size,                                                     \
      int num_heads,                                                      \
      int total_width,                                                    \
      int head_tiles,                                                     \
      int row_tiles,                                                      \
      float* partial_max,                                                 \
      float* partial_sum,                                                 \
      __nv_bfloat16* partial_acc,                                         \
      unsigned long long* profile_cycles,                                 \
      const __nv_bfloat16* prepared_k,                                    \
      const __nv_bfloat16* prepared_v) {                                  \
    ds4_cuda_launch_direct_prepared_wmma_partial_variant<                 \
        WARP_SOFTMAX,                                                     \
        ROWGROUP_K,                                                        \
        STAGED_K,                                                          \
        DIM_SPLIT_QK,                                                      \
        K_COL_B,                                                           \
        V_COL_B,                                                           \
        A_COL,                                                             \
        STAGED_V>(                                                         \
        grid,                                                             \
        block,                                                            \
        stream,                                                           \
        profile_stages,                                                   \
        q,                                                                \
        swa_cache,                                                        \
        swa_indices,                                                      \
        swa_lengths,                                                      \
        swa_width,                                                        \
        swa_page_size,                                                    \
        swa_row_stride,                                                   \
        extra_cache,                                                      \
        extra_indices,                                                    \
        extra_lengths,                                                    \
        extra_width,                                                      \
        extra_page_size,                                                  \
        extra_row_stride,                                                 \
        has_extra,                                                        \
        softmax_scale,                                                    \
        batch_size,                                                       \
        num_heads,                                                        \
        total_width,                                                      \
        head_tiles,                                                       \
        row_tiles,                                                        \
        partial_max,                                                      \
        partial_sum,                                                      \
        partial_acc,                                                      \
        profile_cycles,                                                   \
        prepared_k,                                                       \
        prepared_v);                                                      \
  }

DSV4_DEFINE_DIRECT_PREPARED_LAUNCH(
    ds4_cuda_launch_v50_direct_prepared_wmma_partial,
    false,
    false,
    false,
    false,
    false,
    false,
    false,
    false)
DSV4_DEFINE_DIRECT_PREPARED_LAUNCH(
    ds4_cuda_launch_v51_warp_softmax_direct_prepared_wmma_partial,
    true,
    false,
    false,
    false,
    false,
    false,
    false,
    false)
DSV4_DEFINE_DIRECT_PREPARED_LAUNCH(
    ds4_cuda_launch_v52_rowgroup_k_direct_prepared_wmma_partial,
    false,
    true,
    false,
    false,
    false,
    false,
    false,
    false)
DSV4_DEFINE_DIRECT_PREPARED_LAUNCH(
    ds4_cuda_launch_v53_warp_softmax_rowgroup_k_direct_prepared_wmma_partial,
    true,
    true,
    false,
    false,
    false,
    false,
    false,
    false)
DSV4_DEFINE_DIRECT_PREPARED_LAUNCH(
    ds4_cuda_launch_v54_staged_k_warp_softmax_rowgroup_direct_prepared_wmma_partial,
    true,
    true,
    true,
    false,
    false,
    false,
    false,
    false)
DSV4_DEFINE_DIRECT_PREPARED_LAUNCH(
    ds4_cuda_launch_v55_dimsplit_qk_warp_softmax_rowgroup_direct_prepared_wmma_partial,
    true,
    true,
    false,
    true,
    false,
    false,
    false,
    false)
DSV4_DEFINE_DIRECT_PREPARED_LAUNCH(
    ds4_cuda_launch_v58a_k_colmajor_b_direct_prepared_wmma_partial,
    true,
    true,
    false,
    false,
    true,
    false,
    false,
    false)
DSV4_DEFINE_DIRECT_PREPARED_LAUNCH(
    ds4_cuda_launch_v58b_v_colmajor_b_direct_prepared_wmma_partial,
    true,
    true,
    false,
    false,
    false,
    true,
    false,
    false)
DSV4_DEFINE_DIRECT_PREPARED_LAUNCH(
    ds4_cuda_launch_v58_kv_colmajor_b_direct_prepared_wmma_partial,
    true,
    true,
    false,
    false,
    true,
    true,
    false,
    false)
DSV4_DEFINE_DIRECT_PREPARED_LAUNCH(
    ds4_cuda_launch_v59_a_colmajor_direct_prepared_wmma_partial,
    true,
    true,
    false,
    false,
    false,
    false,
    true,
    false)
DSV4_DEFINE_DIRECT_PREPARED_LAUNCH(
    ds4_cuda_launch_v60_ab_colmajor_direct_prepared_wmma_partial,
    true,
    true,
    false,
    false,
    true,
    true,
    true,
    false)
DSV4_DEFINE_DIRECT_PREPARED_LAUNCH(
    ds4_cuda_launch_v61_staged_v_colmajor_b_direct_prepared_wmma_partial,
    true,
    true,
    false,
    false,
    false,
    true,
    false,
    true)

#undef DSV4_DEFINE_DIRECT_PREPARED_LAUNCH
