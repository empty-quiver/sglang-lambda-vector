// Adatped from
// https://github.com/vllm-project/vllm/blob/755ed7b05be4743237d3339c4ff8c22bcaae04f4/csrc/quantization/gguf/gguf_kernel.cu
#include <c10/cuda/CUDAGuard.h>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/all.h>

static inline void gguf_check_cuda_launch(const char* label, cudaStream_t stream) {
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, label, " launch failed: ", cudaGetErrorString(err));
  if (std::getenv("SGLANG_GGUF_SYNC_KERNEL") || std::getenv("SGLANG_GGUF_SYNC_MOE")) {
    err = cudaStreamSynchronize(stream);
    TORCH_CHECK(err == cudaSuccess, label, " failed: ", cudaGetErrorString(err));
  }
}

// dont use clang-format here, it breaks the include order
// clang-format off
#include "utils.h"

#include "ggml-common.h"
#include "vecdotq.cuh"
#include "dequantize.cuh"
#include "mmvq.cuh"
#include "mmq.cuh"
#include "moe.cuh"
#include "moe_vec.cuh"
// clang-format off

// Q8 gemv
template <typename scalar_t>
static __global__ void
quantize_q8_1(const scalar_t* __restrict__ x, void* __restrict__ vy, const int kx, const int kx_padded) {
  const auto ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix >= kx_padded) {
    return;
  }
  const auto iy = blockDim.y * blockIdx.y + threadIdx.y;
  const int i_padded = iy * kx_padded + ix;

  block_q8_1* y = (block_q8_1*)vy;

  const int ib = i_padded / QK8_1;   // block index
  const int iqs = i_padded % QK8_1;  // quant index

  const float xi = ix < kx ? static_cast<float>(x[iy * kx + ix]) : 0.0f;
  float amax = fabsf(xi);
  float sum = xi;

#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    amax = fmaxf(amax, SGLANG_SHFL_XOR_SYNC_WIDTH(uint32_t(-1), amax, mask, 32));
    sum += SGLANG_SHFL_XOR_SYNC_WIDTH(uint32_t(-1), sum, mask, 32);
  }

  const float d = amax / 127;
  const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

  y[ib].qs[iqs] = q;

  if (iqs > 0) {
    return;
  }

  y[ib].ds.x = __float2half(d);
  y[ib].ds.y = __float2half(sum);
}

template <typename scalar_t>
static void quantize_row_q8_1_cuda(const scalar_t* x, void* vy, const int kx, const int ky, cudaStream_t stream) {
  const int64_t kx_padded = (kx + 512 - 1) / 512 * 512;
  const int block_num_x = (kx_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
  constexpr int MAX_BLOCK_SIZE = 65535;
  for (int off = 0; off < ky; off += MAX_BLOCK_SIZE) {
    const int num_blocks_y = std::min(ky, off + MAX_BLOCK_SIZE) - off;
    const dim3 num_blocks(block_num_x, num_blocks_y, 1);
    const dim3 block_size(CUDA_DEQUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1<<<num_blocks, block_size, 0, stream>>>(
        &x[off * kx], (int32_t*)vy + off * (kx_padded / 32 * 9), kx, kx_padded);
  }
}

torch::Tensor ggml_dequantize(
    torch::Tensor W,  // quant weight
    int64_t type,
    int64_t m,
    int64_t n,
    std::optional<at::ScalarType> const& dtype) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(W));
  auto dtype_ = dtype.value_or(torch::kFloat16);
  auto options = torch::TensorOptions().dtype(dtype_).device(W.device());
  at::Tensor DW = torch::empty({m, n}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  DISPATCH_FLOAT_TYPES(DW.scalar_type(), "ggml_dequantize", [&] {
    auto to_cuda = ggml_get_to_cuda<scalar_t>(type);
    to_cuda((void*)W.data_ptr(), (scalar_t*)DW.data_ptr(), m * n, stream);
  });

  return DW;
}

torch::Tensor ggml_mul_mat_vec_a8(
    torch::Tensor W,  // quant weight
    torch::Tensor X,  // input
    int64_t type,
    int64_t row) {
  int col = X.sizes()[1];
  int vecs = X.sizes()[0];
  const int padded = (col + 512 - 1) / 512 * 512;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  auto options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
  at::Tensor Y = torch::empty({vecs, row}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X = torch::empty({vecs, padded / 32 * 9}, options);
  DISPATCH_FLOAT_TYPES(X.scalar_type(), "ggml_mul_mat_vec_a8", [&] {
    quantize_row_q8_1_cuda<scalar_t>((scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(), col, vecs, stream);
    switch (type) {
      case 2:
        mul_mat_vec_q4_0_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 3:
        mul_mat_vec_q4_1_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 6:
        mul_mat_vec_q5_0_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 7:
        mul_mat_vec_q5_1_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 8:
        mul_mat_vec_q8_0_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 10:
        mul_mat_vec_q2_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 11:
        mul_mat_vec_q3_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 12:
        mul_mat_vec_q4_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 13:
        mul_mat_vec_q5_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 14:
        mul_mat_vec_q6_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 16:
        mul_mat_vec_iq2_xxs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 17:
        mul_mat_vec_iq2_xs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 18:
        mul_mat_vec_iq3_xxs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 19:
        mul_mat_vec_iq1_s_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 20:
        mul_mat_vec_iq4_nl_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 21:
        mul_mat_vec_iq3_s_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 22:
        mul_mat_vec_iq2_s_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 23:
        mul_mat_vec_iq4_xs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 29:
        mul_mat_vec_iq1_m_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
    }
  });
  return Y;
}

torch::Tensor ggml_mul_mat_a8(
    torch::Tensor W,  // quant weight
    torch::Tensor X,  // input
    int64_t type,
    int64_t row) {
  int col = X.sizes()[1];
  int padded = (col + 512 - 1) / 512 * 512;
  int batch = X.sizes()[0];
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  auto options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
  at::Tensor Y = torch::empty({batch, row}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X = torch::empty({batch, padded / 32 * 9}, options);
  DISPATCH_FLOAT_TYPES(X.scalar_type(), "ggml_mul_mat_a8", [&] {
    quantize_row_q8_1_cuda((scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(), col, batch, stream);

    switch (type) {
      case 2:
        ggml_mul_mat_q4_0_q8_1_cuda(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 3:
        ggml_mul_mat_q4_1_q8_1_cuda(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 6:
        ggml_mul_mat_q5_0_q8_1_cuda(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 7:
        ggml_mul_mat_q5_1_q8_1_cuda(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 8:
        ggml_mul_mat_q8_0_q8_1_cuda(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 10:
        ggml_mul_mat_q2_K_q8_1_cuda(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 11:
        ggml_mul_mat_q3_K_q8_1_cuda(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 12:
        ggml_mul_mat_q4_K_q8_1_cuda(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 13:
        ggml_mul_mat_q5_K_q8_1_cuda(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 14:
        ggml_mul_mat_q6_K_q8_1_cuda(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
    }
  });
  return Y;
}

torch::Tensor ggml_moe_a8(
    torch::Tensor X,  // input
    torch::Tensor W,  // expert weights
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_padded,
    int64_t type,
    int64_t row,
    int64_t top_k,
    int64_t tokens) {
  TORCH_CHECK(X.is_cuda() && W.is_cuda() && sorted_token_ids.is_cuda() && expert_ids.is_cuda() && num_tokens_post_padded.is_cuda(),
              "ggml_moe_a8 expects all tensors to be CUDA tensors");
  TORCH_CHECK(X.get_device() == W.get_device() && X.get_device() == sorted_token_ids.get_device() &&
                  X.get_device() == expert_ids.get_device() && X.get_device() == num_tokens_post_padded.get_device(),
              "ggml_moe_a8 expects all tensors on the same CUDA device");
  TORCH_CHECK(X.dim() == 2, "ggml_moe_a8 expects X to be rank 2");
  TORCH_CHECK(W.dim() >= 2, "ggml_moe_a8 expects W to include an expert dimension");
  TORCH_CHECK(X.is_contiguous() && W.is_contiguous(), "ggml_moe_a8 expects contiguous X and W");
  TORCH_CHECK(sorted_token_ids.scalar_type() == torch::kInt32 && expert_ids.scalar_type() == torch::kInt32 &&
                  num_tokens_post_padded.scalar_type() == torch::kInt32,
              "ggml_moe_a8 expects int32 sorted_token_ids, expert_ids, and num_tokens_post_padded");
  TORCH_CHECK(top_k > 0 && tokens == X.sizes()[0], "ggml_moe_a8 received inconsistent top_k/tokens");
  TORCH_CHECK(W.sizes()[0] > 0 && W.sizes()[0] <= 2147483647, "ggml_moe_a8 received an invalid expert count");
  int col = X.sizes()[1];
  int padded = (col + 512 - 1) / 512 * 512;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  auto options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
  at::Tensor Y = torch::empty({tokens * top_k, row}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X = torch::empty({tokens, padded / 32 * 9}, options);
  DISPATCH_FLOAT_TYPES(X.scalar_type(), "ggml_moe_a8", [&] {
    quantize_row_q8_1_cuda((scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(), col, tokens, stream);
    gguf_check_cuda_launch("ggml_moe_a8 quantize_row_q8_1_cuda", stream);
    switch (type) {
      case 2:
        ggml_moe_q4_0_q8_1_cuda(
            (void*)quant_X.data_ptr(),
            (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(),
            W.stride(0),
            static_cast<int>(W.sizes()[0]),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      case 3:
        ggml_moe_q4_1_q8_1_cuda(
            (void*)quant_X.data_ptr(),
            (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(),
            W.stride(0),
            static_cast<int>(W.sizes()[0]),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      case 6:
        ggml_moe_q5_0_q8_1_cuda(
            (void*)quant_X.data_ptr(),
            (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(),
            W.stride(0),
            static_cast<int>(W.sizes()[0]),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      case 7:
        ggml_moe_q5_1_q8_1_cuda(
            (void*)quant_X.data_ptr(),
            (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(),
            W.stride(0),
            static_cast<int>(W.sizes()[0]),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      case 8:
        ggml_moe_q8_0_q8_1_cuda(
            (void*)quant_X.data_ptr(),
            (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(),
            W.stride(0),
            static_cast<int>(W.sizes()[0]),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      case 10:
        ggml_moe_q2_K_q8_1_cuda(
            (void*)quant_X.data_ptr(),
            (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(),
            W.stride(0),
            static_cast<int>(W.sizes()[0]),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      case 11:
        ggml_moe_q3_K_q8_1_cuda(
            (void*)quant_X.data_ptr(),
            (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(),
            W.stride(0),
            static_cast<int>(W.sizes()[0]),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      case 12:
        ggml_moe_q4_K_q8_1_cuda(
            (void*)quant_X.data_ptr(),
            (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(),
            W.stride(0),
            static_cast<int>(W.sizes()[0]),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      case 13:
        ggml_moe_q5_K_q8_1_cuda(
            (void*)quant_X.data_ptr(),
            (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(),
            W.stride(0),
            static_cast<int>(W.sizes()[0]),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      case 14:
        ggml_moe_q6_K_q8_1_cuda(
            (void*)quant_X.data_ptr(),
            (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(),
            W.stride(0),
            static_cast<int>(W.sizes()[0]),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      default:
        TORCH_CHECK(false, "unsupported GGUF MoE quant type: ", type);
    }
    gguf_check_cuda_launch("ggml_moe_a8 qtype kernel", stream);
  });
  return Y;
}

torch::Tensor ggml_moe_a8_vec(
    torch::Tensor X,  // input
    torch::Tensor W,  // expert weights
    torch::Tensor topk_ids,
    int64_t top_k,
    int64_t type,
    int64_t row,
    int64_t tokens) {
  int col = X.sizes()[1];
  const int padded = (col + 512 - 1) / 512 * 512;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  auto options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
  at::Tensor Y = torch::empty({tokens * top_k, row}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X = torch::empty({tokens, padded / 32 * 9}, options);
  DISPATCH_FLOAT_TYPES(X.scalar_type(), "ggml_moe_vec_a8", [&] {
    quantize_row_q8_1_cuda<scalar_t>((scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(), col, tokens, stream);
    switch (type) {
      case 2:
        moe_vec_q4_0_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 3:
        moe_vec_q4_1_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 6:
        moe_vec_q5_0_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 7:
        moe_vec_q5_1_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 8:
        moe_vec_q8_0_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 10:
        moe_vec_q2_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 11:
        moe_vec_q3_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 12:
        moe_vec_q4_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 13:
        moe_vec_q5_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 14:
        moe_vec_q6_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 16:
        moe_vec_iq2_xxs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 17:
        moe_vec_iq2_xs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 18:
        moe_vec_iq3_xxs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 19:
        moe_vec_iq1_s_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 20:
        moe_vec_iq4_nl_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 21:
        moe_vec_iq3_s_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 22:
        moe_vec_iq2_s_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 23:
        moe_vec_iq4_xs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 29:
        moe_vec_iq1_m_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)topk_ids.data_ptr(),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      default:
        TORCH_CHECK(false, "unsupported GGUF MoE vec quant type: ", type);
    }
  });
  return Y;
}

torch::Tensor ggml_moe_a8_vec_weighted_accum(
    torch::Tensor X,  // active input rows
    torch::Tensor W,  // expert weights
    torch::Tensor expert_ids,
    torch::Tensor token_ids,
    torch::Tensor weights,
    int64_t type,
    int64_t row,
    int64_t tokens,
    int64_t output_tokens) {
  TORCH_CHECK(expert_ids.scalar_type() == torch::kInt32, "expert_ids must be int32");
  TORCH_CHECK(token_ids.scalar_type() == torch::kInt64, "token_ids must be int64");
  TORCH_CHECK(weights.scalar_type() == torch::kFloat32, "weights must be float32");
  TORCH_CHECK(X.sizes()[0] == tokens, "X rows must match tokens");
  TORCH_CHECK(expert_ids.numel() >= tokens, "expert_ids must cover tokens");
  TORCH_CHECK(token_ids.numel() >= tokens, "token_ids must cover tokens");
  TORCH_CHECK(weights.numel() >= tokens, "weights must cover tokens");

  int col = X.sizes()[1];
  const int padded = (col + 512 - 1) / 512 * 512;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  auto options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
  at::Tensor Y = torch::zeros({output_tokens, row}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X = torch::empty({tokens, padded / 32 * 9}, options);
  DISPATCH_FLOAT_TYPES(X.scalar_type(), "ggml_moe_vec_weighted_accum_a8", [&] {
    quantize_row_q8_1_cuda<scalar_t>((scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(), col, tokens, stream);
    switch (type) {
      case 10:
        moe_vec_q_weighted_accum_cuda<scalar_t, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int64_t*)token_ids.data_ptr(),
            (float*)weights.data_ptr(),
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 23:
        moe_vec_q_weighted_accum_cuda<scalar_t, QK_K, QI4_XS, block_iq4_xs, 1, vec_dot_iq4_xs_q8_1>(
            (void*)W.data_ptr(),
            (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int64_t*)token_ids.data_ptr(),
            (float*)weights.data_ptr(),
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      default:
        TORCH_CHECK(false, "unsupported GGUF MoE weighted-accum vec quant type: ", type);
    }
    gguf_check_cuda_launch("ggml_moe_a8_vec_weighted_accum qtype kernel", stream);
  });
  return Y;
}

int64_t ggml_moe_get_block_size(int64_t type) {
  switch (type) {
    case 2:
      return MOE_X_Q4_0;
    case 3:
      return MOE_X_Q4_1;
    case 6:
      return MOE_X_Q5_0;
    case 7:
      return MOE_X_Q5_1;
    case 8:
      return MOE_X_Q8_0;
    case 10:
      return MOE_X_Q2_K;
    case 11:
      return MOE_X_Q3_K;
    case 12:
      return MOE_X_Q4_K;
    case 13:
      return MOE_X_Q5_K;
    case 14:
      return MOE_X_Q6_K;
  }
  return 0;
}
