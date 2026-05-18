#include <torch/extension.h>

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
    int64_t extra_page_size);

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
    int64_t extra_page_size);

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
    int64_t extra_page_size);

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
    int64_t extra_page_size);

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
    int64_t extra_page_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "ds4_cuda_reference_attention",
      &ds4_cuda_reference_attention,
      "Debug CUDA DS4 sparse attention reference");
  m.def(
      "ds4_cuda_optimized_attention",
      &ds4_cuda_optimized_attention,
      "Debug CUDA DS4 sparse attention optimized v1");
  m.def(
      "ds4_cuda_optimized_v2_attention",
      &ds4_cuda_optimized_v2_attention,
      "Debug CUDA DS4 sparse attention optimized v2");
  m.def(
      "ds4_cuda_optimized_v3_attention",
      &ds4_cuda_optimized_v3_attention,
      "Debug CUDA DS4 sparse attention optimized v3");
  m.def(
      "ds4_cuda_optimized_v4_attention",
      &ds4_cuda_optimized_v4_attention,
      "Debug CUDA DS4 sparse attention optimized v4");
}
