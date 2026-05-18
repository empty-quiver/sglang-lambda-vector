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
    int64_t extra_page_size);

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
    int64_t extra_page_size);

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
    int64_t extra_page_size);

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
    int64_t extra_page_size);

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
    int64_t extra_page_size);

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
    int64_t extra_page_size);

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
    int64_t extra_page_size);

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
    int64_t extra_page_size);

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
    int64_t extra_page_size);

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
    int64_t extra_page_size);

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
    int64_t extra_page_size);

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
    int64_t extra_page_size);

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
    int64_t extra_page_size);

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
    int64_t extra_page_size);

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
    int64_t extra_page_size);

torch::Tensor ds4_cuda_optimized_v21_attention(
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

torch::Tensor ds4_cuda_optimized_v22_attention(
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

torch::Tensor ds4_cuda_optimized_v23_attention(
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

torch::Tensor ds4_cuda_optimized_v24_attention(
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

torch::Tensor ds4_cuda_optimized_v25_attention(
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
    int64_t extra_page_size);

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
  m.def(
      "ds4_cuda_optimized_v5_attention",
      &ds4_cuda_optimized_v5_attention,
      "Debug CUDA DS4 sparse attention optimized v5");
  m.def(
      "ds4_cuda_optimized_v7_attention",
      &ds4_cuda_optimized_v7_attention,
      "Debug CUDA DS4 sparse attention optimized v7 fused MMA");
  m.def(
      "ds4_cuda_optimized_v8_attention",
      &ds4_cuda_optimized_v8_attention,
      "Debug CUDA DS4 sparse attention optimized v8 split-row fused MMA");
  m.def(
      "ds4_cuda_optimized_v9_attention",
      &ds4_cuda_optimized_v9_attention,
      "Debug CUDA DS4 sparse attention optimized v9 cached-scale split-row fused MMA");
  m.def(
      "ds4_cuda_optimized_v10_attention",
      &ds4_cuda_optimized_v10_attention,
      "Debug CUDA DS4 sparse attention optimized v10 tensor-core P@V");
  m.def(
      "ds4_cuda_optimized_v11_attention",
      &ds4_cuda_optimized_v11_attention,
      "Debug CUDA DS4 sparse attention optimized v11 parallel tensor-core P@V");
  m.def(
      "ds4_cuda_optimized_v12_attention",
      &ds4_cuda_optimized_v12_attention,
      "Debug CUDA DS4 sparse attention optimized v12 cached-P tensor-core P@V");
  m.def(
      "ds4_cuda_optimized_v13_attention",
      &ds4_cuda_optimized_v13_attention,
      "Debug CUDA DS4 sparse attention optimized v13 BF16 partial-acc P@V");
  m.def(
      "ds4_cuda_optimized_v14_attention",
      &ds4_cuda_optimized_v14_attention,
      "Debug CUDA DS4 sparse attention optimized v14 specialized V decode P@V");
  m.def(
      "ds4_cuda_optimized_v15_attention",
      &ds4_cuda_optimized_v15_attention,
      "Debug CUDA DS4 sparse attention optimized v15 row-group accumulating P@V");
  m.def(
      "ds4_cuda_optimized_v16_attention",
      &ds4_cuda_optimized_v16_attention,
      "Debug CUDA DS4 sparse attention optimized v16 coalesced K staging");
  m.def(
      "ds4_cuda_optimized_v17_attention",
      &ds4_cuda_optimized_v17_attention,
      "Debug CUDA DS4 sparse attention optimized v17 Q tile reuse");
  m.def(
      "ds4_cuda_optimized_v18_attention",
      &ds4_cuda_optimized_v18_attention,
      "Debug CUDA DS4 sparse attention optimized v18 reduce-scale reuse");
  m.def(
      "ds4_cuda_optimized_v19_attention",
      &ds4_cuda_optimized_v19_attention,
      "Debug CUDA DS4 sparse attention optimized v19 score-state split");
  m.def(
      "ds4_cuda_optimized_v20_attention",
      &ds4_cuda_optimized_v20_attention,
      "Debug CUDA DS4 sparse attention optimized v20 parallel score-state finalize");
  m.def(
      "ds4_cuda_optimized_v21_attention",
      &ds4_cuda_optimized_v21_attention,
      "Debug CUDA DS4 sparse attention optimized v21 compact score metadata");
  m.def(
      "ds4_cuda_optimized_v22_attention",
      &ds4_cuda_optimized_v22_attention,
      "Debug CUDA DS4 sparse attention optimized v22 grouped score finalize");
  m.def(
      "ds4_cuda_optimized_v23_attention",
      &ds4_cuda_optimized_v23_attention,
      "Debug CUDA DS4 sparse attention optimized v23 dimension-split v18 reduce");
  m.def(
      "ds4_cuda_optimized_v24_attention",
      &ds4_cuda_optimized_v24_attention,
      "Debug CUDA DS4 sparse attention optimized v24 row-contiguous K staging");
  m.def(
      "ds4_cuda_optimized_v25_attention",
      &ds4_cuda_optimized_v25_attention,
      "Debug CUDA DS4 sparse attention optimized v25 whole-span fused P@V");
  m.def(
      "ds4_cuda_reference_scores",
      &ds4_cuda_reference_scores,
      "Debug CUDA DS4 sparse attention scalar QK scores");
  m.def(
      "ds4_cuda_v6_mma_scores",
      &ds4_cuda_v6_mma_scores,
      "Debug CUDA DS4 sparse attention v6 BF16 MMA QK scores");
}
