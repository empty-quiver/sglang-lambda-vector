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

torch::Tensor ds4_cuda_optimized_v26_attention(
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

torch::Tensor ds4_cuda_optimized_v27_attention(
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

torch::Tensor ds4_cuda_optimized_v28_attention(
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

torch::Tensor ds4_cuda_optimized_v29_attention(
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

torch::Tensor ds4_cuda_optimized_v30_attention(
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

torch::Tensor ds4_cuda_optimized_v31_attention(
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

torch::Tensor ds4_cuda_optimized_v32_attention(
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

torch::Tensor ds4_cuda_optimized_v33_attention(
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

torch::Tensor ds4_cuda_optimized_v34_attention(
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

torch::Tensor ds4_cuda_optimized_v35_attention(
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

torch::Tensor ds4_cuda_optimized_v36_attention(
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

torch::Tensor ds4_cuda_optimized_v37_attention(
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

torch::Tensor ds4_cuda_optimized_v38_attention(
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

torch::Tensor ds4_cuda_optimized_v39_attention(
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

torch::Tensor ds4_cuda_optimized_v40_attention(
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

torch::Tensor ds4_cuda_optimized_v41_attention(
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

torch::Tensor ds4_cuda_optimized_v42_attention(
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

torch::Tensor ds4_cuda_optimized_v43_attention(
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

torch::Tensor ds4_cuda_optimized_v44_attention(
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
      "ds4_cuda_optimized_v26_attention",
      &ds4_cuda_optimized_v26_attention,
      "Debug CUDA DS4 sparse attention optimized v26 tiny/v23 dispatch");
  m.def(
      "ds4_cuda_optimized_v27_attention",
      &ds4_cuda_optimized_v27_attention,
      "Debug CUDA DS4 sparse attention optimized v27 lane-row QK K staging");
  m.def(
      "ds4_cuda_optimized_v28_attention",
      &ds4_cuda_optimized_v28_attention,
      "Debug CUDA DS4 sparse attention optimized v28 grouped-head K reuse");
  m.def(
      "ds4_cuda_optimized_v29_attention",
      &ds4_cuda_optimized_v29_attention,
      "Debug CUDA DS4 sparse attention optimized v29 grouped-QK score split");
  m.def(
      "ds4_cuda_optimized_v30_attention",
      &ds4_cuda_optimized_v30_attention,
      "Debug CUDA DS4 sparse attention optimized v30 independent grouped-head CTA");
  m.def(
      "ds4_cuda_optimized_v31_attention",
      &ds4_cuda_optimized_v31_attention,
      "Debug CUDA DS4 sparse attention optimized v31 smaller independent grouped-head CTA");
  m.def(
      "ds4_cuda_optimized_v32_attention",
      &ds4_cuda_optimized_v32_attention,
      "Debug CUDA DS4 sparse attention optimized v32 row32 independent grouped-head CTA");
  m.def(
      "ds4_cuda_optimized_v33_attention",
      &ds4_cuda_optimized_v33_attention,
      "Debug CUDA DS4 sparse attention optimized v33 row48 independent grouped-head CTA");
  m.def(
      "ds4_cuda_optimized_v34_attention",
      &ds4_cuda_optimized_v34_attention,
      "Debug CUDA DS4 sparse attention optimized v34 streaming grouped-KV CTA");
  m.def(
      "ds4_cuda_optimized_v35_attention",
      &ds4_cuda_optimized_v35_attention,
      "Debug CUDA DS4 sparse attention optimized v35 online grouped-KV CTA");
  m.def(
      "ds4_cuda_optimized_v36_attention",
      &ds4_cuda_optimized_v36_attention,
      "Debug CUDA DS4 sparse attention optimized v36 dim-chunk streamed output");
  m.def(
      "ds4_cuda_optimized_v37_attention",
      &ds4_cuda_optimized_v37_attention,
      "Debug CUDA DS4 sparse attention optimized v37 shared-QK direct accumulator");
  m.def(
      "ds4_cuda_optimized_v38_attention",
      &ds4_cuda_optimized_v38_attention,
      "Debug CUDA DS4 sparse attention optimized v38 direct partial-acc store");
  m.def(
      "ds4_cuda_optimized_v39_attention",
      &ds4_cuda_optimized_v39_attention,
      "Debug CUDA DS4 sparse attention optimized v39 inline MMA P@V");
  m.def(
      "ds4_cuda_optimized_v40_attention",
      &ds4_cuda_optimized_v40_attention,
      "Debug CUDA DS4 sparse attention optimized v40 direct scaled FP8 staging");
  m.def(
      "ds4_cuda_optimized_v41_attention",
      &ds4_cuda_optimized_v41_attention,
      "Debug CUDA DS4 sparse attention optimized v41 approximate scaled FP8 staging");
  m.def(
      "ds4_cuda_optimized_v42_attention",
      &ds4_cuda_optimized_v42_attention,
      "Debug CUDA DS4 sparse attention optimized v42 prepared-K tile contract");
  m.def(
      "ds4_cuda_optimized_v43_attention",
      &ds4_cuda_optimized_v43_attention,
      "Debug CUDA DS4 sparse attention optimized v43 prepared-KV tile contract");
  m.def(
      "ds4_cuda_optimized_v44_attention",
      &ds4_cuda_optimized_v44_attention,
      "Debug CUDA DS4 sparse attention optimized v44 row32 prepared-KV direct-P contract");
  m.def(
      "ds4_cuda_reference_scores",
      &ds4_cuda_reference_scores,
      "Debug CUDA DS4 sparse attention scalar QK scores");
  m.def(
      "ds4_cuda_v6_mma_scores",
      &ds4_cuda_v6_mma_scores,
      "Debug CUDA DS4 sparse attention v6 BF16 MMA QK scores");
}
