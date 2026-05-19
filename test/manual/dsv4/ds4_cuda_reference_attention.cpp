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

torch::Tensor ds4_cuda_optimized_v45_attention(
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

torch::Tensor ds4_cuda_optimized_v46_attention(
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

torch::Tensor ds4_cuda_optimized_v47_attention(
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

torch::Tensor ds4_cuda_optimized_v48_attention(
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

torch::Tensor ds4_cuda_optimized_v49a_attention(
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

torch::Tensor ds4_cuda_optimized_v49b_attention(
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

torch::Tensor ds4_cuda_optimized_v49_attention(
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

torch::Tensor ds4_cuda_optimized_v50_attention(
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

torch::Tensor ds4_cuda_optimized_v51_attention(
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

torch::Tensor ds4_cuda_optimized_v52_attention(
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

torch::Tensor ds4_cuda_optimized_v53_attention(
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

torch::Tensor ds4_cuda_optimized_v54_attention(
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

torch::Tensor ds4_cuda_optimized_v55_attention(
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

torch::Tensor ds4_cuda_optimized_v56_attention(
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

torch::Tensor ds4_cuda_optimized_v57_attention(
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

#define DSV4_DECLARE_OPTIMIZED_ATTENTION(NAME) \
  torch::Tensor NAME(                          \
      torch::Tensor q,                         \
      torch::Tensor swa_k_cache,               \
      torch::Tensor swa_indices,               \
      torch::Tensor swa_topk_lengths,          \
      int64_t swa_page_size,                   \
      double softmax_scale,                    \
      torch::Tensor attn_sink,                 \
      torch::Tensor extra_k_cache,             \
      torch::Tensor extra_indices,             \
      torch::Tensor extra_topk_lengths,        \
      int64_t extra_page_size);

DSV4_DECLARE_OPTIMIZED_ATTENTION(ds4_cuda_optimized_v58a_attention)
DSV4_DECLARE_OPTIMIZED_ATTENTION(ds4_cuda_optimized_v58b_attention)
DSV4_DECLARE_OPTIMIZED_ATTENTION(ds4_cuda_optimized_v58_attention)
DSV4_DECLARE_OPTIMIZED_ATTENTION(ds4_cuda_optimized_v59_attention)
DSV4_DECLARE_OPTIMIZED_ATTENTION(ds4_cuda_optimized_v60_attention)
DSV4_DECLARE_OPTIMIZED_ATTENTION(ds4_cuda_optimized_v61_attention)
DSV4_DECLARE_OPTIMIZED_ATTENTION(ds4_cuda_optimized_v62a_attention)
DSV4_DECLARE_OPTIMIZED_ATTENTION(ds4_cuda_optimized_v62b_attention)
DSV4_DECLARE_OPTIMIZED_ATTENTION(ds4_cuda_optimized_v62c_attention)

#undef DSV4_DECLARE_OPTIMIZED_ATTENTION

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
      "ds4_cuda_optimized_v45_attention",
      &ds4_cuda_optimized_v45_attention,
      "Debug CUDA DS4 sparse attention optimized v45 prepared-KV direct-P contract");
  m.def(
      "ds4_cuda_optimized_v46_attention",
      &ds4_cuda_optimized_v46_attention,
      "Debug CUDA DS4 sparse attention optimized v46 prepared-KV rolling-Q direct-P contract");
  m.def(
      "ds4_cuda_optimized_v47_attention",
      &ds4_cuda_optimized_v47_attention,
      "Debug CUDA DS4 sparse attention optimized v47 prepared-KV half-PV-warp direct-P contract");
  m.def(
      "ds4_cuda_optimized_v48_attention",
      &ds4_cuda_optimized_v48_attention,
      "Debug CUDA DS4 sparse attention optimized v48 prepared-P split contract");
  m.def(
      "ds4_cuda_optimized_v49a_attention",
      &ds4_cuda_optimized_v49a_attention,
      "Debug CUDA DS4 sparse attention optimized v49a direct prepared-K WMMA load");
  m.def(
      "ds4_cuda_optimized_v49b_attention",
      &ds4_cuda_optimized_v49b_attention,
      "Debug CUDA DS4 sparse attention optimized v49b direct prepared-V WMMA load");
  m.def(
      "ds4_cuda_optimized_v49_attention",
      &ds4_cuda_optimized_v49_attention,
      "Debug CUDA DS4 sparse attention optimized v49 direct prepared-KV WMMA loads");
  m.def(
      "ds4_cuda_optimized_v50_attention",
      &ds4_cuda_optimized_v50_attention,
      "Debug CUDA DS4 sparse attention optimized v50 standalone direct prepared-KV WMMA loads");
  m.def(
      "ds4_cuda_optimized_v51_attention",
      &ds4_cuda_optimized_v51_attention,
      "Debug CUDA DS4 sparse attention optimized v51 warp-softmax standalone direct prepared-KV WMMA loads");
  m.def(
      "ds4_cuda_optimized_v52_attention",
      &ds4_cuda_optimized_v52_attention,
      "Debug CUDA DS4 sparse attention optimized v52 row-group prepared-K standalone direct prepared-KV WMMA loads");
  m.def(
      "ds4_cuda_optimized_v53_attention",
      &ds4_cuda_optimized_v53_attention,
      "Debug CUDA DS4 sparse attention optimized v53 warp-softmax row-group prepared-K standalone direct prepared-KV WMMA loads");
  m.def(
      "ds4_cuda_optimized_v54_attention",
      &ds4_cuda_optimized_v54_attention,
      "Debug CUDA DS4 sparse attention optimized v54 staged-K warp-softmax row-group standalone direct prepared-KV WMMA loads");
  m.def(
      "ds4_cuda_optimized_v55_attention",
      &ds4_cuda_optimized_v55_attention,
      "Debug CUDA DS4 sparse attention optimized v55 common-path dim-split QK standalone direct prepared-KV WMMA loads");
  m.def(
      "ds4_cuda_optimized_v56_attention",
      &ds4_cuda_optimized_v56_attention,
      "Debug CUDA DS4 sparse attention optimized v56 common-path prepared-KV producer with v53 consumer");
  m.def(
      "ds4_cuda_optimized_v57_attention",
      &ds4_cuda_optimized_v57_attention,
      "Debug CUDA DS4 sparse attention optimized v57 common-path single-scale prepared-KV producer with v53 consumer");
  m.def(
      "ds4_cuda_optimized_v58a_attention",
      &ds4_cuda_optimized_v58a_attention,
      "Debug CUDA DS4 sparse attention optimized v58a K matrix-B col-major prepared contract");
  m.def(
      "ds4_cuda_optimized_v58b_attention",
      &ds4_cuda_optimized_v58b_attention,
      "Debug CUDA DS4 sparse attention optimized v58b V matrix-B col-major prepared contract");
  m.def(
      "ds4_cuda_optimized_v58_attention",
      &ds4_cuda_optimized_v58_attention,
      "Debug CUDA DS4 sparse attention optimized v58 K/V matrix-B col-major prepared contract");
  m.def(
      "ds4_cuda_optimized_v59_attention",
      &ds4_cuda_optimized_v59_attention,
      "Debug CUDA DS4 sparse attention optimized v59 Q/P matrix-A col-major shared contract");
  m.def(
      "ds4_cuda_optimized_v60_attention",
      &ds4_cuda_optimized_v60_attention,
      "Debug CUDA DS4 sparse attention optimized v60 combined A/B col-major fragment contract");
  m.def(
      "ds4_cuda_optimized_v61_attention",
      &ds4_cuda_optimized_v61_attention,
      "Debug CUDA DS4 sparse attention optimized v61 staged V matrix-B col-major contract");
  m.def(
      "ds4_cuda_optimized_v62a_attention",
      &ds4_cuda_optimized_v62a_attention,
      "Debug CUDA DS4 sparse attention optimized v62a stream-2 row-tile online V matrix-B contract");
  m.def(
      "ds4_cuda_optimized_v62b_attention",
      &ds4_cuda_optimized_v62b_attention,
      "Debug CUDA DS4 sparse attention optimized v62b stream-4 row-tile online V matrix-B contract");
  m.def(
      "ds4_cuda_optimized_v62c_attention",
      &ds4_cuda_optimized_v62c_attention,
      "Debug CUDA DS4 sparse attention optimized v62c stream-2 row-tile online staged V contract");
  m.def(
      "ds4_cuda_reference_scores",
      &ds4_cuda_reference_scores,
      "Debug CUDA DS4 sparse attention scalar QK scores");
  m.def(
      "ds4_cuda_v6_mma_scores",
      &ds4_cuda_v6_mma_scores,
      "Debug CUDA DS4 sparse attention v6 BF16 MMA QK scores");
}
