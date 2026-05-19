"""JIT-loaded CUDA debug kernel for DS4 sparse attention fixtures.

This is a first CUDA rung for the DS4 attention ladder. It is deliberately
simple: one CUDA block computes one flattened query/head, streams the packed
SWA and compressed rows, applies one online softmax including the sink logit,
and writes a bf16 output. It is not a production SGLang backend yet.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load


_HERE = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def _load_extension():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    os.environ.setdefault("MAX_JOBS", str(os.cpu_count() or 4))

    return load(
        name="ds4_cuda_reference_attention_opt_v55_direct_prepared_ext",
        sources=[
            str(_HERE / "ds4_cuda_reference_attention.cpp"),
            str(_HERE / "ds4_cuda_reference_attention.cu"),
            str(_HERE / "ds4_cuda_reference_attention_v50.cu"),
        ],
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2", "--use_fast_math", "-lineinfo"],
        verbose=bool(int(os.environ.get("DSV4_CUDA_REF_BUILD_VERBOSE", "0"))),
    )


def _resolve_op(ext, optimized: bool | str | int):
    if optimized is False or optimized == 0 or optimized == "reference":
        return ext.ds4_cuda_reference_attention
    if optimized is True or optimized == 1 or optimized in {"optimized", "v1"}:
        return ext.ds4_cuda_optimized_attention
    if optimized == 2 or optimized in {"v2", "softmax_broadcast"}:
        return ext.ds4_cuda_optimized_v2_attention
    if optimized == 3 or optimized in {"v3", "scale_cache"}:
        return ext.ds4_cuda_optimized_v3_attention
    if optimized == 4 or optimized in {"v4", "grouped_heads"}:
        return ext.ds4_cuda_optimized_v4_attention
    if optimized == 5 or optimized in {"v5", "reduction_total_slot"}:
        return ext.ds4_cuda_optimized_v5_attention
    if optimized == 7 or optimized in {"v7", "fused_mma"}:
        return ext.ds4_cuda_optimized_v7_attention
    if optimized == 8 or optimized in {"v8", "split_row_fused_mma"}:
        return ext.ds4_cuda_optimized_v8_attention
    if optimized == 9 or optimized in {"v9", "split_row_scale_cache"}:
        return ext.ds4_cuda_optimized_v9_attention
    if optimized == 10 or optimized in {"v10", "tensor_core_pv"}:
        return ext.ds4_cuda_optimized_v10_attention
    if optimized == 11 or optimized in {"v11", "parallel_tensor_core_pv"}:
        return ext.ds4_cuda_optimized_v11_attention
    if optimized == 12 or optimized in {"v12", "cached_p_tensor_core_pv"}:
        return ext.ds4_cuda_optimized_v12_attention
    if optimized == 13 or optimized in {"v13", "bf16_partial_acc_tensor_core_pv"}:
        return ext.ds4_cuda_optimized_v13_attention
    if optimized == 14 or optimized in {"v14", "specialized_v_decode_tensor_core_pv"}:
        return ext.ds4_cuda_optimized_v14_attention
    if optimized == 15 or optimized in {"v15", "rowgroup_accum_tensor_core_pv"}:
        return ext.ds4_cuda_optimized_v15_attention
    if optimized == 16 or optimized in {"v16", "coalesced_k_staging_tensor_core_pv"}:
        return ext.ds4_cuda_optimized_v16_attention
    if optimized == 17 or optimized in {"v17", "q_tile_reuse_tensor_core_pv"}:
        return ext.ds4_cuda_optimized_v17_attention
    if optimized == 18 or optimized in {"v18", "reduce_scale_reuse_tensor_core_pv"}:
        return ext.ds4_cuda_optimized_v18_attention
    if optimized == 19 or optimized in {"v19", "score_state_split_tensor_core_pv"}:
        return ext.ds4_cuda_optimized_v19_attention
    if optimized == 20 or optimized in {"v20", "parallel_score_state_finalize"}:
        return ext.ds4_cuda_optimized_v20_attention
    if optimized == 21 or optimized in {"v21", "compact_score_metadata"}:
        return ext.ds4_cuda_optimized_v21_attention
    if optimized == 22 or optimized in {"v22", "grouped_score_finalize"}:
        return ext.ds4_cuda_optimized_v22_attention
    if optimized == 23 or optimized in {"v23", "dim_split_reduce"}:
        return ext.ds4_cuda_optimized_v23_attention
    if optimized == 24 or optimized in {"v24", "row_contiguous_k_colmajor"}:
        return ext.ds4_cuda_optimized_v24_attention
    if optimized == 25 or optimized in {"v25", "whole_span_fused_pv"}:
        return ext.ds4_cuda_optimized_v25_attention
    if optimized == 26 or optimized in {"v26", "tiny_v5_v23_dispatch"}:
        return ext.ds4_cuda_optimized_v26_attention
    if optimized == 27 or optimized in {"v27", "lane_row_qk_k_staging"}:
        return ext.ds4_cuda_optimized_v27_attention
    if optimized == 28 or optimized in {"v28", "grouped_head_k_reuse"}:
        return ext.ds4_cuda_optimized_v28_attention
    if optimized == 29 or optimized in {"v29", "grouped_qk_score_split"}:
        return ext.ds4_cuda_optimized_v29_attention
    if optimized == 30 or optimized in {"v30", "independent_grouped_head_cta"}:
        return ext.ds4_cuda_optimized_v30_attention
    if optimized == 31 or optimized in {"v31", "smaller_independent_grouped_head_cta"}:
        return ext.ds4_cuda_optimized_v31_attention
    if optimized == 32 or optimized in {"v32", "row32_independent_grouped_head_cta"}:
        return ext.ds4_cuda_optimized_v32_attention
    if optimized == 33 or optimized in {"v33", "row48_independent_grouped_head_cta"}:
        return ext.ds4_cuda_optimized_v33_attention
    if optimized == 34 or optimized in {"v34", "streaming_grouped_kv_cta"}:
        return ext.ds4_cuda_optimized_v34_attention
    if optimized == 35 or optimized in {"v35", "online_grouped_kv_cta"}:
        return ext.ds4_cuda_optimized_v35_attention
    if optimized == 36 or optimized in {"v36", "dim_chunk_streamed_output"}:
        return ext.ds4_cuda_optimized_v36_attention
    if optimized == 37 or optimized in {"v37", "shared_qk_direct_acc"}:
        return ext.ds4_cuda_optimized_v37_attention
    if optimized == 38 or optimized in {"v38", "direct_partial_acc_store"}:
        return ext.ds4_cuda_optimized_v38_attention
    if optimized == 39 or optimized in {"v39", "inline_mma_pv"}:
        return ext.ds4_cuda_optimized_v39_attention
    if optimized == 40 or optimized in {"v40", "direct_scaled_fp8"}:
        return ext.ds4_cuda_optimized_v40_attention
    if optimized == 41 or optimized in {"v41", "approx_scaled_fp8"}:
        return ext.ds4_cuda_optimized_v41_attention
    if optimized == 42 or optimized in {"v42", "prepared_k_tiles"}:
        return ext.ds4_cuda_optimized_v42_attention
    if optimized == 43 or optimized in {"v43", "prepared_kv_tiles"}:
        return ext.ds4_cuda_optimized_v43_attention
    if optimized == 44 or optimized in {"v44", "row32_prepared_kv_direct_p"}:
        return ext.ds4_cuda_optimized_v44_attention
    if optimized == 45 or optimized in {"v45", "prepared_kv_direct_p"}:
        return ext.ds4_cuda_optimized_v45_attention
    if optimized == 46 or optimized in {"v46", "prepared_kv_rolling_q"}:
        return ext.ds4_cuda_optimized_v46_attention
    if optimized == 47 or optimized in {"v47", "prepared_kv_half_pv_warps"}:
        return ext.ds4_cuda_optimized_v47_attention
    if optimized == 48 or optimized in {"v48", "prepared_p_grouped_qk"}:
        return ext.ds4_cuda_optimized_v48_attention
    if optimized == 49 or optimized in {"v49a", "direct_prepared_k_wmma"}:
        return ext.ds4_cuda_optimized_v49a_attention
    if optimized == 50 or optimized in {"v49b", "direct_prepared_v_wmma"}:
        return ext.ds4_cuda_optimized_v49b_attention
    if optimized == 51 or optimized in {"v49", "direct_prepared_kv_wmma"}:
        return ext.ds4_cuda_optimized_v49_attention
    if optimized == 52 or optimized in {"v50", "standalone_direct_prepared_kv_wmma"}:
        return ext.ds4_cuda_optimized_v50_attention
    if optimized == 53 or optimized in {"v51", "warp_softmax_direct_prepared_kv_wmma"}:
        return ext.ds4_cuda_optimized_v51_attention
    if optimized == 54 or optimized in {"v52", "rowgroup_k_direct_prepared_kv_wmma"}:
        return ext.ds4_cuda_optimized_v52_attention
    if optimized == 55 or optimized in {"v53", "warp_softmax_rowgroup_k_direct_prepared_kv_wmma"}:
        return ext.ds4_cuda_optimized_v53_attention
    if optimized == 56 or optimized in {"v54", "staged_k_warp_softmax_rowgroup_direct_prepared_kv_wmma"}:
        return ext.ds4_cuda_optimized_v54_attention
    if optimized == 57 or optimized in {"v55", "dimsplit_qk_warp_softmax_rowgroup_direct_prepared_kv_wmma"}:
        return ext.ds4_cuda_optimized_v55_attention
    raise ValueError(f"unknown DS4 CUDA attention variant: {optimized!r}")


def _empty_cuda_tensor(device: torch.device) -> torch.Tensor:
    return torch.empty(0, device=device)


def _cuda_i32(tensor: torch.Tensor | None, device: torch.device) -> torch.Tensor:
    if tensor is None:
        return torch.empty(0, dtype=torch.int32, device=device)
    return tensor.to(device=device, dtype=torch.int32, non_blocking=False).contiguous()


def _cuda_u8(tensor: torch.Tensor | None, device: torch.device) -> torch.Tensor:
    if tensor is None:
        return torch.empty(0, dtype=torch.uint8, device=device)
    return tensor.to(device=device, dtype=torch.uint8, non_blocking=False).contiguous()


def ds4_cuda_sparse_attention(
    *,
    q: torch.Tensor,
    swa_k_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_topk_lengths: torch.Tensor,
    swa_page_size: int,
    softmax_scale: float,
    attn_sink: torch.Tensor | None = None,
    extra_k_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_topk_lengths: torch.Tensor | None = None,
    extra_page_size: int | None = None,
    optimized: bool | str | int = False,
) -> torch.Tensor:
    """Run the debug CUDA DS4 sparse attention kernel."""
    ext = _load_extension()
    device = torch.device("cuda")
    q_cuda = q.to(device=device, dtype=torch.bfloat16, non_blocking=False).contiguous()

    sink_cuda = (
        _empty_cuda_tensor(device)
        if attn_sink is None
        else attn_sink.to(device=device, dtype=torch.float32, non_blocking=False).contiguous()
    )

    has_extra = extra_k_cache is not None
    op = _resolve_op(ext, optimized)
    return op(
        q_cuda,
        _cuda_u8(swa_k_cache, device),
        _cuda_i32(swa_indices, device),
        _cuda_i32(swa_topk_lengths, device),
        int(swa_page_size),
        float(softmax_scale),
        sink_cuda,
        _cuda_u8(extra_k_cache, device),
        _cuda_i32(extra_indices, device),
        _cuda_i32(extra_topk_lengths, device),
        int((extra_page_size or 0) if has_extra else 0),
    )


def ds4_cuda_sparse_attention_from_fixture(
    fixture: dict[str, Any],
    *,
    optimized: bool | str | int = False,
) -> torch.Tensor:
    """Run the CUDA kernel using a fixture emitted by the SGLang DS4 hook."""
    if str(fixture.get("layout", "dsv4_packed")) != "dsv4_packed":
        raise ValueError("CUDA debug kernel only supports dsv4_packed fixtures")

    return ds4_cuda_sparse_attention(
        q=fixture["q"],
        swa_k_cache=fixture["swa_k_cache"],
        swa_indices=fixture["swa_indices"],
        swa_topk_lengths=fixture["swa_topk_lengths"],
        swa_page_size=int(fixture["swa_page_size"]),
        softmax_scale=float(fixture["softmax_scale"]),
        attn_sink=fixture.get("attn_sink"),
        extra_k_cache=fixture.get("extra_k_cache"),
        extra_indices=fixture.get("extra_indices"),
        extra_topk_lengths=fixture.get("extra_topk_lengths"),
        extra_page_size=(
            int(fixture["extra_page_size"])
            if fixture.get("extra_page_size") is not None
            else None
        ),
        optimized=optimized,
    )


def ds4_cuda_sparse_scores(
    *,
    q: torch.Tensor,
    swa_k_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_topk_lengths: torch.Tensor,
    swa_page_size: int,
    softmax_scale: float,
    extra_k_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_topk_lengths: torch.Tensor | None = None,
    extra_page_size: int | None = None,
    tensor_core: bool = False,
) -> torch.Tensor:
    """Run the debug CUDA DS4 pre-softmax QK score kernel."""
    ext = _load_extension()
    device = torch.device("cuda")
    q_cuda = q.to(device=device, dtype=torch.bfloat16, non_blocking=False).contiguous()

    has_extra = extra_k_cache is not None
    op = ext.ds4_cuda_v6_mma_scores if tensor_core else ext.ds4_cuda_reference_scores
    return op(
        q_cuda,
        _cuda_u8(swa_k_cache, device),
        _cuda_i32(swa_indices, device),
        _cuda_i32(swa_topk_lengths, device),
        int(swa_page_size),
        float(softmax_scale),
        _cuda_u8(extra_k_cache, device),
        _cuda_i32(extra_indices, device),
        _cuda_i32(extra_topk_lengths, device),
        int((extra_page_size or 0) if has_extra else 0),
    )


def ds4_cuda_sparse_scores_from_fixture(
    fixture: dict[str, Any],
    *,
    tensor_core: bool = False,
) -> torch.Tensor:
    """Run the CUDA QK score kernel using a fixture emitted by the DS4 hook."""
    if str(fixture.get("layout", "dsv4_packed")) != "dsv4_packed":
        raise ValueError("CUDA debug kernel only supports dsv4_packed fixtures")

    return ds4_cuda_sparse_scores(
        q=fixture["q"],
        swa_k_cache=fixture["swa_k_cache"],
        swa_indices=fixture["swa_indices"],
        swa_topk_lengths=fixture["swa_topk_lengths"],
        swa_page_size=int(fixture["swa_page_size"]),
        softmax_scale=float(fixture["softmax_scale"]),
        extra_k_cache=fixture.get("extra_k_cache"),
        extra_indices=fixture.get("extra_indices"),
        extra_topk_lengths=fixture.get("extra_topk_lengths"),
        extra_page_size=(
            int(fixture["extra_page_size"])
            if fixture.get("extra_page_size") is not None
            else None
        ),
        tensor_core=tensor_core,
    )
