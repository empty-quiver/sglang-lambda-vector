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
        name="ds4_cuda_reference_attention_opt_v5_ext",
        sources=[
            str(_HERE / "ds4_cuda_reference_attention.cpp"),
            str(_HERE / "ds4_cuda_reference_attention.cu"),
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
