"""Synthetic DS4 attention fixtures for kernel stress tests.

These fixtures mimic the SGLang DeepSeek V4 attention fixture contract without
launching a model. They are intentionally synthetic: the point is to stress page
math, packed row strides, longer SWA/compressed lengths, sink logits, and online
softmax before spending time on slow real long-prompt captures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

try:
    from .ds4_torch_reference_attention import (
        DS4_HEAD_DIM,
        DS4_NOPE_DIM,
        DS4_PACKED_BYTES,
        DS4_ROPE_DIM,
        DS4_SCALE_DIM,
    )
except ImportError:
    from ds4_torch_reference_attention import (  # type: ignore
        DS4_HEAD_DIM,
        DS4_NOPE_DIM,
        DS4_PACKED_BYTES,
        DS4_ROPE_DIM,
        DS4_SCALE_DIM,
    )


DS4_CAPTURE_ROW_STRIDE = 584


@dataclass(frozen=True)
class SyntheticDS4FixtureSpec:
    name: str
    batch: int
    heads: int
    swa_len: int
    swa_width: int
    swa_page_size: int
    extra_len: int
    extra_width: int
    extra_page_size: int
    compress_ratio: int
    seed: int
    row_stride: int = DS4_CAPTURE_ROW_STRIDE


DEFAULT_LONG_SPECS = [
    SyntheticDS4FixtureSpec(
        name="swa8_extra8_cr4",
        batch=2,
        heads=64,
        swa_len=8,
        swa_width=16,
        swa_page_size=256,
        extra_len=8,
        extra_width=16,
        extra_page_size=64,
        compress_ratio=4,
        seed=1001,
    ),
    SyntheticDS4FixtureSpec(
        name="swa32_extra64_cr4",
        batch=2,
        heads=64,
        swa_len=32,
        swa_width=64,
        swa_page_size=256,
        extra_len=64,
        extra_width=128,
        extra_page_size=64,
        compress_ratio=4,
        seed=1002,
    ),
    SyntheticDS4FixtureSpec(
        name="swa128_extra512_cr4",
        batch=2,
        heads=64,
        swa_len=128,
        swa_width=128,
        swa_page_size=256,
        extra_len=512,
        extra_width=512,
        extra_page_size=64,
        compress_ratio=4,
        seed=1003,
    ),
    SyntheticDS4FixtureSpec(
        name="swa128_extra64_cr128",
        batch=2,
        heads=64,
        swa_len=128,
        swa_width=128,
        swa_page_size=256,
        extra_len=64,
        extra_width=64,
        extra_page_size=2,
        compress_ratio=128,
        seed=1004,
    ),
]


def _randn(
    shape: tuple[int, ...],
    generator: torch.Generator,
    *,
    scale: float,
) -> torch.Tensor:
    return torch.randn(shape, generator=generator, dtype=torch.float32) * scale


def _make_packed_cache(
    rows: torch.Tensor,
    page_size: int,
    row_stride: int,
) -> torch.Tensor:
    if rows.ndim != 2 or rows.shape[-1] != DS4_HEAD_DIM:
        raise ValueError(f"rows must be [tokens, {DS4_HEAD_DIM}], got {rows.shape}")
    if row_stride < DS4_PACKED_BYTES:
        raise ValueError(f"row_stride {row_stride} < packed payload {DS4_PACKED_BYTES}")

    num_pages = (rows.shape[0] + page_size - 1) // page_size
    cache = torch.zeros(num_pages, page_size, 1, row_stride, dtype=torch.uint8)
    flat = cache.view(-1, 1, row_stride)[: rows.shape[0], 0]

    nope = rows[:, :DS4_NOPE_DIM].contiguous().to(torch.float8_e4m3fn)
    rope = rows[:, DS4_NOPE_DIM:].contiguous().to(torch.bfloat16)
    flat[:, :DS4_NOPE_DIM] = nope.view(torch.uint8)
    flat[:, DS4_NOPE_DIM : DS4_NOPE_DIM + DS4_ROPE_DIM * 2] = rope.view(
        torch.uint8
    )
    flat[
        :,
        DS4_NOPE_DIM
        + DS4_ROPE_DIM * 2 : DS4_NOPE_DIM
        + DS4_ROPE_DIM * 2
        + DS4_SCALE_DIM,
    ] = 127
    return cache


def _make_indices(
    *,
    batch: int,
    width: int,
    length: int,
    total_tokens: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if length > width:
        raise ValueError(f"length {length} > width {width}")
    indices = torch.full((batch, 1, width), -1, dtype=torch.int32)
    lengths = torch.full((batch,), length, dtype=torch.int32)
    for b in range(batch):
        perm = torch.randperm(total_tokens, generator=generator, dtype=torch.int64)
        selected = perm[:length].to(torch.int32)
        if length > 1 and b % 2:
            selected = selected.flip(0)
        indices[b, 0, :length] = selected
    return indices, lengths


def make_synthetic_dsv4_attention_fixture(
    spec: SyntheticDS4FixtureSpec,
) -> dict[str, Any]:
    """Return a fixture dict compatible with the Torch and CUDA DS4 harnesses."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(spec.seed)

    swa_tokens = max(spec.swa_len + 17, spec.swa_page_size * 3)
    extra_tokens = max(spec.extra_len + 17, spec.extra_page_size * 16)

    q = _randn(
        (spec.batch, 1, spec.heads, DS4_HEAD_DIM),
        generator,
        scale=0.05,
    ).to(torch.bfloat16)
    swa_rows = _randn((swa_tokens, DS4_HEAD_DIM), generator, scale=0.20)
    extra_rows = _randn((extra_tokens, DS4_HEAD_DIM), generator, scale=0.20)

    swa_indices, swa_lengths = _make_indices(
        batch=spec.batch,
        width=spec.swa_width,
        length=spec.swa_len,
        total_tokens=swa_tokens,
        generator=generator,
    )
    extra_indices, extra_lengths = _make_indices(
        batch=spec.batch,
        width=spec.extra_width,
        length=spec.extra_len,
        total_tokens=extra_tokens,
        generator=generator,
    )

    return {
        "name": spec.name,
        "q": q,
        "swa_k_cache": _make_packed_cache(
            swa_rows,
            spec.swa_page_size,
            spec.row_stride,
        ),
        "swa_indices": swa_indices,
        "swa_topk_lengths": swa_lengths,
        "swa_page_size": spec.swa_page_size,
        "softmax_scale": DS4_HEAD_DIM**-0.5,
        "attn_sink": _randn((spec.heads,), generator, scale=0.10),
        "extra_k_cache": _make_packed_cache(
            extra_rows,
            spec.extra_page_size,
            spec.row_stride,
        ),
        "extra_indices": extra_indices,
        "extra_topk_lengths": extra_lengths,
        "extra_page_size": spec.extra_page_size,
        "layout": "dsv4_packed",
        "layer_id": -1,
        "compress_ratio": spec.compress_ratio,
        "synthetic": True,
        "spec": spec.__dict__,
    }
