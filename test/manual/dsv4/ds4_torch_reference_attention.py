#!/usr/bin/env python3
"""Torch-only DS4 sparse attention reference.

This is intentionally boring. It gathers rows, dequantizes explicitly, builds
one score matrix, applies one softmax over SWA + compressed rows + optional
sink, and compares outputs. It is a correctness oracle, not a fast path.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch


DS4_NOPE_DIM = 448
DS4_ROPE_DIM = 64
DS4_HEAD_DIM = DS4_NOPE_DIM + DS4_ROPE_DIM
DS4_SCALE_GROUP = 64
DS4_SCALE_DIM = DS4_NOPE_DIM // DS4_SCALE_GROUP
DS4_PACKED_BYTES = DS4_NOPE_DIM + DS4_ROPE_DIM * 2 + DS4_SCALE_DIM
MAX_SUMMARY_ELEMS = 1_000_000


@dataclass
class CompareResult:
    name: str
    passed: bool
    max_abs: float
    max_rel: float
    mean_abs: float
    atol: float
    rtol: float
    actual_shape: tuple[int, ...] = ()
    expected_shape: tuple[int, ...] = ()
    numel: int = 0
    max_abs_index: tuple[int, ...] = ()
    max_abs_actual: float = 0.0
    max_abs_expected: float = 0.0
    max_rel_index: tuple[int, ...] = ()
    max_rel_actual: float = 0.0
    max_rel_expected: float = 0.0
    per_head_max_abs: Optional[list[float]] = None


def _unravel_index(flat_index: int, shape: torch.Size | tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        return ()
    coords = []
    for size in reversed(tuple(shape)):
        coords.append(flat_index % int(size))
        flat_index //= int(size)
    return tuple(reversed(coords))


def _tensor_value(tensor: torch.Tensor, index: tuple[int, ...]) -> float:
    if not index:
        return float(tensor.reshape(-1)[0].item()) if tensor.numel() else 0.0
    return float(tensor[index].item())


def _tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": int(tensor.numel()),
    }
    if tensor.numel() == 0:
        return summary

    detached = tensor.detach()
    values_for_stats = detached.reshape(-1)
    if values_for_stats.numel() > MAX_SUMMARY_ELEMS:
        step = math.ceil(values_for_stats.numel() / MAX_SUMMARY_ELEMS)
        values_for_stats = values_for_stats[::step][:MAX_SUMMARY_ELEMS]
        summary["stats_sampled"] = True
        summary["stats_sample_elems"] = int(values_for_stats.numel())

    if detached.is_floating_point():
        values = values_for_stats.to(torch.float32)
        finite = torch.isfinite(values)
        summary["nan_count"] = int(torch.isnan(values).sum().item())
        summary["inf_count"] = int(torch.isinf(values).sum().item())
        if bool(finite.any().item()):
            finite_values = values[finite]
            summary["min"] = float(finite_values.min().item())
            summary["max"] = float(finite_values.max().item())
            summary["mean"] = float(finite_values.mean().item())
            summary["std"] = (
                float(finite_values.std(unbiased=False).item())
                if finite_values.numel() > 1
                else 0.0
            )
    elif detached.dtype == torch.bool:
        summary["true_count"] = int(values_for_stats.sum().item())
    else:
        summary["min"] = int(values_for_stats.min().item())
        summary["max"] = int(values_for_stats.max().item())
    return summary


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Size):
        return list(value)
    if isinstance(value, torch.dtype):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _as_bhk(q: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    if q.ndim < 3:
        raise ValueError(f"q must have at least 3 dims [..., heads, dim], got {q.shape}")
    original_shape = q.shape
    return q.reshape(-1, q.shape[-2], q.shape[-1]), original_shape


def _valid_mask(lengths: torch.Tensor, width: int) -> torch.Tensor:
    lengths = lengths.reshape(-1).to(torch.long)
    arange = torch.arange(width, device=lengths.device)
    return arange.unsqueeze(0) < lengths.unsqueeze(1)


def _canonical_indices(indices: torch.Tensor) -> torch.Tensor:
    if indices.ndim < 2:
        raise ValueError(f"indices must have at least 2 dims, got {indices.shape}")
    return indices.reshape(-1, indices.shape[-1]).to(torch.long)


def _select_cache_rows(
    cache: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    page_size: int,
    *,
    layout: str,
    fp8_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return selected KV rows as fp32 `[B, K, D]` plus valid mask `[B, K]`."""
    flat_indices = _canonical_indices(indices)
    lengths = lengths.reshape(-1).to(torch.long)
    if flat_indices.shape[0] != lengths.shape[0]:
        raise ValueError(
            f"indices batch {flat_indices.shape[0]} != lengths batch {lengths.shape[0]}"
        )

    max_k = flat_indices.shape[-1]
    valid = _valid_mask(lengths, max_k)
    safe_indices = flat_indices.masked_fill(~valid, 0).clamp_min(0)
    page_indices = safe_indices // page_size
    token_offsets = safe_indices % page_size

    if layout == "dense":
        if cache.ndim == 4 and cache.shape[-2] == 1:
            cache = cache[..., 0, :]
        if cache.ndim != 3:
            raise ValueError(
                "dense cache must have shape [pages, page, dim] or "
                f"[pages, page, 1, dim], got {cache.shape}"
            )
        rows = cache[page_indices, token_offsets].to(torch.float32)
    elif layout == "dsv4_packed":
        if cache.ndim != 4 or cache.shape[-2] != 1:
            raise ValueError(
                "dsv4_packed cache must have shape [pages, page, 1, bytes], "
                f"got {cache.shape}"
            )
        packed = cache[page_indices, token_offsets, 0].contiguous()
        if packed.shape[-1] < DS4_PACKED_BYTES:
            raise ValueError(
                f"packed dim {packed.shape[-1]} is smaller than {DS4_PACKED_BYTES}"
            )

        nope_q = packed[..., :DS4_NOPE_DIM].contiguous().view(fp8_dtype)
        nope = nope_q.to(torch.float32)

        rope_lo = DS4_NOPE_DIM
        rope_hi = rope_lo + DS4_ROPE_DIM * 2
        rope = packed[..., rope_lo:rope_hi].contiguous().view(torch.bfloat16)
        rope = rope.to(torch.float32)

        scales = packed[..., rope_hi : rope_hi + DS4_SCALE_DIM].to(torch.float32)
        scales = torch.exp2(scales - 127.0).repeat_interleave(
            DS4_SCALE_GROUP, dim=-1
        )
        rows = torch.cat([nope * scales, rope], dim=-1)
    else:
        raise ValueError(f"unknown layout {layout!r}")

    rows = rows.masked_fill(~valid.unsqueeze(-1), 0.0)
    return rows, valid


def torch_ds4_sparse_scores_ref(
    *,
    q: torch.Tensor,
    swa_k_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_topk_lengths: torch.Tensor,
    swa_page_size: int,
    softmax_scale: float,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_lengths: Optional[torch.Tensor] = None,
    extra_page_size: Optional[int] = None,
    layout: str = "dsv4_packed",
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    kv_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Reference pre-softmax DS4 QK scores over SWA + compressed rows."""
    q_flat, _ = _as_bhk(q)
    q_flat = q_flat.to(torch.float32)

    swa_rows, swa_valid = _select_cache_rows(
        swa_k_cache,
        swa_indices,
        swa_topk_lengths,
        swa_page_size,
        layout=layout,
        fp8_dtype=fp8_dtype,
    )

    row_parts = [swa_rows]
    valid_parts = [swa_valid]

    if extra_k_cache is not None:
        if extra_indices is None or extra_topk_lengths is None or extra_page_size is None:
            raise ValueError(
                "extra_k_cache requires extra_indices, extra_topk_lengths, "
                "and extra_page_size"
            )
        extra_rows, extra_valid = _select_cache_rows(
            extra_k_cache,
            extra_indices,
            extra_topk_lengths,
            extra_page_size,
            layout=layout,
            fp8_dtype=fp8_dtype,
        )
        row_parts.append(extra_rows)
        valid_parts.append(extra_valid)

    kv = torch.cat(row_parts, dim=1)
    valid = torch.cat(valid_parts, dim=1)
    if kv_dtype is not None:
        kv = kv.to(kv_dtype).to(torch.float32)

    if q_flat.shape[0] != kv.shape[0]:
        raise ValueError(f"q batch {q_flat.shape[0]} != kv batch {kv.shape[0]}")
    if q_flat.shape[-1] != kv.shape[-1]:
        raise ValueError(f"q dim {q_flat.shape[-1]} != kv dim {kv.shape[-1]}")

    scores = torch.einsum("bhd,bkd->bhk", q_flat, kv) * softmax_scale
    return scores.masked_fill(~valid[:, None, :], 0.0)


def torch_ds4_sparse_attention_ref(
    *,
    q: torch.Tensor,
    swa_k_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_topk_lengths: torch.Tensor,
    swa_page_size: int,
    softmax_scale: float,
    attn_sink: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_lengths: Optional[torch.Tensor] = None,
    extra_page_size: Optional[int] = None,
    layout: str = "dsv4_packed",
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    """Reference DS4 sparse attention using only ordinary Torch ops."""
    q_flat, original_shape = _as_bhk(q)
    q_flat = q_flat.to(torch.float32)

    swa_rows, swa_valid = _select_cache_rows(
        swa_k_cache,
        swa_indices,
        swa_topk_lengths,
        swa_page_size,
        layout=layout,
        fp8_dtype=fp8_dtype,
    )

    row_parts = [swa_rows]
    valid_parts = [swa_valid]

    if extra_k_cache is not None:
        if extra_indices is None or extra_topk_lengths is None or extra_page_size is None:
            raise ValueError(
                "extra_k_cache requires extra_indices, extra_topk_lengths, "
                "and extra_page_size"
            )
        extra_rows, extra_valid = _select_cache_rows(
            extra_k_cache,
            extra_indices,
            extra_topk_lengths,
            extra_page_size,
            layout=layout,
            fp8_dtype=fp8_dtype,
        )
        row_parts.append(extra_rows)
        valid_parts.append(extra_valid)

    kv = torch.cat(row_parts, dim=1)
    valid = torch.cat(valid_parts, dim=1)

    if q_flat.shape[0] != kv.shape[0]:
        raise ValueError(f"q batch {q_flat.shape[0]} != kv batch {kv.shape[0]}")
    if q_flat.shape[-1] != kv.shape[-1]:
        raise ValueError(f"q dim {q_flat.shape[-1]} != kv dim {kv.shape[-1]}")

    scores = torch.einsum("bhd,bkd->bhk", q_flat, kv) * softmax_scale
    scores = scores.masked_fill(~valid[:, None, :], -torch.inf)

    if attn_sink is not None:
        sink = attn_sink.to(device=scores.device, dtype=torch.float32)
        sink = sink.reshape(1, -1, 1)
        if sink.shape[1] != scores.shape[1]:
            raise ValueError(f"sink heads {sink.shape[1]} != q heads {scores.shape[1]}")
        scores = torch.cat([scores, sink.expand(scores.shape[0], -1, -1)], dim=-1)
        zero_value = torch.zeros(
            kv.shape[0], 1, kv.shape[-1], device=kv.device, dtype=kv.dtype
        )
        kv = torch.cat([kv, zero_value], dim=1)

    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhk,bkd->bhd", probs, kv)
    return out.reshape(*original_shape[:-2], original_shape[-2], kv.shape[-1])


def compare_tensors(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> CompareResult:
    actual = actual.detach().to(torch.float32)
    expected = expected.detach().to(torch.float32)
    actual_shape = tuple(actual.shape)
    expected_shape = tuple(expected.shape)
    if actual_shape != expected_shape:
        return CompareResult(
            name=name,
            passed=False,
            max_abs=math.inf,
            max_rel=math.inf,
            mean_abs=math.inf,
            atol=atol,
            rtol=rtol,
            actual_shape=actual_shape,
            expected_shape=expected_shape,
            numel=0,
        )

    diff = (actual - expected).abs()
    rel = diff / expected.abs().clamp_min(1e-12)
    passed = bool(torch.allclose(actual, expected, atol=atol, rtol=rtol))
    max_abs_flat = int(diff.reshape(-1).argmax().item()) if diff.numel() else 0
    max_rel_flat = int(rel.reshape(-1).argmax().item()) if rel.numel() else 0
    max_abs_index = _unravel_index(max_abs_flat, diff.shape)
    max_rel_index = _unravel_index(max_rel_flat, rel.shape)
    per_head_max_abs = None
    if diff.ndim >= 2 and diff.numel():
        head_diff = diff.reshape(-1, diff.shape[-2], diff.shape[-1])
        per_head_max_abs = [
            float(value) for value in head_diff.amax(dim=(0, 2)).detach().cpu().tolist()
        ]
    return CompareResult(
        name=name,
        passed=passed,
        max_abs=float(diff.max().item()),
        max_rel=float(rel.max().item()),
        mean_abs=float(diff.mean().item()),
        atol=atol,
        rtol=rtol,
        actual_shape=actual_shape,
        expected_shape=expected_shape,
        numel=int(diff.numel()),
        max_abs_index=max_abs_index,
        max_abs_actual=_tensor_value(actual, max_abs_index),
        max_abs_expected=_tensor_value(expected, max_abs_index),
        max_rel_index=max_rel_index,
        max_rel_actual=_tensor_value(actual, max_rel_index),
        max_rel_expected=_tensor_value(expected, max_rel_index),
        per_head_max_abs=per_head_max_abs,
    )


def _print_result(result: CompareResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    print(
        f"{status} {result.name}: "
        f"shape={list(result.actual_shape)} "
        f"max_abs={result.max_abs:.6g} "
        f"max_rel={result.max_rel:.6g} "
        f"mean_abs={result.mean_abs:.6g} "
        f"atol={result.atol:g} rtol={result.rtol:g}"
    )
    if not result.passed:
        print(
            "  worst_abs="
            f"{list(result.max_abs_index)} actual={result.max_abs_actual:.8g} "
            f"expected={result.max_abs_expected:.8g}; "
            "worst_rel="
            f"{list(result.max_rel_index)} actual={result.max_rel_actual:.8g} "
            f"expected={result.max_rel_expected:.8g}"
        )
        if result.per_head_max_abs:
            preview = ", ".join(f"{value:.4g}" for value in result.per_head_max_abs[:8])
            suffix = " ..." if len(result.per_head_max_abs) > 8 else ""
            print(f"  per_head_max_abs=[{preview}{suffix}]")


def _make_dense_cache(rows: torch.Tensor, page_size: int) -> torch.Tensor:
    if rows.ndim != 2:
        raise ValueError(f"rows must be [tokens, dim], got {rows.shape}")
    num_pages = math.ceil(rows.shape[0] / page_size)
    cache = rows.new_zeros(num_pages, page_size, rows.shape[-1])
    cache.reshape(-1, rows.shape[-1])[: rows.shape[0]] = rows
    return cache


def _make_dsv4_packed_cache(rows: torch.Tensor, page_size: int) -> torch.Tensor:
    if rows.shape[-1] != DS4_HEAD_DIM:
        raise ValueError(f"packed DS4 rows need dim {DS4_HEAD_DIM}, got {rows.shape}")
    num_pages = math.ceil(rows.shape[0] / page_size)
    cache = torch.zeros(num_pages, page_size, 1, DS4_PACKED_BYTES, dtype=torch.uint8)
    flat = cache.view(-1, 1, DS4_PACKED_BYTES)[: rows.shape[0], 0]

    nope = rows[:, :DS4_NOPE_DIM].contiguous().to(torch.float8_e4m3fn)
    rope = rows[:, DS4_NOPE_DIM:].contiguous().to(torch.bfloat16)
    flat[:, :DS4_NOPE_DIM] = nope.view(torch.uint8)
    flat[:, DS4_NOPE_DIM : DS4_NOPE_DIM + DS4_ROPE_DIM * 2] = rope.view(torch.uint8)
    flat[
        :,
        DS4_NOPE_DIM
        + DS4_ROPE_DIM * 2 : DS4_NOPE_DIM
        + DS4_ROPE_DIM * 2
        + DS4_SCALE_DIM,
    ] = 127
    return cache


def _manual_dense_reference(
    q: torch.Tensor,
    rows: torch.Tensor,
    valid: torch.Tensor,
    softmax_scale: float,
    attn_sink: Optional[torch.Tensor],
) -> torch.Tensor:
    q = q.to(torch.float32)
    rows = rows.to(torch.float32)
    scores = torch.einsum("bhd,bkd->bhk", q, rows) * softmax_scale
    scores = scores.masked_fill(~valid[:, None, :], -torch.inf)
    if attn_sink is not None:
        sink = attn_sink.to(torch.float32).reshape(1, -1, 1)
        scores = torch.cat([scores, sink.expand(scores.shape[0], -1, -1)], dim=-1)
        rows = torch.cat([rows, torch.zeros(rows.shape[0], 1, rows.shape[-1])], dim=1)
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bhk,bkd->bhd", probs, rows)


def _slow_loop_reference(
    q: torch.Tensor,
    rows: torch.Tensor,
    valid: torch.Tensor,
    softmax_scale: float,
    attn_sink: Optional[torch.Tensor],
) -> torch.Tensor:
    """Independent Python-loop reference for tiny tests.

    This intentionally avoids einsum, vectorized masking, and torch.softmax so
    it can catch broadcasting or shape bugs in the main Torch reference.
    """
    q = q.detach().cpu().to(torch.float64)
    rows = rows.detach().cpu().to(torch.float64)
    valid = valid.detach().cpu().to(torch.bool)
    sink = None if attn_sink is None else attn_sink.detach().cpu().to(torch.float64)

    batch, heads, dim = q.shape
    out = torch.zeros(batch, heads, dim, dtype=torch.float64)
    for b in range(batch):
        for h in range(heads):
            terms: list[tuple[float, Optional[int]]] = []
            for k in range(rows.shape[1]):
                if not bool(valid[b, k].item()):
                    continue
                score = 0.0
                for d in range(dim):
                    score += float(q[b, h, d].item() * rows[b, k, d].item())
                terms.append((score * softmax_scale, k))

            if sink is not None:
                terms.append((float(sink[h].item()), None))

            if not terms:
                raise ValueError("slow reference needs at least one attention term")

            max_score = max(score for score, _ in terms)
            denom = sum(math.exp(score - max_score) for score, _ in terms)
            for score, k in terms:
                if k is None:
                    continue
                weight = math.exp(score - max_score) / denom
                for d in range(dim):
                    out[b, h, d] += weight * rows[b, k, d]

    return out.to(q.dtype)


def _random_indices(width: int, tokens: int, device: torch.device) -> torch.Tensor:
    perm = torch.randperm(tokens, device=device)
    if width <= tokens:
        return perm[:width]
    pad = torch.full((width - tokens,), -1, device=device, dtype=perm.dtype)
    return torch.cat([perm, pad])


def _run_dense_fuzz_tests(
    device: str,
    fuzz_iters: int,
    *,
    seed: int,
) -> CompareResult:
    torch.manual_seed(seed)
    dev = torch.device(device)
    max_abs = 0.0
    max_rel = 0.0
    mean_abs_total = 0.0

    for _ in range(fuzz_iters):
        batch = int(torch.randint(1, 4, ()).item())
        heads = int(torch.randint(1, 5, ()).item())
        dim = int(torch.randint(1, 17, ()).item())
        swa_tokens = int(torch.randint(2, 9, ()).item())
        extra_tokens = int(torch.randint(1, 7, ()).item())
        swa_width = swa_tokens + int(torch.randint(0, 4, ()).item())
        extra_width = extra_tokens + int(torch.randint(0, 4, ()).item())
        swa_page_size = int(torch.randint(2, 5, ()).item())
        extra_page_size = int(torch.randint(2, 5, ()).item())

        q = torch.randn(batch, heads, dim, device=dev) * 0.5
        swa_source = torch.randn(swa_tokens, dim, device=dev) * 0.5
        extra_source = torch.randn(extra_tokens, dim, device=dev) * 0.5
        swa_cache = _make_dense_cache(swa_source.cpu(), swa_page_size).to(dev)
        extra_cache = _make_dense_cache(extra_source.cpu(), extra_page_size).to(dev)

        swa_indices = []
        extra_indices = []
        swa_lengths = []
        extra_lengths = []
        for _batch in range(batch):
            swa_len = int(torch.randint(1, swa_tokens + 1, ()).item())
            extra_len = int(torch.randint(1, extra_tokens + 1, ()).item())
            swa_lengths.append(swa_len)
            extra_lengths.append(extra_len)
            swa_indices.append(_random_indices(swa_width, swa_tokens, dev))
            extra_indices.append(_random_indices(extra_width, extra_tokens, dev))

        swa_indices_t = torch.stack(swa_indices)
        extra_indices_t = torch.stack(extra_indices)
        swa_lengths_t = torch.tensor(swa_lengths, device=dev)
        extra_lengths_t = torch.tensor(extra_lengths, device=dev)
        sink = torch.randn(heads, device=dev) * 0.25 if bool(torch.randint(0, 2, ())) else None
        softmax_scale = dim**-0.5

        actual = torch_ds4_sparse_attention_ref(
            q=q,
            swa_k_cache=swa_cache,
            swa_indices=swa_indices_t,
            swa_topk_lengths=swa_lengths_t,
            swa_page_size=swa_page_size,
            extra_k_cache=extra_cache,
            extra_indices=extra_indices_t,
            extra_topk_lengths=extra_lengths_t,
            extra_page_size=extra_page_size,
            softmax_scale=softmax_scale,
            attn_sink=sink,
            layout="dense",
        )
        swa_rows, swa_valid = _select_cache_rows(
            swa_cache,
            swa_indices_t,
            swa_lengths_t,
            swa_page_size,
            layout="dense",
            fp8_dtype=torch.float8_e4m3fn,
        )
        extra_rows, extra_valid = _select_cache_rows(
            extra_cache,
            extra_indices_t,
            extra_lengths_t,
            extra_page_size,
            layout="dense",
            fp8_dtype=torch.float8_e4m3fn,
        )
        expected = _slow_loop_reference(
            q,
            torch.cat([swa_rows, extra_rows], dim=1),
            torch.cat([swa_valid, extra_valid], dim=1),
            softmax_scale,
            sink,
        ).to(dev)

        diff = (actual.detach().to(torch.float32) - expected.to(torch.float32)).abs()
        rel = diff / expected.to(torch.float32).abs().clamp_min(1e-12)
        max_abs = max(max_abs, float(diff.max().item()))
        max_rel = max(max_rel, float(rel.max().item()))
        mean_abs_total += float(diff.mean().item())

        expected_for_compare = expected.to(device=actual.device, dtype=actual.dtype)
        if not torch.allclose(actual, expected_for_compare, atol=1e-6, rtol=1e-5):
            return CompareResult(
                name="dense_fuzz_vs_slow_loop",
                passed=False,
                max_abs=max_abs,
                max_rel=max_rel,
                mean_abs=mean_abs_total / (_ + 1),
                atol=1e-6,
                rtol=1e-5,
            )

    return CompareResult(
        name=f"dense_fuzz_vs_slow_loop_{fuzz_iters}x",
        passed=True,
        max_abs=max_abs,
        max_rel=max_rel,
        mean_abs=mean_abs_total / max(fuzz_iters, 1),
        atol=1e-6,
        rtol=1e-5,
    )


def run_synthetic_tests(device: str, fuzz_iters: int) -> list[CompareResult]:
    torch.manual_seed(0)
    dev = torch.device(device)
    results: list[CompareResult] = []

    q = torch.tensor([[[0.2, -0.4, 0.8, 0.1]]], device=dev)
    rows = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=dev,
    )
    cache = _make_dense_cache(rows.cpu(), page_size=2).to(dev)
    indices = torch.tensor([[0, 1, 2, 3]], device=dev)
    lengths = torch.tensor([4], device=dev)
    sink = torch.tensor([0.25], device=dev)
    expected = _manual_dense_reference(
        q, rows.unsqueeze(0), torch.ones(1, 4, dtype=torch.bool, device=dev), 0.5, sink
    )
    actual = torch_ds4_sparse_attention_ref(
        q=q,
        swa_k_cache=cache,
        swa_indices=indices,
        swa_topk_lengths=lengths,
        swa_page_size=2,
        softmax_scale=0.5,
        attn_sink=sink,
        layout="dense",
    )
    results.append(compare_tensors("dense_swa_with_sink", actual, expected, atol=1e-6, rtol=1e-5))

    swa_rows = rows[:2]
    extra_rows = rows[2:]
    swa_cache = _make_dense_cache(swa_rows.cpu(), page_size=2).to(dev)
    extra_cache = _make_dense_cache(extra_rows.cpu(), page_size=2).to(dev)
    actual = torch_ds4_sparse_attention_ref(
        q=q,
        swa_k_cache=swa_cache,
        swa_indices=torch.tensor([[0, 1]], device=dev),
        swa_topk_lengths=torch.tensor([2], device=dev),
        swa_page_size=2,
        extra_k_cache=extra_cache,
        extra_indices=torch.tensor([[0, 1]], device=dev),
        extra_topk_lengths=torch.tensor([2], device=dev),
        extra_page_size=2,
        softmax_scale=0.5,
        attn_sink=sink,
        layout="dense",
    )
    results.append(compare_tensors("dense_swa_plus_extra", actual, expected, atol=1e-6, rtol=1e-5))

    bad = (
        _manual_dense_reference(
            q,
            swa_rows.unsqueeze(0),
            torch.ones(1, 2, dtype=torch.bool, device=dev),
            0.5,
            None,
        )
        + _manual_dense_reference(
            q,
            extra_rows.unsqueeze(0),
            torch.ones(1, 2, dtype=torch.bool, device=dev),
            0.5,
            None,
        )
    )
    if torch.allclose(bad, expected, atol=1e-3, rtol=1e-3):
        results.append(CompareResult("separate_softmax_trap", False, 0.0, 0.0, 0.0, 1e-3, 1e-3))
    else:
        results.append(CompareResult("separate_softmax_trap", True, 0.0, 0.0, 0.0, 1e-3, 1e-3))

    packed_rows = torch.randn(6, DS4_HEAD_DIM) * 0.125
    packed_cache = _make_dsv4_packed_cache(packed_rows, page_size=4).to(dev)
    packed_q = torch.randn(1, 2, DS4_HEAD_DIM, device=dev) * 0.125
    packed_indices = torch.tensor([[0, 1, 2, 3, 4, 5]], device=dev)
    packed_lengths = torch.tensor([6], device=dev)
    actual = torch_ds4_sparse_attention_ref(
        q=packed_q,
        swa_k_cache=packed_cache,
        swa_indices=packed_indices,
        swa_topk_lengths=packed_lengths,
        swa_page_size=4,
        softmax_scale=DS4_HEAD_DIM**-0.5,
        attn_sink=torch.zeros(2, device=dev),
        layout="dsv4_packed",
    )
    dequant_rows, valid = _select_cache_rows(
        packed_cache,
        packed_indices,
        packed_lengths,
        4,
        layout="dsv4_packed",
        fp8_dtype=torch.float8_e4m3fn,
    )
    expected = _manual_dense_reference(
        packed_q, dequant_rows, valid, DS4_HEAD_DIM**-0.5, torch.zeros(2, device=dev)
    )
    results.append(compare_tensors("dsv4_packed_dequant", actual, expected, atol=1e-6, rtol=1e-5))

    results.append(_run_dense_fuzz_tests(device, fuzz_iters, seed=1))

    for result in results:
        _print_result(result)
    return results


def _get_optional(fixture: dict, *names: str):
    for name in names:
        if name in fixture:
            return fixture[name]
    return None


def _int_value(value: Any, name: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be scalar-like, got {value.shape}")
        return int(value.item())
    return int(value)


def _validate_index_block(
    fixture: dict[str, Any],
    *,
    prefix: str,
    batch: int,
    page_size_name: str,
) -> list[str]:
    warnings = []
    indices = fixture[f"{prefix}_indices"]
    lengths = fixture[f"{prefix}_topk_lengths"]
    page_size = _int_value(fixture[page_size_name], page_size_name)
    if page_size <= 0:
        warnings.append(f"{page_size_name} is non-positive: {page_size}")

    flat_indices = _canonical_indices(indices)
    flat_lengths = lengths.reshape(-1)
    if flat_indices.shape[0] != batch:
        warnings.append(
            f"{prefix}_indices batch {flat_indices.shape[0]} does not match q batch {batch}"
        )
    if flat_lengths.numel() != flat_indices.shape[0]:
        warnings.append(
            f"{prefix}_topk_lengths count {flat_lengths.numel()} does not match "
            f"{prefix}_indices batch {flat_indices.shape[0]}"
        )
    if flat_lengths.numel() and int(flat_lengths.max().item()) > flat_indices.shape[-1]:
        warnings.append(
            f"{prefix}_topk_lengths max exceeds index width {flat_indices.shape[-1]}"
        )
    return warnings


def _validate_fixture(path: Path, fixture: dict[str, Any]) -> list[str]:
    required = [
        "q",
        "swa_k_cache",
        "swa_indices",
        "swa_topk_lengths",
        "swa_page_size",
    ]
    missing = [name for name in required if name not in fixture]
    if missing:
        raise KeyError(f"{path} is missing required fixture keys: {missing}")

    warnings = []
    layout = str(fixture.get("layout", "dsv4_packed"))
    q = fixture["q"]
    if not isinstance(q, torch.Tensor):
        raise TypeError(f"{path}: q must be a tensor")
    if q.ndim < 3:
        raise ValueError(f"{path}: q must have at least 3 dims, got {q.shape}")
    q_flat, _ = _as_bhk(q)
    batch = q_flat.shape[0]

    if layout == "dsv4_packed":
        for name in ("swa_k_cache", "extra_k_cache"):
            if name not in fixture or fixture[name] is None:
                continue
            cache = fixture[name]
            if cache.ndim != 4 or cache.shape[-2] != 1:
                warnings.append(f"{name} shape is not packed DS4 cache-like: {tuple(cache.shape)}")
            elif cache.shape[-1] < DS4_PACKED_BYTES:
                warnings.append(
                    f"{name} last dim {cache.shape[-1]} is smaller than {DS4_PACKED_BYTES}"
                )
    elif layout == "dense":
        pass
    else:
        warnings.append(f"unknown layout {layout!r}; oracle will raise if unsupported")

    warnings.extend(
        _validate_index_block(
            fixture, prefix="swa", batch=batch, page_size_name="swa_page_size"
        )
    )

    extra_keys = {"extra_k_cache", "extra_indices", "extra_topk_lengths", "extra_page_size"}
    present_extra = {name for name in extra_keys if fixture.get(name) is not None}
    if present_extra and present_extra != extra_keys:
        warnings.append(f"incomplete extra KV block: present={sorted(present_extra)}")
    elif present_extra == extra_keys:
        warnings.extend(
            _validate_index_block(
                fixture,
                prefix="extra",
                batch=batch,
                page_size_name="extra_page_size",
            )
        )

    expected = fixture.get("expected")
    if isinstance(expected, torch.Tensor) and tuple(expected.shape) != tuple(q.shape):
        warnings.append(
            f"expected shape {tuple(expected.shape)} does not match q/output shape {tuple(q.shape)}"
        )

    return warnings


def _fixture_summary(fixture: dict[str, Any]) -> dict[str, Any]:
    tensors = {
        key: _tensor_summary(value)
        for key, value in sorted(fixture.items())
        if isinstance(value, torch.Tensor)
    }
    metadata: dict[str, Any] = {}
    for key, value in sorted(fixture.items()):
        if isinstance(value, torch.Tensor):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            metadata[key] = value
        else:
            metadata[key] = repr(value)
    return {"metadata": metadata, "tensors": tensors}


def _print_fixture_summary(path: Path, fixture: dict[str, Any], warnings: list[str]) -> None:
    layout = str(fixture.get("layout", "dsv4_packed"))
    q = fixture["q"]
    expected = fixture.get("expected")
    layer_id = fixture.get("layer_id", "?")
    compress_ratio = fixture.get("compress_ratio", "?")
    print(
        f"fixture {path.name}: layout={layout} layer={layer_id} "
        f"compress_ratio={compress_ratio} q={tuple(q.shape)} dtype={q.dtype}"
    )
    if isinstance(expected, torch.Tensor):
        print(f"  expected={tuple(expected.shape)} dtype={expected.dtype}")
    else:
        print("  expected=<missing>")
    for warning in warnings:
        print(f"  warning: {warning}")


def run_fixture(
    path: Path,
    device: str,
    atol: float,
    rtol: float,
    *,
    print_summary: bool,
) -> dict[str, Any]:
    start = time.perf_counter()
    fixture = torch.load(path, map_location=device)
    warnings = _validate_fixture(path, fixture)
    if print_summary:
        _print_fixture_summary(path, fixture, warnings)

    softmax_scale = float(fixture.get("softmax_scale", DS4_HEAD_DIM**-0.5))
    layout = str(fixture.get("layout", "dsv4_packed"))
    output = torch_ds4_sparse_attention_ref(
        q=fixture["q"],
        swa_k_cache=fixture["swa_k_cache"],
        swa_indices=fixture["swa_indices"],
        swa_topk_lengths=fixture["swa_topk_lengths"],
        swa_page_size=int(fixture["swa_page_size"]),
        softmax_scale=softmax_scale,
        attn_sink=_get_optional(fixture, "attn_sink"),
        extra_k_cache=_get_optional(fixture, "extra_k_cache"),
        extra_indices=_get_optional(fixture, "extra_indices"),
        extra_topk_lengths=_get_optional(fixture, "extra_topk_lengths"),
        extra_page_size=(
            _int_value(fixture["extra_page_size"], "extra_page_size")
            if fixture.get("extra_page_size") is not None
            else None
        ),
        layout=layout,
    )
    report: dict[str, Any] = {
        "path": str(path),
        "passed": True,
        "elapsed_sec": time.perf_counter() - start,
        "layout": layout,
        "warnings": warnings,
        "fixture": _fixture_summary(fixture),
        "output": _tensor_summary(output),
    }
    if "expected" not in fixture:
        print(f"fixture output shape: {tuple(output.shape)}")
        return report

    result = compare_tensors(
        f"fixture:{path.name}", output, fixture["expected"], atol=atol, rtol=rtol
    )
    _print_result(result)
    report["passed"] = result.passed
    report["result"] = asdict(result)
    return report


def _collect_fixture_paths(
    fixtures: Optional[list[Path]],
    fixture_dirs: Optional[list[Path]],
) -> list[Path]:
    paths: list[Path] = []
    if fixtures:
        paths.extend(fixtures)
    if fixture_dirs:
        for directory in fixture_dirs:
            paths.extend(sorted(directory.glob("*.pt")))

    deduped = []
    seen = set()
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def run_fixtures(
    paths: list[Path],
    device: str,
    atol: float,
    rtol: float,
    *,
    print_summary: bool,
) -> list[dict[str, Any]]:
    reports = []
    for path in paths:
        try:
            reports.append(
                run_fixture(
                    path,
                    device,
                    atol,
                    rtol,
                    print_summary=print_summary,
                )
            )
        except Exception as exc:
            print(f"FAIL fixture:{path.name}: {type(exc).__name__}: {exc}")
            reports.append(
                {
                    "path": str(path),
                    "passed": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    return reports


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n"
    )
    print(f"wrote report: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="torch device, e.g. cpu or cuda")
    parser.add_argument(
        "--fixture",
        type=Path,
        action="append",
        help="optional .pt fixture to evaluate; may be passed more than once",
    )
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        action="append",
        help="directory of .pt fixtures to evaluate; may be passed more than once",
    )
    parser.add_argument("--fuzz-iters", type=int, default=100)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--report-json", type=Path, help="write a JSON run report")
    parser.add_argument(
        "--no-fixture-summary",
        action="store_true",
        help="suppress per-fixture shape/dtype summaries",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fixture_paths = _collect_fixture_paths(args.fixture, args.fixture_dir)
    if fixture_paths:
        reports = run_fixtures(
            fixture_paths,
            args.device,
            args.atol,
            args.rtol,
            print_summary=not args.no_fixture_summary,
        )
        passed = all(bool(report.get("passed")) for report in reports)
        report = {
            "mode": "fixtures",
            "passed": passed,
            "device": args.device,
            "atol": args.atol,
            "rtol": args.rtol,
            "fixture_count": len(reports),
            "fixtures": reports,
        }
        if args.report_json:
            _write_report(args.report_json, report)
        return 0 if passed else 1

    results = run_synthetic_tests(args.device, args.fuzz_iters)
    passed = all(result.passed for result in results)
    report = {
        "mode": "synthetic",
        "passed": passed,
        "device": args.device,
        "fuzz_iters": args.fuzz_iters,
        "results": [asdict(result) for result in results],
    }
    if args.report_json:
        _write_report(args.report_json, report)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
