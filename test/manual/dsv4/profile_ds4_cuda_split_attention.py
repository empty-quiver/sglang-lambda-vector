"""Profile split-row DS4 CUDA attention partial and reduce kernels.

Run from the repository root:

    uv run --with torch --no-project python \
      test/manual/dsv4/profile_ds4_cuda_split_attention.py

The C++ extension emits one DSV4_CUDA_SPLIT_PROFILE line per profiled sample
when DSV4_CUDA_REF_PROFILE_SPLIT is enabled. This script enables that flag only
for a small number of samples, then disables it for the normal total-time
benchmark.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

try:
    from .ds4_cuda_reference_attention import ds4_cuda_sparse_attention_from_fixture
    from .ds4_synthetic_attention_fixtures import DEFAULT_LONG_SPECS
    from .ds4_synthetic_attention_fixtures import (
        make_synthetic_dsv4_attention_fixture,
    )
except ImportError:
    from ds4_cuda_reference_attention import ds4_cuda_sparse_attention_from_fixture  # type: ignore
    from ds4_synthetic_attention_fixtures import DEFAULT_LONG_SPECS  # type: ignore
    from ds4_synthetic_attention_fixtures import (  # type: ignore
        make_synthetic_dsv4_attention_fixture,
    )


def _fixture_to_cuda(fixture: dict[str, Any]) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in fixture.items():
        if torch.is_tensor(value):
            moved[key] = value.cuda().contiguous()
        else:
            moved[key] = value
    return moved


def _set_profile_enabled(enabled: bool) -> None:
    os.environ["DSV4_CUDA_REF_PROFILE_SPLIT"] = "1" if enabled else "0"


def _variant_list() -> list[str]:
    raw = os.environ.get("DSV4_CUDA_PROFILE_VARIANTS", "v15")
    return [item.strip() for item in raw.split(",") if item.strip()]


def _spec_names() -> set[str] | None:
    raw = os.environ.get("DSV4_CUDA_PROFILE_SPECS", "")
    if not raw:
        return None
    return {item.strip() for item in raw.split(",") if item.strip()}


def _benchmark_total(
    fixture: dict[str, Any],
    *,
    variant: str,
    warmup: int,
    iters: int,
) -> float:
    _set_profile_enabled(False)
    for _ in range(warmup):
        ds4_cuda_sparse_attention_from_fixture(fixture, optimized=variant)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        ds4_cuda_sparse_attention_from_fixture(fixture, optimized=variant)
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / max(iters, 1)


def _warmup(
    fixture: dict[str, Any],
    *,
    variant: str,
    warmup: int,
) -> None:
    _set_profile_enabled(False)
    for _ in range(warmup):
        ds4_cuda_sparse_attention_from_fixture(fixture, optimized=variant)
    torch.cuda.synchronize()


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    variants = _variant_list()
    selected_specs = _spec_names()
    warmup = int(os.environ.get("DSV4_CUDA_PROFILE_WARMUP", "20"))
    iters = int(os.environ.get("DSV4_CUDA_PROFILE_ITERS", "200"))
    profile_samples = int(os.environ.get("DSV4_CUDA_PROFILE_SAMPLES", "3"))

    for spec in DEFAULT_LONG_SPECS:
        if selected_specs is not None and spec.name not in selected_specs:
            continue
        fixture = _fixture_to_cuda(make_synthetic_dsv4_attention_fixture(spec))
        for variant in variants:
            _warmup(fixture, variant=variant, warmup=warmup)
            for sample in range(profile_samples):
                _set_profile_enabled(True)
                ds4_cuda_sparse_attention_from_fixture(fixture, optimized=variant)
                torch.cuda.synchronize()
                print(
                    f"PROFILE_SAMPLE spec={spec.name} variant={variant} "
                    f"sample={sample + 1}/{profile_samples}"
                )

            total_ms = _benchmark_total(
                fixture,
                variant=variant,
                warmup=warmup,
                iters=iters,
            )
            print(
                f"PROFILE_TOTAL spec={spec.name} variant={variant} "
                f"avg_ms={total_ms:.6f} warmup={warmup} iters={iters}"
            )

    _set_profile_enabled(False)


if __name__ == "__main__":
    main()
