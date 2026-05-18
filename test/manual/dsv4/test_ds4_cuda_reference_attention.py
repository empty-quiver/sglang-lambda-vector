"""Manual DS4 CUDA debug-kernel checks against the Torch oracle.

This test intentionally requires captured fixtures. Run from the repository
root with:

    DSV4_TORCH_REF_FIXTURE_DIRS=/home/eve/ds4_ref_tests/fixtures_real_kt_20260516_013958_realmetadata \
      uv run --with torch --no-project python test/manual/dsv4/test_ds4_cuda_reference_attention.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import unittest

import torch

try:
    from .ds4_cuda_reference_attention import ds4_cuda_sparse_attention_from_fixture
    from .ds4_torch_reference_attention import (
        _collect_fixture_paths,
        compare_tensors,
        torch_ds4_sparse_attention_ref,
    )
except ImportError:
    from ds4_cuda_reference_attention import ds4_cuda_sparse_attention_from_fixture  # type: ignore
    from ds4_torch_reference_attention import (  # type: ignore
        _collect_fixture_paths,
        compare_tensors,
        torch_ds4_sparse_attention_ref,
    )


def _env_paths(name: str) -> list[Path]:
    value = os.environ.get(name, "")
    if not value:
        return []
    return [Path(item) for item in value.split(os.pathsep) if item]


def _load_fixture(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _torch_oracle(fixture: dict) -> torch.Tensor:
    return torch_ds4_sparse_attention_ref(
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
        layout=str(fixture.get("layout", "dsv4_packed")),
    )


class TestDS4CudaReferenceAttention(unittest.TestCase):
    def test_captured_fixtures_cuda(self):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")

        fixture_dirs = _env_paths("DSV4_TORCH_REF_FIXTURE_DIRS")
        fixture_paths = _env_paths("DSV4_TORCH_REF_FIXTURES")
        if not fixture_dirs and not fixture_paths:
            raise unittest.SkipTest(
                "set DSV4_TORCH_REF_FIXTURE_DIRS or DSV4_TORCH_REF_FIXTURES "
                "to replay captured DS4 attention fixtures"
            )

        paths = _collect_fixture_paths(fixture_paths, fixture_dirs)
        self.assertTrue(paths, "no .pt fixtures found for DS4 CUDA replay")

        atol = float(os.environ.get("DSV4_CUDA_REF_ATOL", "0.05"))
        rtol = float(os.environ.get("DSV4_CUDA_REF_RTOL", "0.05"))
        failures = []
        for path in paths:
            fixture = _load_fixture(path)
            actual = ds4_cuda_sparse_attention_from_fixture(fixture).cpu()
            oracle = _torch_oracle(fixture)
            cuda_vs_oracle = compare_tensors(
                f"cuda_vs_oracle:{path.name}",
                actual,
                oracle,
                atol=atol,
                rtol=rtol,
            )
            cuda_vs_expected = compare_tensors(
                f"cuda_vs_expected:{path.name}",
                actual,
                fixture["expected"],
                atol=atol,
                rtol=rtol,
            )
            for result in (cuda_vs_oracle, cuda_vs_expected):
                status = "PASS" if result.passed else "FAIL"
                print(
                    f"{status} {result.name}: shape={list(result.actual_shape)} "
                    f"max_abs={result.max_abs:.6g} max_rel={result.max_rel:.6g} "
                    f"mean_abs={result.mean_abs:.6g} atol={result.atol:g} rtol={result.rtol:g}"
                )
                if not result.passed:
                    failures.append(result)

        self.assertFalse(
            failures,
            "DS4 CUDA attention replay failed: "
            + json.dumps([result.__dict__ for result in failures], indent=2, default=str),
        )


if __name__ == "__main__":
    unittest.main()
