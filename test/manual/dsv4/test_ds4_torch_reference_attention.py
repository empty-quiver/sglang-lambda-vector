"""Manual regression wrapper for the DS4 Torch attention oracle.

Run the cheap synthetic checks from the repository root with:

    uv run --with torch --no-project python test/manual/dsv4/test_ds4_torch_reference_attention.py

Replay captured SGLang backend fixtures with:

    DSV4_TORCH_REF_FIXTURE_DIRS=/home/eve/ds4_ref_tests/fixtures_real_kt_20260516_013958_realmetadata \
      uv run --with torch --no-project python test/manual/dsv4/test_ds4_torch_reference_attention.py

This test is intentionally CPU-friendly. It validates the reference contract
that the future Ada kernel has to match; it does not launch SGLang.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import unittest

try:
    from .ds4_torch_reference_attention import (
        _collect_fixture_paths,
        run_fixtures,
        run_synthetic_tests,
    )
except ImportError:
    from ds4_torch_reference_attention import (  # type: ignore
        _collect_fixture_paths,
        run_fixtures,
        run_synthetic_tests,
    )


def _env_paths(name: str) -> list[Path]:
    value = os.environ.get(name, "")
    if not value:
        return []
    return [Path(item) for item in value.split(os.pathsep) if item]


class TestDS4TorchReferenceAttention(unittest.TestCase):
    def test_synthetic_cpu(self):
        fuzz_iters = int(os.environ.get("DSV4_TORCH_REF_FUZZ_ITERS", "20"))
        results = run_synthetic_tests("cpu", fuzz_iters)
        failed = [result for result in results if not result.passed]
        self.assertFalse(
            failed,
            "synthetic DS4 reference checks failed: "
            + json.dumps([result.name for result in failed]),
        )

    def test_captured_fixtures_cpu(self):
        fixture_dirs = _env_paths("DSV4_TORCH_REF_FIXTURE_DIRS")
        fixture_paths = _env_paths("DSV4_TORCH_REF_FIXTURES")
        if not fixture_dirs and not fixture_paths:
            raise unittest.SkipTest(
                "set DSV4_TORCH_REF_FIXTURE_DIRS or DSV4_TORCH_REF_FIXTURES "
                "to replay captured DS4 attention fixtures"
            )

        paths = _collect_fixture_paths(fixture_paths, fixture_dirs)
        self.assertTrue(paths, "no .pt fixtures found for DS4 reference replay")

        atol = float(os.environ.get("DSV4_TORCH_REF_ATOL", "0.05"))
        rtol = float(os.environ.get("DSV4_TORCH_REF_RTOL", "0.05"))
        reports = run_fixtures(
            paths,
            "cpu",
            atol,
            rtol,
            print_summary=bool(int(os.environ.get("DSV4_TORCH_REF_SUMMARY", "0"))),
        )
        failed = [report for report in reports if not report.get("passed")]
        self.assertFalse(
            failed,
            "captured DS4 fixture replay failed: "
            + json.dumps(failed, indent=2, sort_keys=True, default=str),
        )


if __name__ == "__main__":
    unittest.main()
