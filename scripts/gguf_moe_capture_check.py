#!/usr/bin/env python3
"""Validate GGUF MoE active-filter capture files.

This is intentionally narrow: it checks the current SGLang active-filter
behavior first, then optionally reruns the current GGUF MoE implementation if
the capture includes packed weights.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


def _load_capture(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _assert_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
):
    actual_f = actual.float()
    expected_f = expected.float()
    diff = (actual_f - expected_f).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
    close = torch.isclose(actual_f, expected_f, atol=atol, rtol=rtol)
    mismatch = int((~close).sum().item()) if close.numel() else 0
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    print(
        f"{name}: ok shape={tuple(actual.shape)} dtype={actual.dtype} "
        f"max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} "
        f"mismatch={mismatch}/{actual.numel()} atol={atol} rtol={rtol}",
    )


def _reconstruct_from_active(capture: dict, active_out: torch.Tensor) -> torch.Tensor:
    x = capture["x"]
    active_token_ids = capture["active_token_ids"].long()
    reconstructed = torch.zeros_like(x)
    reconstructed.index_add_(0, active_token_ids, active_out)
    return reconstructed


def _check_active_reconstruction(capture: dict, *, atol: float, rtol: float):
    reconstructed = _reconstruct_from_active(capture, capture["active_out"])
    expected = capture["out_hidden_states"]
    _assert_close(
        "active_filter_reconstruction",
        reconstructed,
        expected,
        atol=atol,
        rtol=rtol,
    )


def _check_current_kernel(
    capture: dict, device: str, *, atol: float, rtol: float
) -> torch.Tensor | None:
    missing = [key for key in ("w1", "w2") if key not in capture]
    if missing:
        print(
            "current_kernel: skipped; capture does not include "
            f"{', '.join(missing)}. Set SGLANG_GGUF_MOE_CAPTURE_WEIGHTS=1.",
        )
        return None

    repo_python = Path(__file__).resolve().parents[1] / "python"
    sys.path.insert(0, str(repo_python))

    from sglang.srt.layers.quantization.gguf import fused_moe_gguf

    active_x = capture["active_x"].to(device)
    active_weights = capture["active_weights"].to(device)
    active_topk_ids = capture["active_topk_ids"].to(device).int()
    w1 = capture["w1"].to(device)
    w2 = capture["w2"].to(device)

    out = fused_moe_gguf(
        x=active_x,
        w1=w1,
        w2=w2,
        topk_weights=active_weights,
        topk_ids=active_topk_ids,
        qweight_type=int(capture["qweight_type"]),
        qweight_type2=int(capture["qweight_type2"]),
        activation=capture["activation"],
        debug_layer=capture.get("debug_layer"),
    ).cpu()
    _assert_close(
        "current_kernel_active_out",
        out,
        capture["active_out"],
        atol=atol,
        rtol=rtol,
    )
    return out


def _check_masked_kernel(
    capture: dict, device: str, *, atol: float, rtol: float
) -> torch.Tensor | None:
    missing = [key for key in ("w1", "w2") if key not in capture]
    if missing:
        print(
            "masked_kernel: skipped; capture does not include "
            f"{', '.join(missing)}. Set SGLANG_GGUF_MOE_CAPTURE_WEIGHTS=1.",
        )
        return None

    repo_python = Path(__file__).resolve().parents[1] / "python"
    sys.path.insert(0, str(repo_python))

    os.environ.pop("SGLANG_GGUF_MOE_WEIGHTED_ACCUM", None)
    os.environ["SGLANG_GGUF_MOE_MASKED_VEC"] = "1"
    from sglang.srt.layers.quantization.gguf import fused_moe_gguf

    x = capture["x"].to(device)
    topk_weights = capture["topk_weights"].to(device)
    topk_ids = capture["topk_ids"].to(device).int()
    w1 = capture["w1"].to(device)
    w2 = capture["w2"].to(device)

    out = fused_moe_gguf(
        x=x,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        qweight_type=int(capture["qweight_type"]),
        qweight_type2=int(capture["qweight_type2"]),
        activation=capture["activation"],
        debug_layer=capture.get("debug_layer"),
    ).cpu()
    _assert_close(
        "masked_kernel_full_out",
        out,
        capture["out_hidden_states"],
        atol=atol,
        rtol=rtol,
    )
    return out


def _check_weighted_accum_kernel(
    capture: dict, device: str, *, atol: float, rtol: float
) -> torch.Tensor | None:
    missing = [key for key in ("w1", "w2") if key not in capture]
    if missing:
        print(
            "weighted_accum_kernel: skipped; capture does not include "
            f"{', '.join(missing)}. Set SGLANG_GGUF_MOE_CAPTURE_WEIGHTS=1.",
        )
        return None

    repo_python = Path(__file__).resolve().parents[1] / "python"
    sys.path.insert(0, str(repo_python))

    os.environ.pop("SGLANG_GGUF_MOE_MASKED_VEC", None)
    os.environ["SGLANG_GGUF_MOE_WEIGHTED_ACCUM"] = "1"
    from sglang.srt.layers.quantization.gguf import fused_moe_gguf

    x = capture["x"].to(device)
    topk_weights = capture["topk_weights"].to(device)
    topk_ids = capture["topk_ids"].to(device)
    w1 = capture["w1"].to(device)
    w2 = capture["w2"].to(device)

    out = fused_moe_gguf(
        x=x,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        qweight_type=int(capture["qweight_type"]),
        qweight_type2=int(capture["qweight_type2"]),
        activation=capture["activation"],
        debug_layer=capture.get("debug_layer"),
    ).cpu()
    _assert_close(
        "weighted_accum_kernel_full_out",
        out,
        capture["out_hidden_states"],
        atol=atol,
        rtol=rtol,
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for rerunning current kernel when weights are present.",
    )
    parser.add_argument(
        "--skip-current-kernel",
        action="store_true",
        help="Only validate saved active-filter tensors.",
    )
    parser.add_argument(
        "--skip-masked-kernel",
        action="store_true",
        help="Skip rerunning the masked full-token kernel path.",
    )
    parser.add_argument(
        "--skip-weighted-accum-kernel",
        action="store_true",
        help="Skip rerunning the compact weighted-accumulation kernel path.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=3e-2,
        help="Absolute tolerance for BF16 CUDA replay comparisons.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=3e-2,
        help="Relative tolerance for BF16 CUDA replay comparisons.",
    )
    args = parser.parse_args()

    capture = _load_capture(args.capture)
    print(
        "capture: "
        f"layer={capture.get('debug_layer')} "
        f"qtypes=({capture.get('qweight_type_name')},"
        f"{capture.get('qweight_type2_name')}) "
        f"x={tuple(capture['x'].shape)} "
        f"active={tuple(capture['active_x'].shape)}",
    )

    _check_active_reconstruction(capture, atol=args.atol, rtol=args.rtol)
    current_out = None
    masked_out = None
    weighted_accum_out = None
    if not args.skip_current_kernel:
        if args.device == "cpu":
            print("current_kernel: skipped; GGUF CUDA kernel requires CUDA")
        else:
            current_out = _check_current_kernel(
                capture, args.device, atol=args.atol, rtol=args.rtol
            )
    if not args.skip_masked_kernel:
        if args.device == "cpu":
            print("masked_kernel: skipped; GGUF CUDA kernel requires CUDA")
        else:
            masked_out = _check_masked_kernel(
                capture, args.device, atol=args.atol, rtol=args.rtol
            )
    if not args.skip_weighted_accum_kernel:
        if args.device == "cpu":
            print("weighted_accum_kernel: skipped; GGUF CUDA kernel requires CUDA")
        else:
            weighted_accum_out = _check_weighted_accum_kernel(
                capture, args.device, atol=args.atol, rtol=args.rtol
            )
    if current_out is not None and masked_out is not None:
        current_full = _reconstruct_from_active(capture, current_out)
        _assert_close(
            "masked_vs_current_active_replay",
            masked_out,
            current_full,
            atol=args.atol,
            rtol=args.rtol,
        )
    if current_out is not None and weighted_accum_out is not None:
        current_full = _reconstruct_from_active(capture, current_out)
        _assert_close(
            "weighted_accum_vs_current_active_replay",
            weighted_accum_out,
            current_full,
            atol=args.atol,
            rtol=args.rtol,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
