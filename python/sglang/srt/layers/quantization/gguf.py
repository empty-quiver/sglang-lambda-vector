# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from: https://github.com/vllm-project/vllm/blob/ab3e80042eac24dd362408e6d63ad98768046359/vllm/model_executor/layers/quantization/gguf.py
from __future__ import annotations

import logging
import os
import time
import warnings
from typing import TYPE_CHECKING, Any, List, Optional

import gguf
import torch
from gguf import GGMLQuantizationType as WeightType
from torch.nn.parameter import Parameter, UninitializedParameter

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.utils import is_cuda, is_hip, is_musa, is_npu, is_xpu, set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_xpu = is_xpu()
_is_musa = is_musa()
_is_npu = is_npu()

if _is_cuda:
    from sgl_kernel import moe_sum
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
        moe_align_block_size,
    )
    from sgl_kernel.quantization import (
        ggml_dequantize,
        ggml_moe_a8,
        ggml_moe_a8_vec,
        ggml_moe_get_block_size,
        ggml_mul_mat_a8,
        ggml_mul_mat_vec_a8,
    )

    from sglang.jit_kernel.activation import gelu_and_mul, silu_and_mul
elif _is_musa:
    from sgl_kernel import gelu_and_mul, moe_sum, silu_and_mul
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
        moe_align_block_size,
    )
    from sgl_kernel.quantization import (
        ggml_dequantize,
        ggml_moe_a8,
        ggml_moe_a8_vec,
        ggml_moe_get_block_size,
        ggml_mul_mat_a8,
        ggml_mul_mat_vec_a8,
    )
elif _is_npu:
    from gguf import dequantize as gguf_dequantize
else:
    if not _is_hip:
        warnings.warn(f"Only CUDA, MUSA and NPU support GGUF quantization currently.")

logger = logging.getLogger(__name__)


class GGUFConfig(QuantizationConfig):
    """Config class for GGUF."""

    def __init__(self, modules_to_not_convert: list[str] | None = None) -> None:
        super().__init__()
        if _is_hip:
            warnings.warn(f"Only CUDA and MUSA support GGUF quantization currently.")
        self.modules_to_not_convert = modules_to_not_convert or []

    def __repr__(self) -> str:
        return "GGUFConfig()"

    def get_scaled_act_names(self) -> List[str]:
        return []

    def get_name(self) -> "str":
        return "gguf"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 60 if not _is_musa else 21

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []  # no extra configs.

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GGUFConfig":
        modules_to_not_convert = cls.get_from_keys_or(
            config, ["modules_to_not_convert"], None
        )
        return cls(modules_to_not_convert)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding

        if isinstance(layer, LinearBase):
            if is_layer_skipped_gguf(prefix, self.modules_to_not_convert):
                return UnquantizedLinearMethod()
            if _is_npu:
                return GGUFLinearAscendMethod(self)
            return GGUFLinearMethod(self)
        elif isinstance(layer, VocabParallelEmbedding):
            if _is_npu:
                return GGUFEmbeddingAscendMethod(self)
            return GGUFEmbeddingMethod(self)
        elif isinstance(layer, FusedMoE):
            if _is_npu:
                return GGUFMoEAscendMethod(self)
            return GGUFMoEMethod(self)
        return None


def is_layer_skipped_gguf(prefix: str, modules_to_not_convert: list[str]):
    return any(module_name in prefix for module_name in modules_to_not_convert)


UNQUANTIZED_TYPES = {WeightType.F32, WeightType.F16, WeightType.BF16}
STANDARD_QUANT_TYPES = {
    WeightType.Q4_0,
    WeightType.Q4_1,
    WeightType.Q5_0,
    WeightType.Q5_1,
    WeightType.Q8_0,
    WeightType.Q8_1,
}
KQUANT_TYPES = {
    WeightType.Q2_K,
    WeightType.Q3_K,
    WeightType.Q4_K,
    WeightType.Q5_K,
    WeightType.Q6_K,
}
IMATRIX_QUANT_TYPES = {
    WeightType.IQ1_M,
    WeightType.IQ1_S,
    WeightType.IQ2_XXS,
    WeightType.IQ2_XS,
    WeightType.IQ2_S,
    WeightType.IQ3_XXS,
    WeightType.IQ3_S,
    WeightType.IQ4_XS,
    WeightType.IQ4_NL,
}
# TODO(Isotr0py): Currently, we don't have MMQ kernel for I-Matrix quantization.
# Consolidate DEQUANT_TYPES, MMVQ_QUANT_TYPES and MMQ_QUANT_TYPES after we add
# MMQ kernel for I-Matrix quantization.
DEQUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES | IMATRIX_QUANT_TYPES
MMVQ_QUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES | IMATRIX_QUANT_TYPES
MMQ_QUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES

_GGUF_TRACE_COUNT = 0
_GGUF_MOE_CAPTURE_COUNT = 0


def _gguf_env_enabled(name: str) -> bool:
    value = os.getenv(name, "")
    return value.lower() not in ("", "0", "false", "no", "off")


def _gguf_trace_layer_enabled(debug_layer: Optional[int]) -> bool:
    target = os.getenv("SGLANG_GGUF_MOE_TRACE_LAYER")
    return target is None or debug_layer is None or str(debug_layer) == target


def _gguf_trace_enabled(
    name: str = "SGLANG_GGUF_MOE_TRACE", debug_layer: Optional[int] = None
) -> bool:
    if _is_cuda and torch.cuda.is_current_stream_capturing():
        return False
    return _gguf_env_enabled(name) and _gguf_trace_layer_enabled(debug_layer)


def _gguf_trace_sync():
    if (
        _gguf_env_enabled("SGLANG_GGUF_MOE_TRACE_SYNC")
        and torch.cuda.is_available()
    ):
        torch.cuda.synchronize()


def _gguf_trace_start(enabled: bool) -> Optional[float]:
    if not enabled:
        return None
    _gguf_trace_sync()
    return time.perf_counter()


def _gguf_quant_type_to_int(qweight_type: int) -> int:
    if isinstance(qweight_type, torch.Tensor):
        return int(qweight_type.item())
    return int(qweight_type)


def _gguf_quant_type_name(qweight_type: int) -> str:
    qtype = _gguf_quant_type_to_int(qweight_type)
    try:
        return WeightType(qtype).name
    except ValueError:
        return f"UNKNOWN_{qtype}"


def _gguf_trace_print(tag: str, message: str):
    global _GGUF_TRACE_COUNT

    limit = int(os.getenv("SGLANG_GGUF_MOE_TRACE_LIMIT", "256"))
    if limit >= 0 and _GGUF_TRACE_COUNT >= limit:
        return
    _GGUF_TRACE_COUNT += 1
    print(f"[{tag} #{_GGUF_TRACE_COUNT}] {message}", flush=True)


def _gguf_trace_elapsed_ms(start: Optional[float]) -> str:
    if start is None:
        return "n/a"
    _gguf_trace_sync()
    return f"{(time.perf_counter() - start) * 1000.0:.3f}"


def _gguf_moe_capture_enabled(debug_layer: Optional[int]) -> bool:
    if _is_cuda and torch.cuda.is_current_stream_capturing():
        return False
    capture_dir = os.getenv("SGLANG_GGUF_MOE_CAPTURE_DIR")
    if not capture_dir:
        return False
    target = os.getenv("SGLANG_GGUF_MOE_CAPTURE_LAYER")
    if target is None:
        return False
    return target == "all" or (
        debug_layer is not None and str(debug_layer) == target
    )


def _gguf_moe_capture_limit_reached() -> bool:
    limit = int(os.getenv("SGLANG_GGUF_MOE_CAPTURE_LIMIT", "1"))
    return limit >= 0 and _GGUF_MOE_CAPTURE_COUNT >= limit


def _gguf_maybe_capture_active_moe(
    *,
    debug_layer: Optional[int],
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    active_mask: torch.Tensor,
    active_token_ids: torch.Tensor,
    active_topk_ids: torch.Tensor,
    active_weights: torch.Tensor,
    active_x: torch.Tensor,
    active_out: torch.Tensor,
    out_hidden_states: torch.Tensor,
    qweight_type: int,
    qweight_type2: int,
    activation: str,
):
    global _GGUF_MOE_CAPTURE_COUNT

    if not _gguf_moe_capture_enabled(debug_layer) or _gguf_moe_capture_limit_reached():
        return

    capture_dir = os.getenv("SGLANG_GGUF_MOE_CAPTURE_DIR")
    assert capture_dir is not None
    os.makedirs(capture_dir, exist_ok=True)

    _GGUF_MOE_CAPTURE_COUNT += 1
    capture_path = os.path.join(
        capture_dir,
        (
            f"gguf_moe_layer{debug_layer}_capture{_GGUF_MOE_CAPTURE_COUNT}_"
            f"tokens{x.shape[0]}_active{active_topk_ids.shape[0]}.pt"
        ),
    )

    payload = {
        "debug_layer": debug_layer,
        "activation": activation,
        "qweight_type": _gguf_quant_type_to_int(qweight_type),
        "qweight_type2": _gguf_quant_type_to_int(qweight_type2),
        "qweight_type_name": _gguf_quant_type_name(qweight_type),
        "qweight_type2_name": _gguf_quant_type_name(qweight_type2),
        "x": x.detach().cpu(),
        "topk_weights": topk_weights.detach().cpu(),
        "topk_ids": topk_ids.detach().cpu(),
        "active_mask": active_mask.detach().cpu(),
        "active_token_ids": active_token_ids.detach().cpu(),
        "active_topk_ids": active_topk_ids.detach().cpu(),
        "active_weights": active_weights.detach().cpu(),
        "active_x": active_x.detach().cpu(),
        "active_out": active_out.detach().cpu(),
        "out_hidden_states": out_hidden_states.detach().cpu(),
        "x_shape": tuple(x.shape),
        "w1_shape": tuple(w1.shape),
        "w2_shape": tuple(w2.shape),
        "w1_stride": tuple(w1.stride()),
        "w2_stride": tuple(w2.stride()),
        "w1_dtype": str(w1.dtype),
        "w2_dtype": str(w2.dtype),
    }
    if _gguf_env_enabled("SGLANG_GGUF_MOE_CAPTURE_WEIGHTS"):
        payload["w1"] = w1.detach().cpu()
        payload["w2"] = w2.detach().cpu()

    torch.save(payload, capture_path)
    print(f"[GGUF_MOE_CAPTURE saved] {capture_path}", flush=True)


def fused_mul_mat_gguf(
    x: torch.Tensor, qweight: torch.Tensor, qweight_type: int
) -> torch.Tensor:
    trace_enabled = _gguf_trace_enabled("SGLANG_GGUF_MATMUL_TRACE")
    trace_start = _gguf_trace_start(trace_enabled)
    branch = "unknown"
    if qweight_type in IMATRIX_QUANT_TYPES:
        mmvq_safe = 8 if qweight.shape[0] > 5120 else 16
    else:
        mmvq_safe = 2 if qweight.shape[0] > 5120 else 6
    # HACK: when doing chunked prefill we don't generate output tokens
    # so input to logits generator is empty which causes invalid parameter
    if x.shape[0] == 0:
        branch = "empty"
        return torch.empty(x.shape[0], qweight.shape[0], dtype=x.dtype, device=x.device)
    # there is no need to call any kernel for fp16/bf16
    if qweight_type in UNQUANTIZED_TYPES:
        branch = "unquantized"
        if qweight.dtype != x.dtype:
            qweight = qweight.to(dtype=x.dtype)
        y = x @ qweight.T
    # enable MMVQ in contiguous batching with batch_size=1
    elif x.shape[0] <= mmvq_safe and qweight_type in MMVQ_QUANT_TYPES:
        branch = "mmvq"
        y = ggml_mul_mat_vec_a8(qweight, x, qweight_type, qweight.shape[0])
    # Use MMQ Kernel if it's available (standard + k-quants)
    elif qweight_type in MMQ_QUANT_TYPES:
        branch = "mmq"
        y = ggml_mul_mat_a8(qweight, x, qweight_type, qweight.shape[0])
    # If there is no available MMQ kernel, fallback to dequantize
    elif qweight_type in DEQUANT_TYPES:
        branch = "dequant_fallback"
        block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
        shape = (qweight.shape[0], qweight.shape[1] // type_size * block_size)
        weight = ggml_dequantize(qweight, qweight_type, *shape, x.dtype)
        y = x @ weight.T
    else:
        # Raise an error if the quantization type is not supported.
        # Might be useful if llama.cpp adds a new quantization type.
        # Wrap to GGMLQuantizationType IntEnum to make sure it's a valid type.
        qweight_type = WeightType(qweight_type)
        raise NotImplementedError(f"Unsupported GGUF quantization type: {qweight_type}")
    if trace_enabled:
        _gguf_trace_print(
            "GGUF_MATMUL_TRACE",
            f"branch={branch} elapsed_ms={_gguf_trace_elapsed_ms(trace_start)} "
            f"x={tuple(x.shape)} qweight={tuple(qweight.shape)} "
            f"qtype={_gguf_quant_type_name(qweight_type)} out={tuple(y.shape)} "
            f"mmvq_safe={mmvq_safe}",
        )
    return y


def _gguf_moe_block_size(qweight_type: int) -> int:
    # Matches sgl-kernel's MOE_X_* constants. This is the token tile used by
    # ggml_moe_a8, not the GGUF quantization block size.
    return 8 if _is_hip else 4


def _gguf_moe_masked_vec_enabled(qweight_type: int, qweight_type2: int) -> bool:
    return (
        _gguf_env_enabled("SGLANG_GGUF_MOE_MASKED_VEC")
        and qweight_type in MMVQ_QUANT_TYPES
        and qweight_type2 in MMVQ_QUANT_TYPES
    )


def fused_moe_gguf(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    qweight_type: int,
    qweight_type2: int,
    activation: str,
    debug_layer: Optional[int] = None,
) -> torch.Tensor:
    def act(x: torch.Tensor):
        if activation == "silu":
            return silu_and_mul(x)
        elif activation == "gelu":
            return gelu_and_mul(x)
        raise ValueError(f"Unsupported activation: {activation}")

    trace_enabled = _gguf_trace_enabled("SGLANG_GGUF_MOE_TRACE", debug_layer)
    trace_start = _gguf_trace_start(trace_enabled)

    def trace_done(branch: str, extra: str = ""):
        if not trace_enabled:
            return
        if topk_ids.numel() > 0:
            topk_min = int(topk_ids.min().item())
            topk_max = int(topk_ids.max().item())
            topk_range = f"topk_minmax=({topk_min},{topk_max})"
        else:
            topk_range = "topk_minmax=(empty)"
        suffix = f" {extra}" if extra else ""
        _gguf_trace_print(
            "GGUF_MOE_TRACE",
            f"layer={debug_layer} branch={branch} "
            f"elapsed_ms={_gguf_trace_elapsed_ms(trace_start)} "
            f"x={tuple(x.shape)} w1={tuple(w1.shape)} w2={tuple(w2.shape)} "
            f"topk={tuple(topk_ids.shape)} {topk_range} "
            f"qtypes=({_gguf_quant_type_name(qweight_type)},"
            f"{_gguf_quant_type_name(qweight_type2)}) "
            f"activation={activation}{suffix}",
        )

    out_hidden_states = torch.empty_like(x)
    has_cold_experts = not (
        _is_cuda and torch.cuda.is_current_stream_capturing()
    ) and bool(torch.any(topk_ids < 0).item())
    use_masked_vec = has_cold_experts and _gguf_moe_masked_vec_enabled(
        qweight_type, qweight_type2
    )
    if has_cold_experts and not use_masked_vec:
        out_hidden_states = torch.zeros_like(x)
        active_mask = topk_ids >= 0
        active_topk_ids = topk_ids[active_mask].view(-1, 1).contiguous()
        active_tokens = active_topk_ids.shape[0]
        if active_tokens == 0:
            return out_hidden_states

        num_tokens = x.shape[0]
        token_ids = (
            torch.arange(num_tokens, device=x.device, dtype=torch.long)
            .view(-1, 1)
            .expand_as(topk_ids)
        )
        active_token_ids = token_ids[active_mask].contiguous()
        active_weights = topk_weights[active_mask].view(active_tokens, 1)
        active_x = x.index_select(0, active_token_ids)

        active_out = fused_moe_gguf(
            x=active_x,
            w1=w1,
            w2=w2,
            topk_weights=active_weights,
            topk_ids=active_topk_ids,
            qweight_type=qweight_type,
            qweight_type2=qweight_type2,
            activation=activation,
            debug_layer=debug_layer,
        )
        out_hidden_states.index_add_(0, active_token_ids, active_out)
        _gguf_maybe_capture_active_moe(
            debug_layer=debug_layer,
            x=x,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            active_mask=active_mask,
            active_token_ids=active_token_ids,
            active_topk_ids=active_topk_ids,
            active_weights=active_weights,
            active_x=active_x,
            active_out=active_out,
            out_hidden_states=out_hidden_states,
            qweight_type=qweight_type,
            qweight_type2=qweight_type2,
            activation=activation,
        )
        trace_done(
            "active_filter",
            f"active_tokens={active_tokens} original_tokens={num_tokens}",
        )
        return out_hidden_states

    # unless we decent expert reuse we are better off running moe_vec kernel
    if (
        not os.getenv("SGLANG_GGUF_FORCE_VEC")
        and qweight_type2 in MMQ_QUANT_TYPES
        and qweight_type in MMQ_QUANT_TYPES
        and x.shape[0] > 64
    ):
        num_tokens, _ = x.shape
        E, N, _ = w1.shape
        top_k = topk_ids.shape[1]
        BLOCK_SIZE = _gguf_moe_block_size(qweight_type)

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, BLOCK_SIZE, E
        )
        if os.getenv("SGLANG_GGUF_SYNC_MOE"):
            print(
                "[GGUF_MOE_SYNC post_align] "
                f"layer={debug_layer} sorted={tuple(sorted_token_ids.shape)} "
                f"experts={tuple(expert_ids.shape)} block={BLOCK_SIZE}",
                flush=True,
            )
            torch.cuda.synchronize()
            print(f"[GGUF_MOE_SYNC post_align_sync_ok] layer={debug_layer}", flush=True)
        debug_target = os.getenv("SGLANG_GGUF_DEBUG_LAYER")
        debug_enabled = os.getenv("SGLANG_GGUF_DEBUG_MOE") and (
            debug_target is None
            or debug_layer is None
            or str(debug_layer) == debug_target
        )
        if debug_enabled:
            torch.cuda.synchronize()
            post_padded = int(num_tokens_post_padded.item())
            num_blocks = (post_padded + BLOCK_SIZE - 1) // BLOCK_SIZE
            expert_slice = expert_ids[:num_blocks]
            sorted_slice = sorted_token_ids[:post_padded]
            save_dir = os.getenv("SGLANG_GGUF_DEBUG_SAVE")
            if save_dir and topk_ids.shape[1] == 1:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(
                    save_dir,
                    f"gguf_moe_layer{debug_layer}_active_{x.shape[0]}.pt",
                )
                if not os.path.exists(save_path):
                    torch.save(
                        {
                            "x": x.detach().cpu(),
                            "w1": w1.detach().cpu(),
                            "sorted_token_ids": sorted_token_ids.detach().cpu(),
                            "expert_ids": expert_ids.detach().cpu(),
                            "num_tokens_post_padded": num_tokens_post_padded.detach().cpu(),
                            "topk_ids": topk_ids.detach().cpu(),
                            "topk_weights": topk_weights.detach().cpu(),
                            "qweight_type": qweight_type,
                            "qweight_type2": qweight_type2,
                            "w1_shape": tuple(w1.shape),
                            "w1_stride": tuple(w1.stride()),
                            "w1_dtype": str(w1.dtype),
                            "w2_shape": tuple(w2.shape),
                            "w2_stride": tuple(w2.stride()),
                            "w2_dtype": str(w2.dtype),
                            "debug_layer": debug_layer,
                        },
                        save_path,
                    )
                    print(f"[GGUF_MOE_DEBUG saved] {save_path}", flush=True)
            print(
                "[GGUF_MOE_DEBUG pre_moe1] "
                f"layer={debug_layer} x={tuple(x.shape)} "
                f"w1={tuple(w1.shape)} w2={tuple(w2.shape)} "
                f"topk={tuple(topk_ids.shape)} qtypes=({qweight_type},{qweight_type2}) "
                f"block={BLOCK_SIZE} post={post_padded} "
                f"topk_minmax=({int(topk_ids.min().item())},{int(topk_ids.max().item())}) "
                f"expert_minmax=({int(expert_slice.min().item())},{int(expert_slice.max().item())}) "
                f"sorted_minmax=({int(sorted_slice.min().item())},{int(sorted_slice.max().item())})",
                flush=True,
            )
        out = ggml_moe_a8(
            x,
            w1,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            qweight_type,
            N,
            top_k,
            num_tokens,
        )
        if os.getenv("SGLANG_GGUF_SYNC_MOE"):
            print(
                "[GGUF_MOE_SYNC post_moe1] "
                f"layer={debug_layer} out={tuple(out.shape)} out_stride={out.stride()} "
                f"num_tokens={num_tokens} top_k={top_k} hidden={N}",
                flush=True,
            )
            torch.cuda.synchronize()
            print(f"[GGUF_MOE_SYNC post_moe1_sync_ok] layer={debug_layer}", flush=True)
        if debug_enabled:
            print(
                "[GGUF_MOE_DEBUG post_moe1] "
                f"layer={debug_layer} out={tuple(out.shape)} "
                f"contiguous={out.is_contiguous()} stride={out.stride()}",
                flush=True,
            )
            torch.cuda.synchronize()
            print(f"[GGUF_MOE_DEBUG post_moe1_sync_ok] layer={debug_layer}", flush=True)
        out = act(out)
        if os.getenv("SGLANG_GGUF_SYNC_MOE"):
            print(
                "[GGUF_MOE_SYNC post_act] "
                f"layer={debug_layer} out={tuple(out.shape)} out_stride={out.stride()}",
                flush=True,
            )
            torch.cuda.synchronize()
            print(f"[GGUF_MOE_SYNC post_act_sync_ok] layer={debug_layer}", flush=True)
        second_save_dir = os.getenv("SGLANG_GGUF_DEBUG_SECOND_SAVE")
        if second_save_dir and (
            debug_target is None
            or debug_layer is None
            or str(debug_layer) == debug_target
        ):
            os.makedirs(second_save_dir, exist_ok=True)
            second_save_path = os.path.join(
                second_save_dir,
                f"gguf_moe_layer{debug_layer}_second_{out.shape[0]}.pt",
            )
            if not os.path.exists(second_save_path):
                torch.save(
                    {
                        "x": out.detach().cpu(),
                        "w2": w2.detach().cpu(),
                        "sorted_token_ids": sorted_token_ids.detach().cpu(),
                        "expert_ids": expert_ids.detach().cpu(),
                        "num_tokens_post_padded": num_tokens_post_padded.detach().cpu(),
                        "topk_weights": topk_weights.detach().cpu(),
                        "qweight_type2": qweight_type2,
                        "row": w2.shape[1],
                        "top_k": 1,
                        "tokens": num_tokens * top_k,
                        "debug_layer": debug_layer,
                    },
                    second_save_path,
                )
                print(f"[GGUF_MOE_DEBUG second_saved] {second_save_path}", flush=True)
        if os.getenv("SGLANG_GGUF_FORCE_VEC_W2"):
            out = ggml_moe_a8_vec(
                out, w2, topk_ids, 1, qweight_type2, w2.shape[1], num_tokens * top_k
            )
        else:
            out = ggml_moe_a8(
                out,
                w2,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                qweight_type2,
                w2.shape[1],
                1,
                num_tokens * top_k,
            )
        if os.getenv("SGLANG_GGUF_SYNC_MOE"):
            print(
                "[GGUF_MOE_SYNC pre_mul] "
                f"layer={debug_layer} out={tuple(out.shape)} out_stride={out.stride()} "
                f"weights={tuple(topk_weights.shape)} weights_stride={topk_weights.stride()} "
                f"num_tokens={num_tokens} top_k={top_k} hidden={w2.shape[1]}",
                flush=True,
            )
            torch.cuda.synchronize()
            print(f"[GGUF_MOE_SYNC pre_mul_sync_ok] layer={debug_layer}", flush=True)
        out = out.reshape(num_tokens, top_k, w2.shape[1]).mul_(
            topk_weights.view(num_tokens, top_k, 1)
        )
        if os.getenv("SGLANG_GGUF_SYNC_MOE"):
            print(f"[GGUF_MOE_SYNC post_mul] layer={debug_layer}", flush=True)
            torch.cuda.synchronize()
            print(f"[GGUF_MOE_SYNC post_mul_sync_ok] layer={debug_layer}", flush=True)
        # TODO(FlamingoPg): maybe we can use moe_sum_reduce here?
        moe_sum(out, out_hidden_states)
        if os.getenv("SGLANG_GGUF_SYNC_MOE"):
            print(f"[GGUF_MOE_SYNC post_moe_sum] layer={debug_layer}", flush=True)
            torch.cuda.synchronize()
            print(f"[GGUF_MOE_SYNC post_moe_sum_sync_ok] layer={debug_layer}", flush=True)
        trace_done(
            "mmq_sorted",
            f"block={BLOCK_SIZE} post_padded={int(num_tokens_post_padded.item())}",
        )
    elif qweight_type2 in MMVQ_QUANT_TYPES and qweight_type in MMVQ_QUANT_TYPES:
        num_tokens, _ = x.shape
        E, N, _ = w1.shape
        top_k = topk_ids.shape[1]
        masked_slots = None
        if trace_enabled:
            masked_slots = int((topk_ids < 0).sum().item())

        out = ggml_moe_a8_vec(x, w1, topk_ids, top_k, qweight_type, N, num_tokens)
        out = act(out)

        out = ggml_moe_a8_vec(
            out, w2, topk_ids, 1, qweight_type2, w2.shape[1], num_tokens * top_k
        )
        out = out.reshape(num_tokens, top_k, w2.shape[1]).mul_(
            topk_weights.view(num_tokens, top_k, 1)
        )
        moe_sum(out, out_hidden_states)
        trace_extra = f"num_tokens={num_tokens} top_k={top_k} hidden={N}"
        if masked_slots:
            trace_extra = f"{trace_extra} masked_slots={masked_slots}"
        trace_done(
            "mmvq_vec_masked" if masked_slots else "mmvq_vec",
            trace_extra,
        )
    else:
        logger.warning_once(
            "There is no support for fast MoE kernel "
            "for current quantization method. "
            "Falling back to slow implementation. "
        )
        for tok, (w, idx) in enumerate(zip(topk_weights, topk_ids)):
            inp = x[tok].reshape((1,) + x.shape[1:])
            current_hidden_state = None
            for ww, ii in zip(w, idx):
                expert_up = w1[ii]

                out = fused_mul_mat_gguf(inp, expert_up, qweight_type)
                out = act(out)

                expert_down = w2[ii]
                current_state = fused_mul_mat_gguf(
                    out, expert_down, qweight_type2
                ).mul_(ww)
                if current_hidden_state is None:
                    current_hidden_state = current_state
                else:
                    current_hidden_state.add_(current_state)
            out_hidden_states[tok] = current_hidden_state
        trace_done("slow_fallback", f"num_tokens={x.shape[0]} top_k={topk_ids.shape[1]}")
    return out_hidden_states


def apply_gguf_embedding(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qweight_type: int,
    hidden_size: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if qweight_type in UNQUANTIZED_TYPES:
        return torch.embedding(qweight, x)
    elif qweight_type in DEQUANT_TYPES:
        block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
        x_flat = x.flatten()
        assert hidden_size == qweight.shape[1] // type_size * block_size
        quant = torch.index_select(qweight, dim=0, index=x_flat)
        dequant = ggml_dequantize(
            quant, qweight_type, hidden_size, x_flat.shape[0], dtype
        )
        return dequant.view(*x.shape, hidden_size)
    else:
        qweight_type = WeightType(qweight_type)
        raise NotImplementedError(f"Unsupported GGUF quantization type: {qweight_type}")


class GGUFLinearMethod(LinearMethodBase):
    """Linear method for GGUF.

    Args:
        quant_config: The GGUF quantization config.
    """

    def __init__(self, quant_config: GGUFConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self.params_dtype = params_dtype
        output_size_per_partition = sum(output_partition_sizes)

        tensor_shape = (output_size_per_partition, input_size_per_partition)
        qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "is_gguf_weight": True,
                "data_container": [],
                "shard_id": [],
                "shard_id_map": {},
            },
        )
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("qweight", qweight)

        qweight_type = Parameter(
            torch.empty(len(output_partition_sizes), dtype=torch.uint8),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight_type,
            {
                "is_gguf_weight_type": True,
                "weight_type": 0,
                "shard_weight_type": {},
                "ignore_warning": True,
            },
        )
        set_weight_attrs(qweight_type, extra_weight_attrs)
        layer.register_parameter("qweight_type", qweight_type)

    def process_weights_after_loading(self, layer: torch.nn.Module):
        qweight_type = layer.qweight_type.weight_type
        if not (qweight_type in UNQUANTIZED_TYPES or qweight_type in DEQUANT_TYPES):
            qweight_type = WeightType(qweight_type)
            raise ValueError(
                f"Unsupported GGUF quantization type {qweight_type} in layer {layer}."
            )
        # For MergedColumnParallelLinear and QKVParallelLinear, we need to
        # materialize the padded weight parameter for CUDA Graph compatibility.
        self._create_padded_weight_param(layer)

    def _create_padded_weight_param(self, layer: torch.nn.Module):
        """Create padded weight parameter for GGUF MergedLinear layer."""
        qweight = layer.qweight
        shard_id_map = qweight.shard_id_map
        shard_id = qweight.shard_id
        if len(data_container := qweight.data_container) > 1:
            dtype = {data.dtype for data in data_container}
            assert len(dtype) == 1, ValueError(
                f"Data container has mixed dtypes: {dtype}"
            )
            dtype = next(iter(dtype))
            # concat dim0 and pad dim1
            padded_side = max(x.size(1) for x in data_container)
            concat_side = sum(x.size(0) for x in data_container)
            # Pad the quantized weights to dense tensor, and create a map
            # with the location of each shard in the padded tensor.
            padded_data = torch.zeros(
                (concat_side, padded_side), dtype=dtype, device=qweight.device
            )
            # (dim0_start, dim0_end, dim1_size)
            shard_offset_map = dict[str, tuple[int, int, int]]()
            for idx in shard_id:
                id_in_container = shard_id_map[idx]
                start = sum(x.size(0) for x in data_container[:id_in_container])
                end = start + data_container[id_in_container].size(0)
                size = data_container[id_in_container].size(1)
                padded_data[start:end, :size] = data_container[id_in_container]
                shard_offset_map[idx] = (start, end, size)
            qweight.data_container.clear()
            padded_param = Parameter(padded_data, requires_grad=False)
            set_weight_attrs(padded_param, vars(qweight))
            set_weight_attrs(padded_param, {"shard_offset_map": shard_offset_map})
            layer.register_parameter("qweight", padded_param)

    @staticmethod
    def _logical_shard_sort_key(shard_id):
        if isinstance(shard_id, tuple):
            return (0, min(shard_id))
        if isinstance(shard_id, str):
            return (0, {"q": 0, "k": 1, "v": 2}.get(shard_id, 1000))
        return (0, int(shard_id))

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shard_id = layer.qweight.shard_id

        if shard_id:
            # dequantize shard weights respectively
            shard_id = sorted(shard_id, key=self._logical_shard_sort_key)
            qweight = layer.qweight
            result = []
            for idx in shard_id:
                start, end, offset = layer.qweight.shard_offset_map[idx]
                qweight_type = layer.qweight_type.shard_weight_type[idx]
                result.append(
                    fused_mul_mat_gguf(
                        x, qweight[start:end, :offset].contiguous(), qweight_type
                    )
                )
            out = torch.cat(result, axis=1)
        else:
            qweight = layer.qweight
            qweight_type = layer.qweight_type.weight_type
            out = fused_mul_mat_gguf(x, qweight, qweight_type)
        if bias is not None:
            out.add_(bias)
        return out


class GGUFMoEMethod(FusedMoEMethodBase):
    """MoE method for GGUF.

    Args:
        quant_config: The GGUF quantization config.
    """

    def __init__(self, quant_config: GGUFConfig):
        self.quant_config = quant_config
        self.fused_experts = None
        self.num_gpu_experts = -1

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        tensor_shape = (num_experts, 2 * intermediate_size_per_partition, hidden_size)
        # gate up proj
        w13_qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            w13_qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "is_gguf_weight": True,
                "data_container": [],
            },
        )
        set_weight_attrs(w13_qweight, extra_weight_attrs)
        layer.register_parameter("w13_qweight", w13_qweight)

        w13_qweight_type = Parameter(
            torch.empty(1, dtype=torch.uint8), requires_grad=False
        )
        set_weight_attrs(
            w13_qweight_type,
            {"is_gguf_weight_type": True, "weight_type": 0, "ignore_warning": True},
        )
        set_weight_attrs(w13_qweight_type, extra_weight_attrs)
        layer.register_parameter("w13_qweight_type", w13_qweight_type)

        tensor_shape = (num_experts, intermediate_size_per_partition, hidden_size)
        # gate down proj
        w2_qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            w2_qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "is_gguf_weight": True,
                "data_container": [],
            },
        )
        set_weight_attrs(w2_qweight, extra_weight_attrs)
        layer.register_parameter("w2_qweight", w2_qweight)

        w2_qweight_type = Parameter(
            torch.empty(1, dtype=torch.uint8), requires_grad=False
        )
        set_weight_attrs(
            w2_qweight_type,
            {"is_gguf_weight_type": True, "weight_type": 0, "ignore_warning": True},
        )

        set_weight_attrs(w2_qweight_type, extra_weight_attrs)
        layer.register_parameter("w2_qweight_type", w2_qweight_type)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def process_weights_after_loading(self, layer: torch.nn.Module):
        if hasattr(layer, "materialize_gguf_weights"):
            layer.materialize_gguf_weights()

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        assert self.fused_experts is None

        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        assert (
            self.moe_runner_config.activation == "silu"
        ), "Only SiLU activation is supported."

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        moe_runner_config = self.moe_runner_config

        topk_weights, topk_ids, _ = topk_output
        debug_layer = getattr(layer, "layer_id", None)
        if os.getenv("SGLANG_GGUF_TRACE_LAYER_IDS"):
            print(
                "[GGUF_MOE_TRACE apply] "
                f"layer={debug_layer} x={tuple(x.shape)} topk={tuple(topk_ids.shape)} "
                f"topk_minmax=({int(topk_ids.min().item())},{int(topk_ids.max().item())})",
                flush=True,
            )

        output = fused_moe_gguf(
            x=x,
            w1=layer.w13_qweight,
            w2=layer.w2_qweight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            qweight_type=layer.w13_qweight_type.weight_type,
            qweight_type2=layer.w2_qweight_type.weight_type,
            activation=moe_runner_config.activation,
            debug_layer=debug_layer,
        )
        return StandardCombineInput(hidden_states=output)


class GGUFEmbeddingMethod(GGUFLinearMethod):
    """Embedding method for GGUF.

    Args:
        quant_config: The GGUF quantization config.
    """

    def embedding(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        qweight = layer.qweight
        qweight_type = layer.qweight_type.weight_type
        hidden_size = qweight.tensor_shape[1]

        return apply_gguf_embedding(
            x, qweight, qweight_type, hidden_size, dtype=self.params_dtype
        )


class GGUFUninitializedParameter(UninitializedParameter):
    cls_to_become = Parameter
    data_container: list[torch.Tensor]


# =============================================================================
# NPU-specific implementations for Ascend hardware
# =============================================================================
def ggml_dequantize_ascend(
    qweight: torch.Tensor,
    qweight_type: int,
    rows: int,
    cols: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize GGML quantized weights for NPU.

    Uses gguf library's reference implementation which supports all GGML formats
    and is guaranteed to be correct. The dequantization runs on CPU during model
    loading, then the dequantized weights are transferred to NPU for inference.
    """

    # Move to CPU for dequantization using gguf library
    qweight_cpu = qweight.cpu().numpy()

    # Use gguf library's dequantize (supports all GGML formats)
    dequant_np = gguf_dequantize(qweight_cpu, qweight_type)

    # Convert to torch and move to target device
    result = torch.from_numpy(dequant_np).to(dtype=dtype, device=qweight.device)
    result = result.reshape(rows, cols)

    return result


class GGUFLinearAscendMethod(LinearMethodBase):
    """Linear method for GGUF on Ascend NPU."""

    def __init__(self, quant_config: GGUFConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self.params_dtype = params_dtype
        output_size_per_partition = sum(output_partition_sizes)

        tensor_shape = (output_size_per_partition, input_size_per_partition)
        qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "is_gguf_weight": True,
                "data_container": [],
                "shard_id": [],
                "shard_id_map": {},
            },
        )
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("qweight", qweight)

        qweight_type = Parameter(
            torch.empty(len(output_partition_sizes), dtype=torch.uint8),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight_type,
            {
                "is_gguf_weight_type": True,
                "weight_type": 0,
                "shard_weight_type": {},
                "ignore_warning": True,
            },
        )
        set_weight_attrs(qweight_type, extra_weight_attrs)
        layer.register_parameter("qweight_type", qweight_type)

    def process_weights_after_loading(self, layer: torch.nn.Module):
        qweight_type = layer.qweight_type.weight_type
        if not (qweight_type in UNQUANTIZED_TYPES or qweight_type in DEQUANT_TYPES):
            raise ValueError(
                f"Unsupported GGUF quantization type {WeightType(qweight_type)} in layer."
            )
        self._create_padded_weight_param(layer)
        # Pre-dequantize weights for faster inference
        self._pre_dequantize_weights(layer)

    def _create_padded_weight_param(self, layer: torch.nn.Module):
        """Create padded weight parameter for GGUF MergedLinear layer."""
        qweight = layer.qweight
        shard_id_map = qweight.shard_id_map
        shard_id = qweight.shard_id
        if len(data_container := qweight.data_container) > 1:
            dtype = {data.dtype for data in data_container}
            assert len(dtype) == 1
            dtype = next(iter(dtype))
            padded_side = max(x.size(1) for x in data_container)
            concat_side = sum(x.size(0) for x in data_container)
            padded_data = torch.zeros(
                (concat_side, padded_side), dtype=dtype, device=qweight.device
            )
            shard_offset_map = dict[str, tuple[int, int, int]]()
            for idx in shard_id:
                id_in_container = shard_id_map[idx]
                start = sum(x.size(0) for x in data_container[:id_in_container])
                end = start + data_container[id_in_container].size(0)
                size = data_container[id_in_container].size(1)
                padded_data[start:end, :size] = data_container[id_in_container]
                shard_offset_map[idx] = (start, end, size)
            qweight.data_container.clear()
            padded_param = Parameter(padded_data, requires_grad=False)
            set_weight_attrs(padded_param, vars(qweight))
            set_weight_attrs(padded_param, {"shard_offset_map": shard_offset_map})
            layer.register_parameter("qweight", padded_param)

    def _pre_dequantize_weights(self, layer: torch.nn.Module):
        """Pre-dequantize GGML weights to FP16 for faster inference.

        This eliminates runtime dequantization overhead at the cost of more memory.
        """
        qweight = layer.qweight
        qweight_type = layer.qweight_type.weight_type

        if qweight_type in UNQUANTIZED_TYPES and qweight.dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            layer.dequantized_weight = qweight
            return

        shard_id = getattr(qweight, "shard_id", None)
        has_shard_offset = hasattr(qweight, "shard_offset_map")

        if shard_id and has_shard_offset:
            # Handle sharded weights (QKV merged)
            shard_id = ["q", "k", "v"] if "q" in shard_id else shard_id
            dequant_shards = []
            for idx in shard_id:
                start, end, offset = qweight.shard_offset_map[idx]
                shard_qtype = layer.qweight_type.shard_weight_type[idx]
                shard_data = qweight[start:end, :offset].contiguous()

                block_size, type_size = gguf.GGML_QUANT_SIZES[shard_qtype]
                shape = (
                    shard_data.shape[0],
                    shard_data.shape[1] // type_size * block_size,
                )
                dequant = ggml_dequantize_ascend(
                    shard_data, shard_qtype, *shape, self.params_dtype
                )
                dequant_shards.append(dequant)

            dequant_weight = torch.cat(dequant_shards, dim=0)
        else:
            # Handle single weight
            block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
            shape = (qweight.shape[0], qweight.shape[1] // type_size * block_size)
            dequant_weight = ggml_dequantize_ascend(
                qweight, qweight_type, *shape, self.params_dtype
            )

        layer.dequantized_weight = dequant_weight

        if hasattr(layer, "qweight"):
            del layer.qweight
        if hasattr(layer, "qweight_type"):
            del layer.qweight_type

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Use pre-dequantized weight (always available after process_weights_after_loading)
        weight = layer.dequantized_weight
        out = x @ weight.T
        if bias is not None:
            out.add_(bias)
        return out


class GGUFMoEAscendMethod(FusedMoEMethodBase):
    """MoE method for GGUF on Ascend NPU."""

    def __init__(self, quant_config: GGUFConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        tensor_shape = (num_experts, 2 * intermediate_size_per_partition, hidden_size)
        w13_qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            w13_qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "is_gguf_weight": True,
                "data_container": [],
            },
        )
        set_weight_attrs(w13_qweight, extra_weight_attrs)
        layer.register_parameter("w13_qweight", w13_qweight)

        w13_qweight_type = Parameter(
            torch.empty(1, dtype=torch.uint8), requires_grad=False
        )
        set_weight_attrs(
            w13_qweight_type,
            {"is_gguf_weight_type": True, "weight_type": 0, "ignore_warning": True},
        )
        set_weight_attrs(w13_qweight_type, extra_weight_attrs)
        layer.register_parameter("w13_qweight_type", w13_qweight_type)

        tensor_shape = (num_experts, intermediate_size_per_partition, hidden_size)
        w2_qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            w2_qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "is_gguf_weight": True,
                "data_container": [],
            },
        )
        set_weight_attrs(w2_qweight, extra_weight_attrs)
        layer.register_parameter("w2_qweight", w2_qweight)

        w2_qweight_type = Parameter(
            torch.empty(1, dtype=torch.uint8), requires_grad=False
        )
        set_weight_attrs(
            w2_qweight_type,
            {"is_gguf_weight_type": True, "weight_type": 0, "ignore_warning": True},
        )
        set_weight_attrs(w2_qweight_type, extra_weight_attrs)
        layer.register_parameter("w2_qweight_type", w2_qweight_type)

        # Store params_dtype for pre-dequantization
        self.params_dtype = params_dtype

    def process_weights_after_loading(self, layer: torch.nn.Module):
        """Pre-dequantize MoE weights to FP16 for faster inference."""

        if hasattr(layer, "materialize_gguf_weights"):
            layer.materialize_gguf_weights()

        # Check if weights are actually loaded (not still UninitializedParameter/empty)
        w13_qweight = layer.w13_qweight
        w13_qtype = layer.w13_qweight_type.weight_type

        # Pre-dequantize w13 weights (gate+up projections)
        if w13_qtype not in UNQUANTIZED_TYPES:
            num_experts = w13_qweight.shape[0]
            w13_dequant_list = []

            block_size, type_size = gguf.GGML_QUANT_SIZES[w13_qtype]

            for e in range(num_experts):
                qweight_cpu = w13_qweight[e].cpu().numpy()
                rows = w13_qweight[e].shape[0]
                cols = w13_qweight[e].shape[1] // type_size * block_size

                dequant_np = gguf_dequantize(qweight_cpu.flatten(), w13_qtype)
                dequant = (
                    torch.from_numpy(dequant_np)
                    .to(dtype=self.params_dtype, device=w13_qweight.device)
                    .reshape(rows, cols)
                    .transpose(-1, -2)
                    .contiguous()
                )
                w13_dequant_list.append(dequant)

            w13_full = torch.stack(w13_dequant_list, dim=0)

            layer.register_buffer("w13_dequant", w13_full, persistent=False)
        else:
            layer.register_buffer("w13_dequant", w13_qweight.data, persistent=False)

        # Pre-dequantize w2 weights (down projection)
        w2_qweight = layer.w2_qweight
        w2_qtype = layer.w2_qweight_type.weight_type

        if w2_qtype not in UNQUANTIZED_TYPES:
            num_experts = w2_qweight.shape[0]
            w2_dequant_list = []

            block_size, type_size = gguf.GGML_QUANT_SIZES[w2_qtype]

            for e in range(num_experts):
                qweight_cpu = w2_qweight[e].cpu().numpy()
                rows = w2_qweight[e].shape[0]
                cols = w2_qweight[e].shape[1] // type_size * block_size

                dequant_np = gguf_dequantize(qweight_cpu.flatten(), w2_qtype)
                dequant = (
                    torch.from_numpy(dequant_np)
                    .to(dtype=self.params_dtype, device=w2_qweight.device)
                    .reshape(rows, cols)
                    .transpose(-1, -2)
                    .contiguous()
                )
                w2_dequant_list.append(dequant)

            w2_full = torch.stack(w2_dequant_list, dim=0)

            layer.register_buffer("w2_dequant", w2_full, persistent=False)
        else:
            layer.register_buffer("w2_dequant", w2_qweight.data, persistent=False)

        if hasattr(layer, "w2_qweight"):
            del layer.w2_qweight
        if hasattr(layer, "w13_qweight"):
            del layer.w13_qweight

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        """Apply MoE forward pass on NPU using npu_grouped_matmul for maximum performance."""
        from sglang.srt.distributed.communication_op import (
            tensor_model_parallel_all_gather,
        )
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights, topk_ids, _ = topk_output

        # Check if pre-dequantized weights are available
        use_pre_dequant = hasattr(layer, "w13_dequant") and hasattr(layer, "w2_dequant")

        if not use_pre_dequant:
            raise RuntimeError(
                "GGUF MoE on NPU requires pre-dequantization (FusedMoE fix). Please report if this occurs."
            )

        w13 = layer.w13_dequant
        w2 = layer.w2_dequant

        num_experts = w13.shape[0]

        tp_size = getattr(layer, "moe_tp_size", 1)

        original_dtype = x.dtype
        num_tokens = x.shape[0]
        top_k = topk_ids.shape[1]

        # Ensure correct dtypes for NPU ops
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = topk_weights.to(x.dtype)

        #  MoE routing initialization - reorder tokens by expert
        row_idx_len = num_tokens * top_k
        row_idx = (
            torch.arange(0, row_idx_len, dtype=torch.int32, device=x.device)
            .view(top_k, -1)
            .permute(1, 0)
            .contiguous()
        )

        sorted_hidden_states, expanded_row_idx, expanded_expert_idx = (
            torch.ops.npu.npu_moe_init_routing(
                x, row_idx=row_idx, expert_idx=topk_ids, active_num=num_tokens
            )
        )

        # Compute tokens per expert
        expert_tokens = torch.ops.npu.npu_moe_compute_expert_tokens(
            expanded_expert_idx, num_experts
        )
        expert_tokens = expert_tokens.to(torch.int64)

        w13_gmm = w13  # No transpose needed

        hidden_states = torch.ops.npu.npu_grouped_matmul(
            x=[sorted_hidden_states],
            weight=[w13_gmm],
            split_item=2,
            group_list_type=0,
            group_type=0,
            group_list=expert_tokens,
            output_dtype=original_dtype,
        )[0]

        #  Activation (SwiGLU)
        hidden_states = torch.ops.npu.npu_swiglu(hidden_states)

        # TP all-gather for intermediate dimension if needed
        if tp_size > 1:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, dim=-1)

        w2_gmm = w2

        hidden_states = torch.ops.npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[w2_gmm],
            split_item=2,
            group_list_type=0,
            group_type=0,
            group_list=expert_tokens,
            output_dtype=original_dtype,
        )[0]

        # Finalize routing - reorder back and apply weights
        final_hidden_states = torch.ops.npu.npu_moe_finalize_routing(
            hidden_states,
            skip1=None,
            skip2=None,
            bias=None,
            scales=topk_weights,
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=topk_ids,
        )

        if tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, dim=-1
            )

        # Ensure output matches input dtype
        final_hidden_states = final_hidden_states.to(dtype=original_dtype)

        return StandardCombineInput(hidden_states=final_hidden_states)


class GGUFEmbeddingAscendMethod(GGUFLinearAscendMethod):
    """Embedding method for GGUF on Ascend NPU."""

    def embedding(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        return torch.embedding(layer.dequantized_weight, x)
