# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Config loading utilities."""

from pathlib import Path
from typing import Optional

from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from sglang.srt.configs.model_config_parser_registry import (
    ModelConfigParserBase,
    get_model_config_parser,
    register_model_config_parser,
)
from sglang.srt.connector import create_remote_connector
from sglang.srt.utils import is_remote_url, lru_cache_frozenset

from ..hf_transformers_patches import _ensure_gguf_version
from .common import (
    _CONFIG_REGISTRY,
    AutoConfig,
    DeepseekVLV2Config,
    _is_deepseek_ocr2_model,
    _is_deepseek_ocr_model,
    _override_v_head_dim_if_zero,
    check_gguf_file,
    get_hf_text_config,
    resolve_runai_obj_uri,
)
from .mistral_utils import is_mistral_model, load_mistral_config


def _set_architectures(config, arch_name):
    config.update({"architectures": [arch_name]})


def _apply_deepseek_ocr_overrides(config, model):
    _override_v_head_dim_if_zero(config)
    _set_architectures(config, "DeepseekOCRForCausalLM")
    config._name_or_path = model


def _gguf_field_value(reader, key: str, default=None):
    field = reader.fields.get(key)
    if field is None:
        return default

    values = []
    for index in field.data:
        value = field.parts[index]
        if isinstance(value, bytes):
            values.append(value.decode("utf-8"))
        elif hasattr(value, "dtype"):
            if str(value.dtype) == "uint8" and getattr(value, "ndim", 0) == 1:
                values.append(bytes(value).decode("utf-8"))
            elif getattr(value, "shape", ()) == (1,):
                values.append(value.item())
            else:
                values.append(value.tolist())
        else:
            values.append(value)

    if len(values) == 1:
        return values[0]
    return values


def _synthesize_deepseek_v4_gguf_config(gguf_file: str):
    import gguf

    from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config

    reader = gguf.GGUFReader(gguf_file)

    def get(name: str, default=None):
        return _gguf_field_value(reader, name, default)

    def scalar(name: str, default=None):
        value = get(name, default)
        if isinstance(value, list):
            return value[0] if value else default
        return value

    block_count = int(scalar("deepseek4.block_count", 43))
    rope_dim = int(scalar("deepseek4.rope.dimension_count", 64))
    head_dim = int(scalar("deepseek4.attention.key_length", 512))
    compress_ratios = get("deepseek4.attention.compress_ratios", None)
    if compress_ratios is None:
        compress_ratios = [
            0 if i < 2 else (4 if i % 2 == 0 else 128)
            for i in range(block_count)
        ]
    compress_ratios = [int(x) for x in compress_ratios[:block_count]]
    if len(compress_ratios) != block_count:
        raise ValueError(
            "DeepSeek V4 GGUF compress ratios length "
            f"{len(compress_ratios)} does not match block_count={block_count}"
        )

    rope_scaling_type = scalar("deepseek4.rope.scaling.type", "yarn")
    rope_scaling = {
        "type": rope_scaling_type,
        "rope_type": rope_scaling_type,
        "factor": float(scalar("deepseek4.rope.scaling.factor", 16.0)),
        "original_max_position_embeddings": int(
            scalar("deepseek4.rope.scaling.original_context_length", 65536)
        ),
        "beta_fast": float(scalar("deepseek4.rope.scaling.yarn_beta_fast", 32.0)),
        "beta_slow": float(scalar("deepseek4.rope.scaling.yarn_beta_slow", 1.0)),
    }

    config = DeepSeekV4Config(
        architectures=["DeepseekV4ForCausalLM"],
        hidden_size=int(scalar("deepseek4.embedding_length", 4096)),
        vocab_size=int(scalar("deepseek4.vocab_size", 129280)),
        max_position_embeddings=int(scalar("deepseek4.context_length", 1048576)),
        num_hidden_layers=block_count,
        n_routed_experts=int(scalar("deepseek4.expert_count", 256)),
        num_experts_per_tok=int(scalar("deepseek4.expert_used_count", 6)),
        n_shared_experts=int(scalar("deepseek4.expert_shared_count", 1)),
        moe_intermediate_size=int(
            scalar("deepseek4.expert_feed_forward_length", 2048)
        ),
        intermediate_size=int(scalar("deepseek4.expert_feed_forward_length", 2048)),
        num_attention_heads=int(scalar("deepseek4.attention.head_count", 64)),
        num_key_value_heads=int(scalar("deepseek4.attention.head_count_kv", 1)),
        q_lora_rank=int(scalar("deepseek4.attention.q_lora_rank", 1024)),
        kv_lora_rank=head_dim,
        v_head_dim=int(scalar("deepseek4.attention.value_length", head_dim)),
        qk_nope_head_dim=head_dim - rope_dim,
        qk_rope_head_dim=rope_dim,
        o_lora_rank=int(scalar("deepseek4.attention.output_lora_rank", 1024)),
        o_groups=int(scalar("deepseek4.attention.output_group_count", 8)),
        window_size=int(scalar("deepseek4.attention.sliding_window", 128)),
        index_n_heads=int(scalar("deepseek4.attention.indexer.head_count", 64)),
        index_head_dim=int(scalar("deepseek4.attention.indexer.key_length", 128)),
        index_topk=int(scalar("deepseek4.attention.indexer.top_k", 512)),
        rms_norm_eps=float(
            scalar("deepseek4.attention.layer_norm_rms_epsilon", 1e-6)
        ),
        rope_theta=float(scalar("deepseek4.rope.freq_base", 10000.0)),
        compress_rope_theta=float(
            scalar("deepseek4.attention.compress_rope_freq_base", 160000.0)
        ),
        rope_scaling=rope_scaling,
        compress_ratios=compress_ratios,
        quantization_config={},
        n_hash_layers=int(scalar("deepseek4.hash_layer_count", 3)),
        hc_mult=int(scalar("deepseek4.hyper_connection.count", 4)),
        hc_sinkhorn_iters=int(
            scalar("deepseek4.hyper_connection.sinkhorn_iterations", 20)
        ),
        hc_eps=float(scalar("deepseek4.hyper_connection.epsilon", 1e-6)),
        routed_scaling_factor=float(scalar("deepseek4.expert_weights_scale", 1.5)),
        scoring_func="sqrtsoftplus",
        topk_group=int(scalar("deepseek4.attention.output_group_count", 8)),
        n_group=int(scalar("deepseek4.attention.output_group_count", 8)),
        bos_token_id=int(scalar("tokenizer.ggml.bos_token_id", 0)),
        eos_token_id=int(scalar("tokenizer.ggml.eos_token_id", 1)),
        tie_word_embeddings=False,
    )
    config.head_dim = head_dim
    config.sliding_window = config.window_size
    config.num_hash_layers = config.n_hash_layers
    config.swiglu_limit = float(scalar("deepseek4.swiglu_clamp_exp", 10.0))
    config._name_or_path = str(Path(gguf_file).parent)
    return config


def _try_synthesize_gguf_config(gguf_file: str):
    import gguf

    reader = gguf.GGUFReader(gguf_file)
    arch = _gguf_field_value(reader, "general.architecture")
    if arch == "deepseek4":
        return _synthesize_deepseek_v4_gguf_config(gguf_file)
    return None


@register_model_config_parser("hf")
class HfModelConfigParser(ModelConfigParserBase):
    def parse(
        self,
        model,
        trust_remote_code: bool,
        revision: Optional[str] = None,
        **kwargs,
    ):
        config = AutoConfig.from_pretrained(
            model,
            trust_remote_code=trust_remote_code,
            revision=revision,
            **kwargs,
        )

        if (
            config.architectures is not None
            and config.architectures[0] == "Phi4MMForCausalLM"
        ):
            from transformers import SiglipVisionConfig

            config.vision_config = SiglipVisionConfig(
                hidden_size=1152,
                image_size=448,
                intermediate_size=4304,
                model_type="siglip_vision_model",
                num_attention_heads=16,
                num_hidden_layers=26,
                patch_size=14,
            )

        if config.architectures in [
            ["LongcatCausalLM"],
            ["LongcatFlashForCausalLM"],
            ["LongcatFlashNgramForCausalLM"],
        ]:
            config.model_type = "longcat_flash"

        text_config = get_hf_text_config(config=config)

        if isinstance(model, str) and text_config is not None:
            items = (
                text_config.items()
                if hasattr(text_config, "items")
                else vars(text_config).items()
            )
            for key, val in items:
                if not hasattr(config, key) and val is not None:
                    setattr(config, key, val)

        is_ocr = _is_deepseek_ocr_model(config)
        is_ocr2 = _is_deepseek_ocr2_model(config)

        if is_ocr2:
            _override_v_head_dim_if_zero(config)
            config.model_type = "deepseek-ocr"
            _set_architectures(config, "DeepseekOCRForCausalLM")
            config = DeepseekVLV2Config.from_pretrained(model, revision=revision)
            _apply_deepseek_ocr_overrides(config, model)
        elif config.model_type in _CONFIG_REGISTRY:
            model_type = config.model_type
            if model_type == "deepseek_vl_v2" and is_ocr:
                model_type = "deepseek-ocr"
            config = _CONFIG_REGISTRY[model_type].from_pretrained(
                model, revision=revision
            )

            # Re-check after reloading config from registry
            if _is_deepseek_ocr_model(config) or _is_deepseek_ocr2_model(config):
                _apply_deepseek_ocr_overrides(config, model)
            else:
                config._name_or_path = model

        if isinstance(model, str) and config.model_type == "internvl_chat":
            for key, val in config.llm_config.__dict__.items():
                if not hasattr(config, key):
                    setattr(config, key, val)

        if config.model_type == "multi_modality":
            _set_architectures(config, "MultiModalityCausalLM")

        if config.model_type in ("gemma4", "gemma4_assistant"):
            # Gemma4 configs use base attributes for SWA layers and `global_*`
            # variants for full-attention layers.  SGLang expects the opposite:
            # base = full-attention, `swa_*` = sliding-window overrides.
            text_config = config.text_config
            global_head_dim = getattr(text_config, "global_head_dim", None)
            global_kv_heads = getattr(text_config, "num_global_key_value_heads", None)

            swa_head_dim = text_config.head_dim
            swa_kv_heads = text_config.num_key_value_heads

            text_config.swa_head_dim = swa_head_dim
            text_config.swa_v_head_dim = swa_head_dim
            text_config.swa_num_key_value_heads = swa_kv_heads

            if global_head_dim is not None:
                text_config.head_dim = global_head_dim
            if global_kv_heads is not None:
                text_config.num_key_value_heads = global_kv_heads

            if not hasattr(text_config, "v_head_dim"):
                text_config.v_head_dim = text_config.head_dim
            if not hasattr(text_config, "swa_v_head_dim"):
                text_config.swa_v_head_dim = text_config.swa_head_dim

        if config.model_type == "longcat_flash":
            _set_architectures(config, "LongcatFlashForCausalLM")

        return config


@register_model_config_parser("mistral")
class MistralModelConfigParser(ModelConfigParserBase):
    def parse(
        self,
        model,
        trust_remote_code: bool,
        revision: Optional[str] = None,
        **kwargs,
    ):
        del kwargs
        return load_mistral_config(
            model, trust_remote_code=trust_remote_code, revision=revision
        )


@lru_cache_frozenset(maxsize=32)
def get_config(
    model: str,
    trust_remote_code: bool,
    revision: Optional[str] = None,
    model_override_args: Optional[dict] = None,
    model_config_parser: str = "auto",
    **kwargs,
):
    is_gguf = check_gguf_file(model)
    if is_gguf:
        if model_config_parser not in ("auto", "hf"):
            raise ValueError(
                f"model_config_parser={model_config_parser!r} is incompatible "
                "with GGUF inputs; only 'hf' (or 'auto') is supported."
            )
        _ensure_gguf_version()
        config = _try_synthesize_gguf_config(str(model))
        if config is not None:
            if model_override_args:
                config.update(model_override_args)
            return config
        kwargs["gguf_file"] = model
        model = Path(model).parent
        # Skip auto-resolution for GGUF: the name-based Mistral heuristic
        # would misfire on the rewritten parent dir.
        model_config_parser = "hf"

    model = resolve_runai_obj_uri(model)

    if is_remote_url(model):
        client = create_remote_connector(model)
        client.pull_files(ignore_pattern=["*.pt", "*.safetensors", "*.bin"])
        model = client.get_local_dir()

    if model_config_parser == "auto":
        # `model` is post-rewrite (gguf parent / runai uri / remote pull).
        model_config_parser = "mistral" if is_mistral_model(model) else "hf"

    parser = get_model_config_parser(model_config_parser)
    try:
        config = parser.parse(
            model, trust_remote_code=trust_remote_code, revision=revision, **kwargs
        )
    except ValueError as exc:
        if (
            is_gguf
            and "GGUF model with architecture" in str(exc)
            and (Path(model) / "config.json").exists()
        ):
            fallback_kwargs = dict(kwargs)
            fallback_kwargs.pop("gguf_file", None)
            config = parser.parse(
                model,
                trust_remote_code=trust_remote_code,
                revision=revision,
                **fallback_kwargs,
            )
            if getattr(config, "model_type", None) in ("qwen3_5", "qwen3_5_moe"):
                config = config.text_config
        else:
            raise

    if model_override_args:
        config.update(model_override_args)

    if is_gguf:
        if config.model_type == "qwen3_5_moe_text":
            _set_architectures(config, "Qwen3_5MoeForCausalLM")
        elif config.model_type == "qwen3_5_text":
            _set_architectures(config, "Qwen3_5ForCausalLM")
        elif config.model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            raise RuntimeError(f"Can't get gguf config for {config.model_type}.")
        else:
            _set_architectures(
                config, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type]
            )

    return config
