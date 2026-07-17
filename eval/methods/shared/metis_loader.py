#!/usr/bin/env python3
"""Shared checkpoint-loading helpers for Metis evaluation runners."""

from __future__ import annotations

import gc
import json
import os
import re
import types
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_MODEL_ROOTS = [REPO_ROOT / "artifacts/models"] + [
    Path(item).expanduser()
    for item in os.environ.get("METIS_BASE_MODEL_ROOTS", "").split(os.pathsep)
    if item
]
PUBLIC_BASE_MODELS = {
    "Qwen3.5-4B": "Qwen/Qwen3.5-4B",
    "Qwen3.5-9B": "Qwen/Qwen3.5-9B",
    "Qwen3.5-27B": "Qwen/Qwen3.5-27B",
    "Qwen3-4B-Instruct-2507": "Qwen/Qwen3-4B-Instruct-2507",
}

from metis.configuration_metis import MetisConfig  # noqa: E402
from metis.modeling_metis import MetisForCausalLM  # noqa: E402


DEPRECATED_CHECKPOINT_ALIASES = {
    "v2.1_freezeq",
    "v2.1_trainq",
    "v2.2_trainq_nongdn",
    "v2.3_trainq_nongdn",
}


def resolve_checkpoint(checkpoint: str) -> Path:
    if checkpoint in DEPRECATED_CHECKPOINT_ALIASES:
        raise ValueError(
            f"Deprecated Metis checkpoint alias {checkpoint!r}. "
            "Pass an explicit checkpoint directory or public artifact identifier."
        )
    return Path(checkpoint).expanduser().resolve()


def parse_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def parse_max_memory(items: list[str] | None) -> dict[int | str, str] | None:
    if not items:
        return None
    out: dict[int | str, str] = {}
    for item in items:
        if ":" not in item:
            raise ValueError(f"Bad --max-memory item {item!r}; expected DEVICE:VALUE, e.g. 0:75GiB")
        key, value = item.split(":", 1)
        out[int(key) if key.isdigit() else key] = value
    return out


def _chat_template(tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": False,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        return tokenizer.apply_chat_template(messages, **kwargs)


def _normalize_cuda_device(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("empty device entry")
    if value.isdigit():
        return f"cuda:{value}"
    return value


def parse_model_parallel_devices(value: str | None, fallback_device: str) -> list[str]:
    if value:
        return [_normalize_cuda_device(item) for item in value.split(",") if item.strip()]
    if fallback_device.startswith("cuda") and torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        return ["cuda:0", "cuda:1"]
    return [fallback_device]


def _text_config(config: MetisConfig) -> Any:
    return getattr(config.backbone_configs, "text_config", config.backbone_configs)


def _normalize_backbone_type_for_local_wrappers(config: MetisConfig) -> tuple[str | None, str | None]:
    backbone_meta = getattr(config, "backbone_meta", None)
    if not isinstance(backbone_meta, dict):
        return None, None
    original = backbone_meta.get("backbone_type")
    aliases = {
        "qwen3": "Qwen3",
        "qwen3_5": "Qwen3_5",
    }
    if isinstance(original, str) and original in aliases:
        backbone_meta["backbone_type"] = aliases[original]
    resolved = backbone_meta.get("backbone_type")
    return original if isinstance(original, str) else None, resolved if isinstance(resolved, str) else None


def _checkpoint_state_path(ckpt_dir: Path) -> tuple[Path, str]:
    full_path = ckpt_dir / "model.safetensors"
    if full_path.exists():
        return full_path, "full"
    delta_path = ckpt_dir / "metis_delta.safetensors"
    if delta_path.exists():
        return delta_path, "delta"
    raise FileNotFoundError(
        f"Metis checkpoint {ckpt_dir} has neither model.safetensors nor metis_delta.safetensors"
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_base_paths(raw_path: str | None) -> list[Path]:
    candidates: list[Path] = []
    if raw_path:
        raw = Path(raw_path).expanduser()
        candidates.append(raw)
        for root in BASE_MODEL_ROOTS:
            candidates.append(root / raw.name)
    return candidates


def _base_model_path_from_checkpoint(ckpt_dir: Path, config: MetisConfig) -> str | Path:
    raw_paths: list[str] = []
    manifest_path = ckpt_dir / "metis_delta_manifest.json"
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        if manifest.get("base_model_path"):
            raw_paths.append(str(manifest["base_model_path"]))

    backbone_meta = getattr(config, "backbone_meta", None) or {}
    if isinstance(backbone_meta, dict) and backbone_meta.get("backbone_path"):
        raw_paths.append(str(backbone_meta["backbone_path"]))

    name_or_path = getattr(getattr(config, "backbone_configs", None), "_name_or_path", None)
    if name_or_path:
        raw_paths.append(str(name_or_path))

    tried: list[str] = []
    for raw_path in raw_paths:
        for candidate in _candidate_base_paths(raw_path):
            tried.append(str(candidate))
            if candidate.exists():
                return candidate.resolve()

    for raw_path in raw_paths:
        public_id = PUBLIC_BASE_MODELS.get(Path(raw_path).name)
        if public_id:
            return public_id

    raise FileNotFoundError(
        f"Could not resolve base model for delta checkpoint {ckpt_dir}. Tried local paths: {tried}"
    )


def _load_backbone_weights_into_metis(
    model: MetisForCausalLM,
    base_model_path: str | Path,
    dtype: torch.dtype,
) -> dict[str, Any]:
    backbone = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        trust_remote_code=True,
        dtype=dtype,
    )
    metis_backbone = model.model.metis_backbone
    metis_backbone.model.load_state_dict(backbone.model.state_dict(), strict=True)
    metis_backbone.lm_head.load_state_dict(backbone.lm_head.state_dict(), strict=True)
    report = {
        "base_model_path": str(base_model_path),
        "base_model_class": backbone.__class__.__name__,
    }
    del backbone
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def _validate_delta_manifest(ckpt_dir: Path, state_keys: set[str]) -> dict[str, Any]:
    manifest_path = ckpt_dir / "metis_delta_manifest.json"
    if not manifest_path.exists():
        return {
            "delta_manifest": None,
            "delta_manifest_missing": [],
            "delta_manifest_extra": [],
        }
    manifest = _read_json(manifest_path)
    manifest_keys = set((manifest.get("tensors") or {}).keys())
    missing = sorted(manifest_keys - state_keys)
    extra = sorted(state_keys - manifest_keys)
    if missing or extra:
        raise RuntimeError(
            f"Delta checkpoint manifest mismatch for {ckpt_dir}: "
            f"missing={missing[:10]} extra={extra[:10]}"
        )
    return {
        "delta_manifest": str(manifest_path),
        "delta_manifest_format": manifest.get("format"),
        "delta_manifest_base_model_path": manifest.get("base_model_path"),
        "delta_manifest_tensor_count": len(manifest_keys),
        "delta_manifest_num_parameters": manifest.get("num_parameters"),
        "delta_manifest_missing": [],
        "delta_manifest_extra": [],
    }


def infer_metis_model_family(ckpt_dir: str | Path, config: MetisConfig) -> str:
    """Infer the Metis size family used for device policy checks."""

    path_text = str(ckpt_dir).lower()
    for family in ("27b", "9b", "4b"):
        if re.search(rf"(?<![a-z0-9]){family}(?![a-z0-9])", path_text):
            return family

    text_cfg = _text_config(config)
    hidden_size = int(getattr(text_cfg, "hidden_size", 0) or 0)
    num_layers = int(getattr(text_cfg, "num_hidden_layers", 0) or 0)
    if hidden_size and num_layers:
        if hidden_size <= 3072 and num_layers <= 40:
            return "4b"
        if hidden_size == 4096 and num_layers <= 40:
            return "9b"
        if hidden_size >= 5120 or num_layers > 40:
            return "27b"
    return "unknown"


def enforce_metis_device_policy(
    ckpt_dir: str | Path,
    config: MetisConfig,
    *,
    device_map: str,
    model_parallel_devices: list[str] | None = None,
) -> str:
    """Enforce the paper evaluation policy: 4B/9B single GPU, 27B two GPU."""

    family = infer_metis_model_family(ckpt_dir, config)
    devices = model_parallel_devices or []
    if device_map == "paired_layers" and len(devices) != 2:
        raise ValueError(
            "Metis evaluation device policy requires paired_layers to use exactly two devices; "
            f"got {devices!r}. Use --model-parallel-devices cuda:0,cuda:1 for 27B, "
            "and --device-map single for 4B/9B."
        )
    if family in {"4b", "9b"} and device_map != "single":
        raise ValueError(
            f"Metis evaluation device policy requires Metis {family.upper()} to run single-GPU; "
            f"got --device-map {device_map!r}. Use --device-map single."
        )
    if family == "27b" and device_map != "paired_layers":
        raise ValueError(
            "Metis evaluation device policy requires Metis 27B to run on exactly two GPUs with "
            "--device-map paired_layers --model-parallel-devices cuda:0,cuda:1."
        )
    return family


def build_paired_layer_device_map(config: MetisConfig, devices: list[str]) -> dict[str, str]:
    if len(devices) < 2:
        raise ValueError("paired_layers device_map requires at least two devices")
    text_cfg = _text_config(config)
    num_layers = int(text_cfg.num_hidden_layers)
    device_map: dict[str, str] = {
        "model.metis_backbone.model.embed_tokens": devices[0],
        "model.metis_backbone.model.rotary_emb": devices[0],
        "model.metis_backbone.model.norm": devices[-1],
        "model.metis_backbone.lm_head": devices[-1],
    }
    for layer_idx in range(num_layers):
        device = devices[min((layer_idx * len(devices)) // num_layers, len(devices) - 1)]
        device_map[f"model.metis_backbone.model.layers.{layer_idx}"] = device
        device_map[f"model.metis_blocks.{layer_idx}"] = device
    return device_map


def _stringify_device_map(device_map: Any) -> Any:
    if isinstance(device_map, dict):
        return {str(key): str(value) for key, value in device_map.items()}
    return device_map


def _patch_model_parallel_commit_memory(model: MetisForCausalLM) -> None:
    def _commit_memory(self: MetisForCausalLM, outputs: Any, attention_mask: torch.Tensor | None = None) -> None:
        offset = self.config.memory_configs.get("commit_hidden_offset", 0)
        if offset not in (0, 1):
            raise ValueError(f"commit_hidden_offset must be 0 or 1, got {offset!r}")
        all_hidden = outputs.hidden_states
        for k, layer in enumerate(self.model.metis_blocks):
            if layer.local_memory is None:
                continue
            layer_h = all_hidden[k + offset]
            layer_attention_mask = attention_mask
            if layer_attention_mask is not None and layer_attention_mask.device != layer_h.device:
                layer_attention_mask = layer_attention_mask.to(layer_h.device)
            layer.hyper_memory.update_local_memory(
                layer_h,
                layer.local_memory,
                attention_mask=layer_attention_mask,
            )

    model._commit_memory = types.MethodType(_commit_memory, model)


def _dispatch_model(
    model: MetisForCausalLM,
    config: MetisConfig,
    *,
    ckpt_dir: str | Path,
    device: str,
    dtype: torch.dtype,
    device_map: str,
    model_parallel_devices: str | None,
    max_memory: dict[int | str, str] | None,
) -> tuple[MetisForCausalLM, dict[str, Any]]:
    paired_devices = None
    if device_map == "paired_layers":
        paired_devices = parse_model_parallel_devices(model_parallel_devices, device)
    model_family = enforce_metis_device_policy(
        ckpt_dir,
        config,
        device_map=device_map,
        model_parallel_devices=paired_devices,
    )

    if device_map == "single":
        model.to(device=device, dtype=dtype)
        return model, {
            "model_parallel": False,
            "model_family": model_family,
            "device_map": "single",
            "device": device,
            "max_memory": _stringify_device_map(max_memory),
            "hf_device_map": None,
        }

    from accelerate import dispatch_model, infer_auto_device_map

    model.to(dtype=dtype)
    no_split = ["MetisBlock", "Qwen3_5DecoderLayer", "Qwen3DecoderLayer"]
    if device_map == "paired_layers":
        devices = paired_devices or parse_model_parallel_devices(model_parallel_devices, device)
        resolved_map = build_paired_layer_device_map(config, devices)
        dispatch_note = "paired backbone decoder layers and Metis blocks by index"
    elif device_map == "auto":
        resolved_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=no_split,
            dtype=dtype,
        )
        dispatch_note = "accelerate inferred device_map; Metis block/backbone co-location is not guaranteed"
    else:
        raise ValueError(f"Unsupported Metis device_map: {device_map}")

    model = dispatch_model(model, device_map=resolved_map, force_hooks=True)
    _patch_model_parallel_commit_memory(model)
    return model, {
        "model_parallel": True,
        "model_family": model_family,
        "device_map": device_map,
        "device": device,
        "model_parallel_devices": parse_model_parallel_devices(model_parallel_devices, device),
        "max_memory": _stringify_device_map(max_memory),
        "hf_device_map": _stringify_device_map(getattr(model, "hf_device_map", resolved_map)),
        "dispatch_note": dispatch_note,
    }


def infer_input_device(model: Any, fallback: str = "cuda:0") -> str:
    try:
        return str(model.model.metis_backbone.model.embed_tokens.weight.device)
    except Exception:
        pass
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for key in (
            "model.metis_backbone.model.embed_tokens",
            "model.metis_backbone.model",
            "model.metis_backbone",
            "model",
            "",
        ):
            if key in hf_device_map:
                return str(hf_device_map[key])
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return fallback


def load_v2_full_checkpoint(
    ckpt_dir: str | Path,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    *,
    device_map: str = "single",
    model_parallel_devices: str | None = None,
    max_memory: dict[int | str, str] | None = None,
) -> tuple[MetisForCausalLM, Any, dict[str, Any]]:
    ckpt_dir = resolve_checkpoint(str(ckpt_dir))
    config = MetisConfig.from_pretrained(ckpt_dir)
    original_backbone_type, resolved_backbone_type = _normalize_backbone_type_for_local_wrappers(config)
    model = MetisForCausalLM(config)

    state_path, checkpoint_format = _checkpoint_state_path(ckpt_dir)
    base_report: dict[str, Any] = {}
    if checkpoint_format == "delta":
        base_path = _base_model_path_from_checkpoint(ckpt_dir, config)
        base_report = _load_backbone_weights_into_metis(model, base_path, dtype)

    state = load_file(state_path, device="cpu")
    remapped: dict[str, torch.Tensor] = {}
    outer_remapped_count = 0
    backbone_remapped_count = 0
    for key, value in state.items():
        new_key = key
        if key.startswith("model.language_model."):
            new_key = "model." + key[len("model.language_model.") :]
            outer_remapped_count += 1
        backbone_prefix = "model.metis_backbone.model.language_model."
        if new_key.startswith(backbone_prefix):
            new_key = "model.metis_backbone.model." + new_key[len(backbone_prefix) :]
            backbone_remapped_count += 1
        remapped[new_key] = value

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    state_key_count = len(state)
    delta_manifest_report: dict[str, Any] = {}
    if checkpoint_format == "delta":
        delta_manifest_report = _validate_delta_manifest(ckpt_dir, set(remapped))
        if unexpected:
            raise RuntimeError(f"Unexpected keys while loading delta checkpoint {ckpt_dir}: {list(unexpected)[:50]}")
    del state
    del remapped
    gc.collect()

    important_missing = [
        key
        for key in missing
        if (
            "metis_blocks" in key
            or "hyper_memory" in key
            or "query_proj" in key
            or "query_norm" in key
        )
    ]

    model, dispatch_report = _dispatch_model(
        model,
        config,
        ckpt_dir=ckpt_dir,
        device=device,
        dtype=dtype,
        device_map=device_map,
        model_parallel_devices=model_parallel_devices,
        max_memory=max_memory,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    report = {
        "checkpoint_dir": str(ckpt_dir),
        "checkpoint_format": checkpoint_format,
        "backbone_type_original": original_backbone_type,
        "backbone_type_resolved": resolved_backbone_type,
        "state_path": str(state_path),
        "state_keys": state_key_count,
        "outer_remapped_keys": outer_remapped_count,
        "backbone_remapped_keys": backbone_remapped_count,
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "important_missing": important_missing[:50],
        "unexpected": list(unexpected)[:50],
        "device": device,
        "input_device": infer_input_device(model, device),
        "dtype": str(dtype),
        "tokenizer": tokenizer.__class__.__name__,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    report.update(base_report)
    report.update(delta_manifest_report)
    report.update(dispatch_report)
    return model, tokenizer, report
