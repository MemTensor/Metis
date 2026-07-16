"""Checkpoint helpers for Metis delta checkpoints.

The backbone is frozen during Metis training, so a normal HF checkpoint
mostly duplicates immutable backbone weights.  Delta checkpoints save only the
trainable Metis parameters plus the regular Trainer state files.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .configuration_metis import MetisConfig
from .modeling_metis import MetisForCausalLM
from .weight_utils import load_metis_from_backbone

logger = logging.getLogger(__name__)

DELTA_WEIGHTS_NAME = "metis_delta.safetensors"
DELTA_MANIFEST_NAME = "metis_delta_manifest.json"
DELTA_FORMAT_VERSION = "metis_delta_v1"

FULL_WEIGHT_FILES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


def is_delta_checkpoint(checkpoint_path: str | os.PathLike[str]) -> bool:
    path = Path(checkpoint_path)
    return (path / DELTA_MANIFEST_NAME).is_file() and (path / DELTA_WEIGHTS_NAME).is_file()


def is_full_checkpoint(checkpoint_path: str | os.PathLike[str]) -> bool:
    path = Path(checkpoint_path)
    return any((path / name).is_file() for name in FULL_WEIGHT_FILES)


def is_metis_checkpoint(checkpoint_path: str | os.PathLike[str]) -> bool:
    return is_delta_checkpoint(checkpoint_path) or is_full_checkpoint(checkpoint_path)


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _safe_dtype_name(tensor: torch.Tensor) -> str:
    return str(tensor.dtype).replace("torch.", "")


def _trainable_state_dict(model, state_dict: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
    unwrapped = _unwrap_model(model)
    source = state_dict if state_dict is not None else unwrapped.state_dict()
    trainable_names = [name for name, param in unwrapped.named_parameters() if param.requires_grad]

    delta: dict[str, torch.Tensor] = {}
    missing: list[str] = []
    for name in trainable_names:
        tensor = source.get(name)
        if tensor is None:
            tensor = source.get(f"module.{name}")
        if tensor is None:
            missing.append(name)
            continue
        delta[name] = tensor.detach().cpu()

    if missing:
        preview = ", ".join(missing[:10])
        raise KeyError(
            f"Could not find {len(missing)} trainable tensors in checkpoint state_dict: {preview}"
        )
    if not delta:
        raise RuntimeError("No trainable tensors found; refusing to write an empty Metis delta checkpoint.")
    return delta


def save_metis_delta_checkpoint(
    model,
    output_dir: str | os.PathLike[str],
    *,
    tokenizer=None,
    base_model_path: str = "",
    state_dict: dict[str, torch.Tensor] | None = None,
) -> None:
    """Save a compact checkpoint containing only trainable Metis weights."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    unwrapped = _unwrap_model(model)

    if base_model_path:
        meta = dict(getattr(unwrapped.config, "backbone_meta", {}) or {})
        meta["backbone_path"] = base_model_path
        unwrapped.config.backbone_meta = meta

    unwrapped.config.save_pretrained(output)
    generation_config = getattr(unwrapped, "generation_config", None)
    if generation_config is not None:
        generation_config.save_pretrained(output)
    if tokenizer is not None:
        tokenizer.save_pretrained(output)

    delta = _trainable_state_dict(unwrapped, state_dict=state_dict)
    save_file(delta, str(output / DELTA_WEIGHTS_NAME))

    manifest = {
        "format": DELTA_FORMAT_VERSION,
        "base_model_path": base_model_path or getattr(unwrapped.config, "backbone_meta", {}).get("backbone_path", ""),
        "backbone_type": getattr(unwrapped.config, "backbone_meta", {}).get("backbone_type", ""),
        "num_tensors": len(delta),
        "num_parameters": int(sum(t.numel() for t in delta.values())),
        "tensors": {
            name: {
                "shape": list(tensor.shape),
                "dtype": _safe_dtype_name(tensor),
            }
            for name, tensor in sorted(delta.items())
        },
    }
    with open(output / DELTA_MANIFEST_NAME, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        "Saved Metis delta checkpoint -> %s (%d tensors, %.2fM parameters)",
        output,
        len(delta),
        manifest["num_parameters"] / 1_000_000,
    )


def load_delta_manifest(checkpoint_path: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(checkpoint_path) / DELTA_MANIFEST_NAME
    with open(path) as f:
        manifest = json.load(f)
    if manifest.get("format") != DELTA_FORMAT_VERSION:
        raise ValueError(f"Unsupported Metis delta checkpoint format in {path}: {manifest.get('format')!r}")
    return manifest


def resolve_backbone_path(
    checkpoint_path: str | os.PathLike[str],
    *,
    model_path: str | None = None,
) -> str:
    """Resolve the immutable backbone path for a delta checkpoint.

    Explicit ``model_path`` wins.  ``METIS_BASE_MODEL_PATH`` is a convenient
    machine-local override when moving a delta checkpoint between machines.
    """
    if model_path:
        return model_path
    env_path = os.environ.get("METIS_BASE_MODEL_PATH") or os.environ.get("MODEL_PATH")
    if env_path:
        return env_path

    manifest = load_delta_manifest(checkpoint_path)
    if manifest.get("base_model_path"):
        return manifest["base_model_path"]

    config_path = Path(checkpoint_path) / "config.json"
    if config_path.is_file():
        with open(config_path) as f:
            config = json.load(f)
        backbone_path = (config.get("backbone_meta") or {}).get("backbone_path")
        if backbone_path:
            return backbone_path

    raise ValueError(
        f"Could not resolve backbone path for delta checkpoint {checkpoint_path}. "
        "Pass --model_path or set METIS_BASE_MODEL_PATH."
    )


def _memory_arg(memory: dict[str, Any], key: str, default: Any) -> Any:
    return memory.get(key, default)


def _build_metis_from_delta_config(
    checkpoint_path: str | os.PathLike[str],
    *,
    model_path: str | None,
    backbone_type: str | None,
    device: str | torch.device,
    dtype: torch.dtype,
) -> MetisForCausalLM:
    config = MetisConfig.from_pretrained(checkpoint_path)
    memory = dict(getattr(config, "memory_configs", {}) or {})
    backbone_meta = dict(getattr(config, "backbone_meta", {}) or {})
    base_model_path = resolve_backbone_path(checkpoint_path, model_path=model_path)
    resolved_backbone_type = backbone_type or backbone_meta.get("backbone_type") or "qwen3_5"

    model, _ = load_metis_from_backbone(
        base_model_path,
        backbone_type=resolved_backbone_type,
        device=device,
        dtype=dtype,
        metis_block_type=_memory_arg(memory, "metis_block_type", "NormedReweightLearnedQueryMetisBlock"),
        metis_hyper_memory_type=_memory_arg(memory, "metis_hyper_memory_type", "StraightThroughAlphaTopPGatedDeltaRuleMetisHyperMemory"),
        metis_local_memory_type=_memory_arg(memory, "metis_local_memory_type", "NormalizedDeltaNetMetisLocalMemory"),
        update_ratio=float(_memory_arg(memory, "update_ratio", 0.9)),
        commit_hidden_offset=int(_memory_arg(memory, "commit_hidden_offset", 0)),
        mem_norm_init=float(_memory_arg(memory, "mem_norm_init", 1.0)),
        uniform_num_selected=int(_memory_arg(memory, "uniform_num_selected", 16)),
        stride_interval=int(_memory_arg(memory, "stride_interval", 8)),
        pool_temperature=float(_memory_arg(memory, "pool_temperature", 1.0)),
        gumbel_topk_noise=bool(_memory_arg(memory, "gumbel_topk_noise", True)),
        alpha_top_p=float(_memory_arg(memory, "alpha_top_p", 0.9)),
        alpha_min_tokens=int(_memory_arg(memory, "alpha_min_tokens", 1)),
        alpha_max_tokens=int(_memory_arg(memory, "alpha_max_tokens", 0)),
        alpha_max_fraction=float(_memory_arg(memory, "alpha_max_fraction", 0.0)),
        gated_delta_alpha_init=float(_memory_arg(memory, "gated_delta_alpha_init", 1.0)),
        gated_delta_beta_init=float(_memory_arg(memory, "gated_delta_beta_init", 1.0)),
        qk_kernel_type=_memory_arg(memory, "qk_kernel_type", "elu_plus_one"),
        metis_reweight_gamma=float(_memory_arg(memory, "metis_reweight_gamma", 0.9)),
    )
    model.config.backbone_meta = {
        **backbone_meta,
        "backbone_type": resolved_backbone_type,
        "backbone_path": base_model_path,
    }
    return model


def load_metis_delta_into_model(
    model,
    checkpoint_path: str | os.PathLike[str],
) -> None:
    """Load trainable Metis weights from a delta checkpoint into an existing model."""
    checkpoint = Path(checkpoint_path)
    if not is_delta_checkpoint(checkpoint):
        raise ValueError(f"Not a Metis delta checkpoint: {checkpoint}")

    unwrapped = _unwrap_model(model)
    delta = load_file(str(checkpoint / DELTA_WEIGHTS_NAME), device="cpu")
    incompatible = unwrapped.load_state_dict(delta, strict=False)
    if incompatible.unexpected_keys:
        preview = ", ".join(incompatible.unexpected_keys[:10])
        raise RuntimeError(
            f"Delta checkpoint has {len(incompatible.unexpected_keys)} unexpected keys: {preview}"
        )
    logger.info("Loaded Metis delta weights from %s (%d tensors)", checkpoint, len(delta))


def load_metis_model_from_checkpoint(
    checkpoint_path: str | os.PathLike[str],
    *,
    model_path: str | None = None,
    backbone_type: str | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> MetisForCausalLM:
    """Load either a compact delta checkpoint or a legacy full checkpoint."""
    checkpoint = Path(checkpoint_path)
    if is_delta_checkpoint(checkpoint):
        model = _build_metis_from_delta_config(
            checkpoint,
            model_path=model_path,
            backbone_type=backbone_type,
            device=device,
            dtype=dtype,
        )
        load_metis_delta_into_model(model, checkpoint)
        return model.to(device=device, dtype=dtype)

    if is_full_checkpoint(checkpoint):
        model = MetisForCausalLM.from_pretrained(checkpoint, dtype=dtype)
        return model.to(device=device, dtype=dtype)

    raise ValueError(
        f"{checkpoint} is neither a Metis delta checkpoint nor a legacy full checkpoint."
    )
