"""Shared utilities for Metis training.

Covers:
  - File logging setup
  - Model setup (freeze_backbone, count_params)
  - DataLoader debugging (dump_sampler_tree)
  - Code snapshots (save_code_snapshot)
  - LoRA adapters
  - Trainer callbacks (FileLoggingCallback)
  - MasterWeightAdamW (fp32 master weights for bf16)
"""

from __future__ import annotations

import logging
import math
import os
import shutil
from collections import Counter

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  Logging
# ═══════════════════════════════════════════════════════════════════

def setup_file_logging(log: logging.Logger, output_dir: str, log_file: str) -> str:
    """Attach a FileHandler to *log* (idempotent)."""
    log_path = os.path.join(output_dir, log_file)
    abs_path = os.path.abspath(log_path)
    for h in log.handlers:
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == abs_path:
            return abs_path
    fh = logging.FileHandler(abs_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S",
    ))
    log.addHandler(fh)
    return abs_path


class FileLoggingCallback(TrainerCallback):
    """Write Trainer step-level logs to the Python logger (rank-0 only)."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs:
            msg = "  ".join(
                f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in logs.items()
                if k != "total_flos"
            )
            logger.info(msg)


# ═══════════════════════════════════════════════════════════════════
#  Model setup
# ═══════════════════════════════════════════════════════════════════

def is_full_attention(block) -> bool:
    return getattr(block.backbone_decoder.raw_decoder, "layer_type", "full_attention") == "full_attention"


def freeze_backbone(model: nn.Module) -> int:
    """Freeze backbone, then unfreeze memory modules and learned read-query layers."""
    for p in model.parameters():
        p.requires_grad = False
    count = 0
    for block in model.model.metis_blocks:
        if not is_full_attention(block):
            continue
        hm = getattr(block, "hyper_memory", None)
        if hm is None:
            continue
        for p in hm.parameters():
            p.requires_grad = True
            count += 1
        mn = getattr(block, "mem_norm", None)
        if mn is not None:
            for p in mn.parameters():
                p.requires_grad = True
                count += 1
        for name in ("query_proj", "query_norm"):
            module = getattr(block, name, None)
            if module is not None:
                for p in module.parameters():
                    p.requires_grad = True
                    count += 1
    return count


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ═══════════════════════════════════════════════════════════════════
#  DataLoader debugging
# ═══════════════════════════════════════════════════════════════════

def dump_sampler_tree(dl) -> None:
    print("DataLoader.sampler      :", type(getattr(dl, "sampler", None)).__name__)
    obj = getattr(dl, "batch_sampler", None)
    level = 0
    while obj is not None:
        print(f"level {level} batch_sampler:", type(obj).__name__)
        next_obj = None
        for name in ["batch_sampler", "sampler"]:
            if hasattr(obj, name):
                child = getattr(obj, name)
                print(f"  └─ {name}: {type(child).__name__}")
                if next_obj is None and child is not None and child is not obj:
                    next_obj = child
        obj = next_obj if next_obj is not None and "BatchSampler" in type(next_obj).__name__ else None
        level += 1


# ═══════════════════════════════════════════════════════════════════
#  Code snapshot
# ═══════════════════════════════════════════════════════════════════

_SNAPSHOT_EXCLUDE_DIRS = {
    "experiments", "wandb", "__pycache__", ".git",
    "pretrained_models", "data", "checkpoint", "log",
}

def save_code_snapshot(output_dir: str, project_root: str | None = None) -> None:
    """Copy all .py / .sh source files to <output_dir>/code_snapshot/."""
    if project_root is None:
        project_root = os.path.dirname(os.path.abspath(__file__))

    snap_dir = os.path.join(output_dir, "code_snapshot")
    os.makedirs(snap_dir, exist_ok=True)

    copied = 0
    for dirpath, dirnames, filenames in os.walk(project_root):
        dirnames[:] = [d for d in dirnames
                       if d not in _SNAPSHOT_EXCLUDE_DIRS and not d.startswith(".")]
        for fname in filenames:
            if not fname.endswith((".py", ".sh")):
                continue
            src = os.path.join(dirpath, fname)
            rel = os.path.relpath(src, project_root)
            dst = os.path.join(snap_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1

    logger.info(f"Code snapshot saved → {snap_dir}  ({copied} files)")


# ═══════════════════════════════════════════════════════════════════
#  LoRA
# ═══════════════════════════════════════════════════════════════════

class LoRALinear(nn.Module):
    def __init__(self, original: nn.Linear, r: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        self.original = original
        self.scaling = alpha / r
        device, dtype = original.weight.device, original.weight.dtype
        in_f, out_f = original.in_features, original.out_features
        self.lora_A = nn.Linear(in_f, r, bias=False, device=device, dtype=dtype)
        self.lora_B = nn.Linear(r, out_f, bias=False, device=device, dtype=dtype)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        for p in original.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original(x) + self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling


def apply_lora(model: nn.Module, target_names: list[str], r: int, alpha: int,
               dropout: float = 0.0) -> list[str]:
    replaced: list[str] = []
    for parent_name, parent_module in list(model.named_modules()):
        for child_name in list(parent_module._modules.keys()):
            child = parent_module._modules[child_name]
            if isinstance(child, nn.Linear) and child_name in target_names:
                full_path = f"{parent_name}.{child_name}" if parent_name else child_name
                setattr(parent_module, child_name, LoRALinear(child, r, alpha, dropout))
                replaced.append(full_path)
    return replaced


def collect_lora_stats(model: nn.Module) -> dict[str, float]:
    a_norms, b_norms = [], []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            a_norms.append(module.lora_A.weight.data.norm().item())
            b_norms.append(module.lora_B.weight.data.norm().item())
    if not a_norms:
        return {}
    return {
        "weights/lora_A_norm_avg": sum(a_norms) / len(a_norms),
        "weights/lora_B_norm_avg": sum(b_norms) / len(b_norms),
    }


# ═══════════════════════════════════════════════════════════════════
#  MasterWeightAdamW  (fp32 master weights for bf16 params)
# ═══════════════════════════════════════════════════════════════════

class MasterWeightAdamW(AdamW):
    """AdamW with fp32 master weights for bf16 parameters.

    Keeps model parameters in bf16 for forward/backward while maintaining
    fp32 copies inside the optimizer for precise updates.
    """

    def __init__(self, params: list[nn.Parameter] | list[dict], **adam_kwargs):
        params = list(params)
        self.bf16_params: list[nn.Parameter] = []
        self._bf16_to_fp32: dict[int, torch.Tensor] = {}

        def clone_master(param: nn.Parameter) -> torch.Tensor:
            master = param.detach().float().clone()
            master.requires_grad = False
            self.bf16_params.append(param)
            self._bf16_to_fp32[id(param)] = master
            return master

        if params and isinstance(params[0], dict):
            fp32_groups = []
            for group in params:
                bf16_ps = list(group["params"])
                fp32_ps = [clone_master(p) for p in bf16_ps]
                fp32_group = {k: v for k, v in group.items() if k != "params"}
                fp32_group["params"] = fp32_ps
                fp32_groups.append(fp32_group)
        else:
            fp32_groups = [clone_master(p) for p in params]

        super().__init__(fp32_groups, **adam_kwargs)

        n_params = sum(p.numel() for p in self.bf16_params)
        n_groups = len(self.param_groups)
        group_info = ", ".join(
            f"g{i}({sum(p.numel() for p in g['params']):,} params, lr={g['lr']:.1e})"
            for i, g in enumerate(self.param_groups)
        )
        logger.info(
            "MasterWeightAdamW: %d groups, %d total params [%s]",
            n_groups,
            n_params,
            group_info,
        )

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for bp in self.bf16_params:
            mp = self._bf16_to_fp32[id(bp)]
            if bp.grad is not None:
                mp.grad = bp.grad.detach().float()
            else:
                mp.grad = None

        super().step()

        for bp in self.bf16_params:
            mp = self._bf16_to_fp32[id(bp)]
            bp.data.copy_(mp.data)
        return loss

    def zero_grad(self, set_to_none: bool = True):
        for bp in self.bf16_params:
            if set_to_none:
                bp.grad = None
            elif bp.grad is not None:
                bp.grad.zero_()
        super().zero_grad(set_to_none=set_to_none)
