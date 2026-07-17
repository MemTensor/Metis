#!/usr/bin/env python3
"""Low-rank projection helpers for Metis LocalMemory states.

This module is intentionally repository-local.  It does not modify the Metis
source tree; runners opt in by calling the projector after commit and/or before
query.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


LOW_RANK_POLICIES = {
    "none",
    "after_each_commit",
    "before_query",
    "after_each_commit_and_before_query",
}
LOW_RANK_TARGETS = {"state", "state_and_key"}


@dataclass(frozen=True)
class LowRankLocalMemoryConfig:
    enabled: bool = False
    rank: int | None = None
    policy: str = "none"
    target: str = "state"

    def __post_init__(self) -> None:
        if self.policy not in LOW_RANK_POLICIES:
            raise ValueError(f"Unsupported low-rank policy: {self.policy}")
        if self.target not in LOW_RANK_TARGETS:
            raise ValueError(f"Unsupported low-rank target: {self.target}")
        if self.enabled:
            if self.rank is None:
                raise ValueError("Low-rank LocalMemory is enabled but rank is None")
            if self.rank <= 0:
                raise ValueError(f"Low-rank rank must be positive, got {self.rank}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "rank": self.rank,
            "policy": self.policy,
            "target": self.target,
            "implementation": "eval.methods.shared.metis_low_rank_memory",
            "projection": "torch.linalg.svd on the last two tensor dimensions in fp32, on the tensor device",
        }


def build_low_rank_local_memory_config(
    *,
    enabled: bool,
    rank: int | None,
    policy: str,
    target: str,
) -> LowRankLocalMemoryConfig:
    if not enabled:
        return LowRankLocalMemoryConfig(enabled=False, rank=rank, policy="none", target=target)
    return LowRankLocalMemoryConfig(enabled=True, rank=rank, policy=policy, target=target)


def _base_tensor_stats(
    tensor: torch.Tensor,
    *,
    requested_rank: int | None,
    action: str,
    reason: str | None = None,
) -> dict[str, Any]:
    rows = int(tensor.shape[-2]) if tensor.ndim >= 2 else None
    cols = int(tensor.shape[-1]) if tensor.ndim >= 2 else None
    full_rank = min(rows, cols) if rows is not None and cols is not None else None
    return {
        "shape": [int(dim) for dim in tensor.shape],
        "device": str(tensor.device),
        "dtype": str(tensor.dtype),
        "requested_rank": requested_rank,
        "effective_rank": None,
        "full_rank": full_rank,
        "energy_retained": None,
        "action": action,
        "reason": reason,
    }


@torch.no_grad()
def project_tensor_last2_low_rank(
    tensor: torch.Tensor,
    rank: int | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Project ``tensor`` to rank ``rank`` over its final two dimensions.

    Leading dimensions are treated as batch/head dimensions.  Computation stays
    on the tensor device and uses fp32 SVD for numerical stability.  ``rank`` is
    clipped to the matrix full rank; rank None or rank >= full rank is a no-op.
    """

    if rank is not None and rank <= 0:
        raise ValueError(f"rank must be positive or None, got {rank}")
    if not tensor.is_floating_point():
        return tensor, _base_tensor_stats(tensor, requested_rank=rank, action="skipped", reason="non_floating")
    if tensor.ndim < 2:
        return tensor, _base_tensor_stats(tensor, requested_rank=rank, action="skipped", reason="ndim_lt_2")

    rows = int(tensor.shape[-2])
    cols = int(tensor.shape[-1])
    full_rank = min(rows, cols)
    if full_rank == 0:
        return tensor, _base_tensor_stats(tensor, requested_rank=rank, action="skipped", reason="empty_matrix")

    if rank is None or rank >= full_rank:
        stats = _base_tensor_stats(tensor, requested_rank=rank, action="noop", reason="rank_is_full_or_none")
        stats["effective_rank"] = full_rank
        stats["energy_retained"] = 1.0
        return tensor, stats

    effective_rank = min(int(rank), full_rank)
    original_shape = tensor.shape
    original_dtype = tensor.dtype
    matrix = tensor.to(dtype=torch.float32).reshape(-1, rows, cols)
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)

    kept_energy = s[..., :effective_rank].square().sum(dim=-1)
    total_energy = s.square().sum(dim=-1)
    energy = torch.where(total_energy > 0, kept_energy / total_energy, torch.ones_like(total_energy))

    projected = (u[..., :, :effective_rank] * s[..., :effective_rank].unsqueeze(-2)) @ vh[..., :effective_rank, :]
    projected = projected.reshape(original_shape).to(dtype=original_dtype)

    stats = _base_tensor_stats(tensor, requested_rank=rank, action="projected")
    stats["effective_rank"] = effective_rank
    stats["energy_retained"] = float(energy.detach().float().mean().item())
    stats["energy_retained_min"] = float(energy.detach().float().min().item())
    stats["energy_retained_max"] = float(energy.detach().float().max().item())
    return projected, stats


def _set_memory_tensor(local_memory: Any, attr_name: str, value: torch.Tensor) -> None:
    if hasattr(local_memory, attr_name):
        setattr(local_memory, attr_name, value)
        return
    raise AttributeError(f"{local_memory.__class__.__name__} has no {attr_name} attribute")


def _safe_getattr(obj: Any, name: str) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return None


def summarize_projection_stats(stats: list[dict[str, Any]]) -> dict[str, Any]:
    energies = [
        item.get("energy_retained")
        for item in stats
        if isinstance(item.get("energy_retained"), (int, float))
    ]
    return {
        "tensor_count": len(stats),
        "projected_count": sum(1 for item in stats if item.get("action") == "projected"),
        "noop_count": sum(1 for item in stats if item.get("action") == "noop"),
        "skipped_count": sum(1 for item in stats if item.get("action") == "skipped"),
        "mean_energy_retained": float(sum(energies) / len(energies)) if energies else None,
        "min_energy_retained": float(min(energies)) if energies else None,
        "max_energy_retained": float(max(energies)) if energies else None,
    }


class MetisLowRankLocalMemoryProjector:
    """Apply low-rank projection to active Metis LocalMemory tensors."""

    def __init__(self, model: Any, config: LowRankLocalMemoryConfig) -> None:
        self.model = model
        self.config = config
        self._events: list[dict[str, Any]] = []

    @property
    def enabled(self) -> bool:
        return self.config.enabled and self.config.policy != "none"

    def reset_record_stats(self) -> None:
        self._events = []

    def config_dict(self) -> dict[str, Any]:
        return self.config.to_dict()

    @torch.no_grad()
    def project_active_memories(self) -> list[dict[str, Any]]:
        if not self.enabled:
            return []

        metis_model = _safe_getattr(self.model, "model")
        blocks = _safe_getattr(metis_model, "metis_blocks") or []
        out: list[dict[str, Any]] = []
        for layer_idx, layer in enumerate(blocks):
            local_memory = _safe_getattr(layer, "local_memory")
            if local_memory is None:
                continue

            state = _safe_getattr(local_memory, "state")
            if isinstance(state, torch.Tensor):
                projected, stats = project_tensor_last2_low_rank(state, self.config.rank)
                _set_memory_tensor(local_memory, "_state", projected)
                stats.update(
                    {
                        "layer_idx": layer_idx,
                        "memory_class": local_memory.__class__.__name__,
                        "tensor_name": "state",
                    }
                )
                out.append(stats)

            if self.config.target == "state_and_key":
                key_state = _safe_getattr(local_memory, "key_state")
                if isinstance(key_state, torch.Tensor):
                    projected_key, stats = project_tensor_last2_low_rank(key_state, self.config.rank)
                    _set_memory_tensor(local_memory, "_key_state", projected_key)
                    stats.update(
                        {
                            "layer_idx": layer_idx,
                            "memory_class": local_memory.__class__.__name__,
                            "tensor_name": "key_state",
                        }
                    )
                    out.append(stats)
        return out

    def _record_event(self, *, event: str, step_id: Any = None) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        tensor_stats = self.project_active_memories()
        event_record = {
            "event": event,
            "step_id": step_id,
            "summary": summarize_projection_stats(tensor_stats),
            "tensor_stats": tensor_stats,
        }
        self._events.append(event_record)
        return event_record

    def after_commit(self, *, step_id: Any = None) -> dict[str, Any] | None:
        if self.config.policy in {"after_each_commit", "after_each_commit_and_before_query"}:
            return self._record_event(event="after_commit", step_id=step_id)
        return None

    def before_query(self) -> dict[str, Any] | None:
        if self.config.policy in {"before_query", "after_each_commit_and_before_query"}:
            return self._record_event(event="before_query")
        return None

    def record_debug(self) -> dict[str, Any]:
        tensor_stats = [
            stat
            for event in self._events
            for stat in event.get("tensor_stats", [])
        ]
        return {
            "config": self.config_dict(),
            "event_count": len(self._events),
            "summary": summarize_projection_stats(tensor_stats),
            "events": self._events,
        }
