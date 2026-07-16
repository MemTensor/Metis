"""Shared runtime helpers for Metis memory writes."""

from __future__ import annotations

import torch


def is_full_attention(block) -> bool:
    return getattr(block.backbone_decoder.raw_decoder, "layer_type", "full_attention") == "full_attention"


def commit_hidden_offset(model) -> int:
    offset = getattr(model.config, "memory_configs", {}).get("commit_hidden_offset", 0)
    if offset not in (0, 1):
        raise ValueError(f"commit_hidden_offset must be 0 or 1, got {offset!r}")
    return offset


def encode_and_commit_memory(
    model,
    enc_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> None:
    """Encode tokens and write selected per-layer hidden states to local memory.

    Only full-attention layers are committed. ``commit_hidden_offset`` controls
    whether each layer writes its input (0) or output (1).
    """
    offset = commit_hidden_offset(model)
    captured: dict[int, torch.Tensor] = {}
    hooks = []

    for k, block in enumerate(model.model.metis_blocks):
        if not is_full_attention(block):
            continue

        def _hook(mod, inp, out, idx=k):
            captured[idx] = inp[0] if offset == 0 else out

        hooks.append(block.register_forward_hook(_hook))

    try:
        model.model(input_ids=enc_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for h in hooks:
            h.remove()

    for k, block in enumerate(model.model.metis_blocks):
        if k in captured:
            block.hyper_memory.update_local_memory(
                captured[k],
                block.local_memory,
                attention_mask=attention_mask,
            )
