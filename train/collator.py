"""Per-chunk-position padding collate for bucketed memory training.

All samples in a batch are expected to share the same ``num_chunks``
(enforced by the bucket sampler).  Each chunk position is padded to
the max length within the batch, minimising padding waste.
"""

from __future__ import annotations

import torch

IGNORE_INDEX = -100


def build_collate_fn(pad_token_id: int):
    """Return a collate_fn for per-chunk-position padding.

    Input:  list of dicts with ``chunks``, ``num_chunks``, ``eval_chunk_idx``,
            ``task``, ``operation``, ``style``.
    Output: dict with ``input_ids``, ``attention_mask``, ``labels`` as lists
            of length T, plus scalar metadata.
    """

    def collate(batch: list[dict]) -> dict:
        num_chunks = batch[0]["num_chunks"]
        eval_idx = batch[0]["eval_chunk_idx"]
        for s in batch[1:]:
            assert s["num_chunks"] == num_chunks, \
                "Bucket violation: mixed num_chunks in one batch"
            assert s["eval_chunk_idx"] == eval_idx, \
                "Bucket violation: mixed eval_chunk_idx in one batch"

        per_chunk_ids: list[torch.Tensor] = []
        per_chunk_attn: list[torch.Tensor] = []
        per_chunk_lbls: list[torch.Tensor] = []

        for t in range(num_chunks):
            ids_list = [s["chunks"][t]["input_ids"] for s in batch]
            lbl_list = [s["chunks"][t]["labels"] for s in batch]
            max_len = max(x.size(0) for x in ids_list)

            ids_pad = torch.full(
                (len(batch), max_len), pad_token_id, dtype=torch.long,
            )
            attn_pad = torch.zeros(len(batch), max_len, dtype=torch.long)
            lbl_pad = torch.full(
                (len(batch), max_len), IGNORE_INDEX, dtype=torch.long,
            )
            for i, (ids, lbl) in enumerate(zip(ids_list, lbl_list)):
                L = ids.size(0)
                ids_pad[i, :L] = ids
                attn_pad[i, :L] = 1
                lbl_pad[i, :L] = lbl

            per_chunk_ids.append(ids_pad)
            per_chunk_attn.append(attn_pad)
            per_chunk_lbls.append(lbl_pad)

        return {
            "input_ids": per_chunk_ids,
            "attention_mask": per_chunk_attn,
            "labels": per_chunk_lbls,
            "num_chunks": num_chunks,
            "eval_chunk_idx": eval_idx,
            "task": [s["task"] for s in batch],
            "operation": [s["operation"] for s in batch],
        }

    return collate
