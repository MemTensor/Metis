"""Unified memory dataset for Metis multi-task training.

Reads from ``data/train_data/`` with the following task mapping.  Both
layouts are supported:

* flat merged files: ``data_dir/remember_explicit.jsonl``
* legacy task dirs: ``data_dir/remember/explicit_data.jsonl``

======== ================================================ ==============================
Task     Operations / v2 task marker                       Styles
======== ================================================ ==============================
0 (fact_recall)    reconstruction, remember                explicit, implicit
1 (memory_op)      remember, forget, update, reflection    explicit, implicit
2 (long_term)      remember, forget, update, reflection    distract
3 (v2_task3)       metadata.v2_task starts with task3      llm snippet task3 data
4 (v2_task4)       metadata.v2_task starts with task4      normal/no-query task4 data
======== ================================================ ==============================

Each sample is a multi-turn dialogue encoded via ``apply_chat_template``.
Memory flows through all chunks; loss is computed only on the query chunk's
assistant response.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100

# Task assignments based on (operation, style).
_TASK_0_OPS = {"remember"}
_TASK_0_STYLES = {"explicit", "implicit"}

_TASK_1_OPS = {"remember", "forget", "update", "reflection"}
_TASK_1_STYLES = {"explicit", "implicit"}

_TASK_2_STYLES = {"distract", "distract_explicit", "distract_implicit", "distract_exp"}

logger = logging.getLogger(__name__)


def _assign_task(operation: str, style: str, v2_task: str | None = None) -> int:
    """Map a sample to training task id.

    v2 Task3/Task4 are kept as independent training tasks when metadata or
    file-derived operation marks them as such. Older data still maps to the
    original task ids 0/1/2.
    """
    v2_task = v2_task or ""
    if v2_task.startswith("task3") or operation == "task3":
        return 3
    if v2_task.startswith("task4") or operation == "task4":
        return 4
    if operation == "reconstruction":
        return 0
    if "distract" in style:
        return 2
    if operation in _TASK_0_OPS and style in _TASK_0_STYLES:
        return 0
    if operation in _TASK_1_OPS and style in _TASK_1_STYLES:
        return 1
    # Fallback: explicitly specified distract styles -> task 2; everything else -> 1
    return 2 if "distract" in style else 1


def _file_may_contain_task(operation: str, style: str, tasks: set[int]) -> bool:
    """Conservative file-level filter before reading metadata-bearing rows."""
    if _assign_task(operation, style) in tasks:
        return True
    return bool(tasks & {3, 4})


def _infer_op_style(path: Path, data_dir: Path) -> tuple[str, str]:
    """Infer operation/style from supported dataset file layouts."""
    if path.parent == data_dir:
        stem = path.stem
        if stem == "reconstruction":
            return "reconstruction", "reconstruction"
        operation, sep, style = stem.partition("_")
        return operation, style if sep else ""

    operation = path.parent.name
    stem = path.stem
    if stem.endswith("_data"):
        return operation, stem.removesuffix("_data")
    if "_" in stem:
        return operation, stem.rsplit("_", 1)[1]
    return operation, stem


def _resolve_turn_ids(value, num_chunks: int) -> list[int]:
    values = value if isinstance(value, list) else [value]
    out: list[int] = []
    for item in values:
        if not isinstance(item, int):
            continue
        idx = item if item >= 0 else num_chunks + item
        if 0 <= idx < num_chunks and idx not in out:
            out.append(idx)
    return out


def _label_count(labels) -> int:
    if isinstance(labels, torch.Tensor):
        return int((labels != IGNORE_INDEX).sum().item())
    return sum(1 for token in labels if int(token) != IGNORE_INDEX)


def _has_loss_after_memory_context(sample: dict) -> bool:
    """True when at least one labelled chunk can read prior committed memory."""
    chunks = sample.get("chunks") or []
    eval_chunk_ids = sample.get("eval_chunk_ids")
    if isinstance(eval_chunk_ids, list) and eval_chunk_ids:
        candidate_ids = eval_chunk_ids
    else:
        candidate_ids = range(1, len(chunks))

    for idx in candidate_ids:
        try:
            chunk_idx = int(idx)
        except (TypeError, ValueError):
            continue
        if chunk_idx <= 0 or chunk_idx >= len(chunks):
            continue
        labels = chunks[chunk_idx].get("labels")
        if labels is not None and _label_count(labels) > 0:
            return True
    return False


class MemoryDataset(Dataset):
    """Unified dataset for all three memory training tasks.

    Args:
        data_dir: Path to ``data/train_data/``.
        tokenizer: HuggingFace tokenizer (must support ``apply_chat_template``).
        tasks: Subset of task ids to load, e.g. ``[0, 1]``.  ``None`` = all.
        max_samples_per_task: Cap samples per input JSONL for quick debug (0 = all).
    """

    def __init__(
        self,
        data_dir: str | Path,
        tokenizer,
        tasks: list[int] | None = None,
        max_samples_per_task: int = 0,
        max_total_tokens: int = 0,
    ):
        data_dir = Path(data_dir)
        tasks = set(tasks) if tasks else {0, 1, 2, 3, 4}

        self.samples: list[dict] = []
        per_task_counts: dict[int, int] = defaultdict(int)
        per_nchunks_counts: dict[int, int] = defaultdict(int)
        per_op_counts: dict[str, int] = defaultdict(int)
        memory_context_skipped_total = 0

        root_files = sorted(data_dir.glob("*.jsonl"))
        if root_files:
            jsonl_files = root_files
            logger.info(f"Using flat JSONL dataset layout from {data_dir} ({len(jsonl_files)} files)")
        else:
            jsonl_files = sorted(p for p in data_dir.glob("*/*.jsonl") if p.is_file())
            logger.info(f"Using nested JSONL dataset layout from {data_dir} ({len(jsonl_files)} files)")

        for jsonl_file in jsonl_files:
            inferred_operation, inferred_style = _infer_op_style(jsonl_file, data_dir)
            if not _file_may_contain_task(inferred_operation, inferred_style, tasks):
                continue

            task_counts_before = dict(per_task_counts)
            kept = 0
            parse_skipped = 0
            parse_examples: list[str] = []
            skipped = 0
            skipped_examples: list[str] = []
            length_skipped = 0
            memory_context_skipped = 0

            with open(jsonl_file) as f:
                for lineno, line in enumerate(f, 1):
                    if not line.strip():
                        parse_skipped += 1
                        if len(parse_examples) < 3:
                            parse_examples.append(f"line={lineno} error=blank line")
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError as exc:
                        parse_skipped += 1
                        if len(parse_examples) < 3:
                            parse_examples.append(f"line={lineno} error={exc}")
                        continue
                    if not isinstance(raw, dict):
                        parse_skipped += 1
                        if len(parse_examples) < 3:
                            parse_examples.append(f"line={lineno} error=top-level JSON is not an object")
                        continue
                    metadata = raw.get("metadata") or {}
                    operation = metadata.get("type") or inferred_operation
                    style = metadata.get("style") or inferred_style
                    task_id = _assign_task(operation, style, metadata.get("v2_task"))
                    if task_id not in tasks:
                        continue

                    try:
                        sample = self._encode_sample(raw, tokenizer, task_id, operation, style)
                    except Exception as exc:
                        skipped += 1
                        if len(skipped_examples) < 3:
                            sample_id = raw.get("sample_id", "<unknown>")
                            skipped_examples.append(f"line={lineno} sample_id={sample_id} error={exc}")
                        continue
                    if sample is None:
                        continue
                    if not _has_loss_after_memory_context(sample):
                        memory_context_skipped += 1
                        continue
                    if max_total_tokens and sample["total_tokens"] > max_total_tokens:
                        length_skipped += 1
                        continue

                    self.samples.append(sample)
                    per_task_counts[task_id] += 1
                    per_nchunks_counts[sample["num_chunks"]] += 1
                    per_op_counts[f"{operation}/{style or 'unknown'}"] += 1
                    kept += 1

                    if max_samples_per_task and kept >= max_samples_per_task:
                        break

            for task_id in sorted(set(task_counts_before) | set(per_task_counts)):
                added = per_task_counts[task_id] - task_counts_before.get(task_id, 0)
                if added:
                    logger.info(f"  Loaded task={task_id} +{added:>5d} samples from {jsonl_file}")
            if skipped:
                logger.warning(
                    f"  Skipped {skipped} samples from {jsonl_file} due to encoding errors. "
                    f"Examples: {'; '.join(skipped_examples)}"
                )
            if parse_skipped:
                logger.warning(
                    f"  Skipped {parse_skipped} lines from {jsonl_file} due to JSONL parse errors. "
                    f"Examples: {'; '.join(parse_examples)}"
                )
            if length_skipped:
                logger.info(
                    f"  Skipped {length_skipped} samples from {jsonl_file} "
                    f"with total_tokens > {max_total_tokens}"
                )
            if memory_context_skipped:
                memory_context_skipped_total += memory_context_skipped
                logger.info(
                    f"  Skipped {memory_context_skipped} samples from {jsonl_file} "
                    "with no labelled chunk after committed memory"
                )

        self._per_task_counts = dict(per_task_counts)
        self._per_nchunks_counts = dict(per_nchunks_counts)
        self._per_op_counts = dict(per_op_counts)
        self._memory_context_skipped = memory_context_skipped_total
        self._summary()

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _encode_sample(self, raw: dict, tokenizer,
                       task_id: int, operation: str, style: str) -> dict | None:
        chunks_raw = raw.get("messages")
        if not chunks_raw:
            return None

        num_chunks = len(chunks_raw)
        query_turn_ids = _resolve_turn_ids(raw.get("query_turn_id", -1), num_chunks)
        if not query_turn_ids:
            query_turn_ids = [num_chunks - 1]
        eval_chunk_idx = query_turn_ids[0]
        query_turn_id_set = set(query_turn_ids)

        if eval_chunk_idx >= num_chunks:
            return None

        chunks: list[dict] = []
        for ci, turns in enumerate(chunks_raw):
            ids_t, lbl_t = self._encode_chunk(
                turns, tokenizer,
                is_eval_chunk=(ci in query_turn_id_set),
            )
            chunks.append({"input_ids": ids_t, "labels": lbl_t})

        # Skip samples with zero trainable labels in every query chunk.
        label_tokens = sum(
            int((chunks[idx]["labels"] != IGNORE_INDEX).sum().item())
            for idx in query_turn_ids
            if 0 <= idx < len(chunks)
        )
        if label_tokens == 0:
            return None

        return {
            "chunks": chunks,
            "num_chunks": num_chunks,
            "total_tokens": sum(int(c["input_ids"].size(0)) for c in chunks),
            "eval_chunk_idx": eval_chunk_idx,
            "eval_chunk_ids": query_turn_ids,
            "task": task_id,
            "operation": operation,
            "style": style,
        }

    @staticmethod
    def _encode_chunk(
        turns: list[dict],
        tokenizer,
        is_eval_chunk: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode one chunk's turns via chat_template (non-think mode).

        Returns ``(input_ids, labels)``.  For the eval chunk, the last
        assistant turn's answer is labelled for loss computation.
        """
        messages = [{"role": t["role"], "content": t["content"]} for t in turns]

        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        labels = [IGNORE_INDEX] * len(full_ids)

        if not is_eval_chunk:
            return (
                torch.tensor(full_ids, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long),
            )

        # Find the last assistant turn and label its answer tokens.
        last_asst_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "assistant":
                last_asst_idx = i
                break

        if last_asst_idx is None:
            return (
                torch.tensor(full_ids, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long),
            )

        # Prefix up to (but not including) the last assistant's content.
        prefix_text = tokenizer.apply_chat_template(
            messages[:last_asst_idx],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)

        if full_ids[: len(prefix_ids)] != prefix_ids:
            raise RuntimeError(
                "chat_template prefix drift: partial-render tokens do not match "
                "full-chunk render. Check tokenizer / template."
            )
        asst_start = len(prefix_ids)

        # Find where the assistant response ends.
        # Rendering up to and including the last assistant turn.
        upto_text = tokenizer.apply_chat_template(
            messages[: last_asst_idx + 1],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        upto_ids = tokenizer.encode(upto_text, add_special_tokens=False)
        if full_ids[: len(upto_ids)] != upto_ids:
            raise RuntimeError("chat_template suffix drift")
        asst_end = len(upto_ids)

        for i in range(asst_start, asst_end):
            labels[i] = full_ids[i]

        return (
            torch.tensor(full_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]

    def num_chunks_of(self, idx: int) -> int:
        return self.samples[idx]["num_chunks"]

    def task_of(self, idx: int) -> int:
        return self.samples[idx]["task"]

    def eval_chunk_idx_of(self, idx: int) -> int:
        return self.samples[idx]["eval_chunk_idx"]

    def total_tokens_of(self, idx: int) -> int:
        return self.samples[idx]["total_tokens"]

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _summary(self) -> None:
        lines = [f"\nMemoryDataset  total={len(self)}"]
        for tid in sorted(self._per_task_counts):
            lines.append(f"  Task {tid}: {self._per_task_counts[tid]:>5d} samples")
        for op_style, c in sorted(self._per_op_counts.items()):
            lines.append(f"    {op_style:30s} {c:>5d}")
        for nc, c in sorted(self._per_nchunks_counts.items()):
            lines.append(f"  num_chunks={nc}:           {c:>5d} samples")
        if self._memory_context_skipped:
            lines.append(f"  memory_context_filtered: {self._memory_context_skipped:>5d} samples")
        logger.info("\n".join(lines))


class TokenizedMemoryDataset(Dataset):
    """Dataset backed by pre-tokenized Metis memory shards."""

    def __init__(
        self,
        cache_dir: str | Path,
        tasks: list[int] | None = None,
        max_samples_per_task: int = 0,
        max_total_tokens: int = 0,
    ):
        cache_dir = Path(cache_dir)
        manifest_path = cache_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Tokenized dataset manifest not found: {manifest_path}")

        with open(manifest_path) as f:
            self.manifest = json.load(f)

        if self.manifest.get("format") != "metis-tokenized-memory-v1":
            raise ValueError(f"Unsupported tokenized dataset format in {manifest_path}")

        tasks_filter = set(tasks) if tasks else None
        self.samples: list[dict] = []
        per_task_counts: dict[int, int] = defaultdict(int)
        per_nchunks_counts: dict[int, int] = defaultdict(int)
        per_op_counts: dict[str, int] = defaultdict(int)
        length_skipped = 0
        memory_context_skipped = 0

        shard_infos = self.manifest.get("shards") or []
        logger.info(f"Loading tokenized dataset from {cache_dir} ({len(shard_infos)} shards)")

        for shard_info in shard_infos:
            shard_path = cache_dir / shard_info["file"]
            for sample in _torch_load(shard_path):
                task_id = int(sample["task"])
                if tasks_filter is not None and task_id not in tasks_filter:
                    continue
                if not _has_loss_after_memory_context(sample):
                    memory_context_skipped += 1
                    continue
                if max_total_tokens and int(sample["total_tokens"]) > max_total_tokens:
                    length_skipped += 1
                    continue
                if max_samples_per_task and per_task_counts[task_id] >= max_samples_per_task:
                    continue

                self.samples.append(sample)
                per_task_counts[task_id] += 1
                per_nchunks_counts[sample["num_chunks"]] += 1
                per_op_counts[f"{sample.get('operation')}/{sample.get('style') or 'unknown'}"] += 1

        self._per_task_counts = dict(per_task_counts)
        self._per_nchunks_counts = dict(per_nchunks_counts)
        self._per_op_counts = dict(per_op_counts)
        self._length_skipped = length_skipped
        self._memory_context_skipped = memory_context_skipped
        self._summary()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]

    def num_chunks_of(self, idx: int) -> int:
        return self.samples[idx]["num_chunks"]

    def task_of(self, idx: int) -> int:
        return self.samples[idx]["task"]

    def eval_chunk_idx_of(self, idx: int) -> int:
        return self.samples[idx]["eval_chunk_idx"]

    def total_tokens_of(self, idx: int) -> int:
        return self.samples[idx]["total_tokens"]

    def _summary(self) -> None:
        lines = [f"\nTokenizedMemoryDataset  total={len(self)}"]
        for tid in sorted(self._per_task_counts):
            lines.append(f"  Task {tid}: {self._per_task_counts[tid]:>5d} samples")
        for op_style, c in sorted(self._per_op_counts.items()):
            lines.append(f"    {op_style:30s} {c:>5d}")
        for nc, c in sorted(self._per_nchunks_counts.items()):
            lines.append(f"  num_chunks={nc}:           {c:>5d} samples")
        if self._length_skipped:
            lines.append(f"  length_filtered:     {self._length_skipped:>5d} samples")
        if self._memory_context_skipped:
            lines.append(f"  memory_context_filtered: {self._memory_context_skipped:>5d} samples")
        logger.info("\n".join(lines))


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")
