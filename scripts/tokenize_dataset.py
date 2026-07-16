#!/usr/bin/env python
"""Pre-tokenize Metis JSONL memory data into sharded torch caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from train.dataset import (
    MemoryDataset,
    _assign_task,
    _file_may_contain_task,
    _has_loss_after_memory_context,
    _infer_op_style,
)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - only used when tqdm is unavailable.
    tqdm = None


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class _SimpleProgress:
    def __init__(self, total: int, desc: str):
        self.total = total
        self.desc = desc
        self.n = 0

    def __enter__(self):
        logger.info(f"{self.desc}: 0/{self.total} lines")
        return self

    def __exit__(self, exc_type, exc, tb):
        logger.info(f"{self.desc}: {self.n}/{self.total} lines")

    def update(self, value: int = 1):
        self.n += value
        if self.n % 10000 == 0 or self.n == self.total:
            logger.info(f"{self.desc}: {self.n}/{self.total} lines")


def _progress(total: int, desc: str):
    if tqdm is not None:
        return tqdm(total=total, desc=desc, unit="line")
    return _SimpleProgress(total, desc)


def _discover_jsonl_files(data_dir: Path) -> list[Path]:
    root_files = sorted(data_dir.glob("*.jsonl"))
    if root_files:
        return root_files
    return sorted(p for p in data_dir.glob("*/*.jsonl") if p.is_file())


def _count_lines(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        with path.open() as f:
            for _ in f:
                total += 1
    return total


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.iterdir())
    if existing and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Pass --overwrite to replace existing tokenized shards."
        )
    if overwrite:
        for path in output_dir.glob("shard_*.pt"):
            path.unlink()
        manifest = output_dir / "manifest.json"
        if manifest.exists():
            manifest.unlink()


def _save_shard(samples: list[dict], output_dir: Path, shard_idx: int) -> dict:
    filename = f"shard_{shard_idx:05d}.pt"
    torch.save(samples, output_dir / filename)
    return {"file": filename, "num_samples": len(samples)}


def tokenize_dataset(args) -> None:
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    tasks = {int(t) for t in args.tasks.split(",") if t.strip()}

    _prepare_output_dir(output_dir, args.overwrite)

    tokenizer_path = args.tokenizer_path or args.model_path
    if not tokenizer_path:
        raise ValueError("Pass --tokenizer_path or --model_path")

    logger.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    jsonl_files = _discover_jsonl_files(data_dir)
    selected_files: list[Path] = []
    for jsonl_file in jsonl_files:
        operation, style = _infer_op_style(jsonl_file, data_dir)
        if _file_may_contain_task(operation, style, tasks):
            selected_files.append(jsonl_file)

    total_lines = _count_lines(selected_files)
    logger.info(f"Tokenizing {len(selected_files)} JSONL files ({total_lines} lines)")

    encoder = object.__new__(MemoryDataset)
    shard_buffer: list[dict] = []
    shards: list[dict] = []
    file_summaries: list[dict] = []
    per_task_counts: Counter[int] = Counter()
    per_nchunks_counts: Counter[int] = Counter()
    per_op_counts: Counter[str] = Counter()
    parse_skipped_total = 0
    encode_skipped_total = 0
    length_skipped_total = 0
    memory_context_skipped_total = 0
    parse_examples: list[str] = []
    encode_examples: list[str] = []

    with _progress(total_lines, "Tokenizing") as pbar:
        for jsonl_file in selected_files:
            inferred_operation, inferred_style = _infer_op_style(jsonl_file, data_dir)
            file_summary = {
                "file": str(jsonl_file),
                "kept": 0,
                "parse_skipped": 0,
                "encode_skipped": 0,
                "length_skipped": 0,
                "memory_context_skipped": 0,
                "task_counts": {},
            }

            with jsonl_file.open() as f:
                for lineno, line in enumerate(f, 1):
                    pbar.update(1)
                    if args.max_samples_per_file and file_summary["kept"] >= args.max_samples_per_file:
                        continue
                    if not line.strip():
                        file_summary["parse_skipped"] += 1
                        parse_skipped_total += 1
                        if len(parse_examples) < 5:
                            parse_examples.append(f"{jsonl_file}:{lineno} blank line")
                        continue

                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError as exc:
                        file_summary["parse_skipped"] += 1
                        parse_skipped_total += 1
                        if len(parse_examples) < 5:
                            parse_examples.append(f"{jsonl_file}:{lineno} {exc}")
                        continue
                    if not isinstance(raw, dict):
                        file_summary["parse_skipped"] += 1
                        parse_skipped_total += 1
                        if len(parse_examples) < 5:
                            parse_examples.append(f"{jsonl_file}:{lineno} top-level JSON is not an object")
                        continue

                    metadata = raw.get("metadata") or {}
                    operation = metadata.get("type") or inferred_operation
                    style = metadata.get("style") or inferred_style
                    task_id = _assign_task(operation, style, metadata.get("v2_task"))
                    if task_id not in tasks:
                        continue

                    try:
                        sample = MemoryDataset._encode_sample(
                            encoder, raw, tokenizer, task_id, operation, style,
                        )
                    except Exception as exc:
                        file_summary["encode_skipped"] += 1
                        encode_skipped_total += 1
                        if len(encode_examples) < 5:
                            sample_id = raw.get("sample_id", "<unknown>")
                            encode_examples.append(f"{jsonl_file}:{lineno} sample_id={sample_id} {exc}")
                        continue
                    if sample is None:
                        continue
                    if not _has_loss_after_memory_context(sample):
                        file_summary["memory_context_skipped"] += 1
                        memory_context_skipped_total += 1
                        continue

                    total_tokens = int(sample["total_tokens"])
                    if args.min_total_tokens and total_tokens < args.min_total_tokens:
                        file_summary["length_skipped"] += 1
                        length_skipped_total += 1
                        continue
                    if args.max_total_tokens and total_tokens > args.max_total_tokens:
                        file_summary["length_skipped"] += 1
                        length_skipped_total += 1
                        continue

                    shard_buffer.append(sample)
                    file_summary["kept"] += 1
                    file_summary["task_counts"][str(task_id)] = (
                        file_summary["task_counts"].get(str(task_id), 0) + 1
                    )
                    per_task_counts[task_id] += 1
                    per_nchunks_counts[sample["num_chunks"]] += 1
                    per_op_counts[f"{operation}/{style or 'unknown'}"] += 1

                    if len(shard_buffer) >= args.shard_size:
                        shards.append(_save_shard(shard_buffer, output_dir, len(shards)))
                        shard_buffer = []

            file_summaries.append(file_summary)
            logger.info(
                f"Processed {jsonl_file}: kept={file_summary['kept']} "
                f"parse_skipped={file_summary['parse_skipped']} "
                f"encode_skipped={file_summary['encode_skipped']} "
                f"length_skipped={file_summary['length_skipped']} "
                f"memory_context_skipped={file_summary['memory_context_skipped']}"
            )

    if shard_buffer:
        shards.append(_save_shard(shard_buffer, output_dir, len(shards)))

    chat_template = tokenizer.chat_template or ""
    manifest = {
        "format": "metis-tokenized-memory-v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_data_dir": str(data_dir),
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_class": type(tokenizer).__name__,
        "chat_template_sha256": hashlib.sha256(chat_template.encode("utf-8")).hexdigest(),
        "enable_thinking": False,
        "add_special_tokens": False,
        "tasks": sorted(tasks),
        "max_samples_per_file": args.max_samples_per_file,
        "min_total_tokens": args.min_total_tokens,
        "max_total_tokens": args.max_total_tokens,
        "shard_size": args.shard_size,
        "total_samples": sum(per_task_counts.values()),
        "task_counts": {str(k): v for k, v in sorted(per_task_counts.items())},
        "num_chunks_counts": {str(k): v for k, v in sorted(per_nchunks_counts.items())},
        "operation_style_counts": dict(sorted(per_op_counts.items())),
        "parse_skipped": parse_skipped_total,
        "parse_examples": parse_examples,
        "encode_skipped": encode_skipped_total,
        "encode_examples": encode_examples,
        "length_skipped": length_skipped_total,
        "memory_context_skipped": memory_context_skipped_total,
        "files": file_summaries,
        "shards": shards,
    }

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info(f"Wrote {len(shards)} shards to {output_dir}")
    logger.info(f"Wrote manifest to {manifest_path}")
    logger.info(
        f"Total samples={manifest['total_samples']} "
        f"parse_skipped={parse_skipped_total} encode_skipped={encode_skipped_total} "
        f"length_skipped={length_skipped_total} "
        f"memory_context_skipped={memory_context_skipped_total}"
    )


def parse_args():
    p = argparse.ArgumentParser(description="Tokenize Metis JSONL data into sharded cache")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_path", default=None, help="Model path used as tokenizer path if --tokenizer_path is unset")
    p.add_argument("--tokenizer_path", default=None)
    p.add_argument("--tasks", default="0,1,2,3,4")
    p.add_argument("--shard_size", type=int, default=10000)
    p.add_argument("--max_samples_per_file", type=int, default=0, help="0 = all")
    p.add_argument("--min_total_tokens", type=int, default=0, help="0 = no minimum length filter")
    p.add_argument("--max_total_tokens", type=int, default=0, help="0 = no maximum length filter")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    tokenize_dataset(parse_args())
