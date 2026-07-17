#!/usr/bin/env python3
"""Run sentence-level DenseRAG on normalized MemOP JSONL instances."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_jsonl(path: Path, *, limit: int = 0) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return records[:limit] if limit else records


def chat_text(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def parse_max_memory(items: list[str] | None) -> dict[int | str, str] | None:
    if not items:
        return None
    out: dict[int | str, str] = {}
    for item in items:
        if ":" not in item:
            raise ValueError(f"Bad --max-memory item {item!r}; expected DEVICE:VALUE")
        key, value = item.split(":", 1)
        out[int(key) if key.isdigit() else key] = value
    return out


def load_generator(args: argparse.Namespace, dtype: torch.dtype) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": dtype}
    if args.device_map == "auto":
        kwargs["device_map"] = "auto"
        max_memory = parse_max_memory(args.max_memory)
        if max_memory:
            kwargs["max_memory"] = max_memory
    else:
        kwargs["device_map"] = {"": args.device}
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)
    model.eval()
    return model, tokenizer


def split_sentences(text: str) -> list[str]:
    text = " ".join(str(text or "").split())
    if not text:
        return []
    return [piece.strip() for piece in SENTENCE_SPLIT_RE.split(text) if piece.strip()] or [text]


def split_by_token_guard(text: str, tokenizer: Any, max_tokens: int) -> list[str]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return [text]
    chunks = []
    for start in range(0, len(ids), max_tokens):
        part = tokenizer.decode(ids[start : start + max_tokens], skip_special_tokens=True).strip()
        if part:
            chunks.append(part)
    return chunks


def render_messages(messages: list[dict[str, Any]]) -> str:
    lines = []
    for message in messages:
        role = str(message.get("role", "message")).upper()
        content = str(message.get("content", "")).strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def context_item_text(item: dict[str, Any]) -> str:
    if item.get("content") is not None:
        return str(item.get("content", ""))
    messages = item.get("messages")
    if isinstance(messages, list):
        return render_messages(messages)
    return str(item.get("text", ""))


def build_sentence_chunks(instance: dict[str, Any], tokenizer: Any, max_chunk_tokens: int) -> list[dict[str, Any]]:
    source_items = instance.get("context") or []
    if not source_items:
        source_items = instance.get("memory_steps") or []

    chunks: list[dict[str, Any]] = []
    for item_index, item in enumerate(source_items, start=1):
        source_text = context_item_text(item)
        source_id = item.get("turn_id") or item.get("segment_id") or item.get("step_id") or f"item_{item_index}"
        for sentence_index, sentence in enumerate(split_sentences(source_text)):
            for piece_index, piece in enumerate(split_by_token_guard(sentence, tokenizer, max_chunk_tokens)):
                chunks.append(
                    {
                        "chunk_id": f"c{len(chunks) + 1:04d}",
                        "source_id": source_id,
                        "turn_id": item.get("turn_id"),
                        "segment_id": item.get("segment_id"),
                        "step_id": item.get("step_id"),
                        "sentence_index": sentence_index,
                        "piece_index": piece_index,
                        "text": piece,
                    }
                )
    return chunks


@torch.no_grad()
def embed_texts(model: Any, tokenizer: Any, texts: list[str], device: str, max_tokens: int, batch_size: int) -> torch.Tensor:
    vectors: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=max_tokens, return_tensors="pt").to(device)
        outputs = model(**encoded)
        hidden = outputs.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        vectors.append(F.normalize(pooled.float().cpu(), p=2, dim=1))
    return torch.cat(vectors, dim=0) if vectors else torch.empty((0, 1), dtype=torch.float32)


def retrieve(instance: dict[str, Any], embed_model: Any, embed_tokenizer: Any, args: argparse.Namespace) -> tuple[list[dict[str, Any]], int]:
    chunks = build_sentence_chunks(instance, embed_tokenizer, args.max_chunk_tokens)
    if not chunks:
        return [], 0
    texts = [chunk["text"] for chunk in chunks]
    chunk_vectors = embed_texts(embed_model, embed_tokenizer, texts, args.embedding_device, args.embedding_max_tokens, args.embedding_batch_size)
    question_vector = embed_texts(embed_model, embed_tokenizer, [instance["question"]], args.embedding_device, args.embedding_max_tokens, 1)
    scores = torch.matmul(chunk_vectors, question_vector[0]).tolist()
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[: args.top_k]
    out: list[dict[str, Any]] = []
    for rank, (idx, score) in enumerate(ranked, start=1):
        item = dict(chunks[idx])
        item["rank"] = rank
        item["score"] = float(score)
        out.append(item)
    return out, len(chunks)


def prompt_for(instance: dict[str, Any], retrieved: list[dict[str, Any]]) -> str:
    lines = ["Retrieved context:"]
    if not retrieved:
        lines.append("No retrieved context.")
    for idx, chunk in enumerate(retrieved, start=1):
        lines.append(f"[chunk {idx}]")
        lines.append(str(chunk.get("text", "")))
    lines.extend(
        [
            "",
            f"Question: {instance['question']}",
            'Answer with a short phrase using only the retrieved context. If the answer is not known from the retrieved context, say "No information available".',
            "Short answer:",
        ]
    )
    return "\n".join(lines)


def audit_retrieval_prompt(instance: dict[str, Any], prompt: str) -> list[str]:
    issues: list[str] = []
    id_label_re = r"(?:evidence|source|turn|segment|step|id)(?:[ _-]*id)?"
    for evidence_id in instance.get("evidence", []) or []:
        evidence_text = str(evidence_id).strip()
        if not evidence_text:
            continue
        escaped = re.escape(evidence_text)
        explicit_id_patterns = [
            rf"(?im)^\s*{id_label_re}\s*[:=#-]\s*{escaped}\b",
            rf"(?i)\[\s*(?:{id_label_re}\s*)?{escaped}\s*\]",
        ]
        if any(re.search(pattern, prompt) for pattern in explicit_id_patterns):
            issues.append(f"prompt includes evidence id string: {evidence_text}")
    if "gold answer" in prompt.lower() or "rubric" in prompt.lower():
        issues.append("prompt includes gold-answer/rubric wording")
    return issues


@torch.no_grad()
def generate_answer(model: Any, tokenizer: Any, prompt: str, device: str, max_new_tokens: int) -> tuple[str, int, float]:
    started = time.time()
    text = chat_text(tokenizer, prompt)
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
    output_ids = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_ids = output_ids[:, encoded.input_ids.shape[1] :]
    return tokenizer.decode(new_ids[0], skip_special_tokens=True).strip(), int(encoded.input_ids.shape[1]), round(time.time() - started, 3)


def run(args: argparse.Namespace) -> Path:
    instances = read_jsonl(args.instances, limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    embed_tokenizer = AutoTokenizer.from_pretrained(args.embedding_model, trust_remote_code=True)
    embed_model = AutoModel.from_pretrained(args.embedding_model, trust_remote_code=True, torch_dtype=dtype).to(args.embedding_device)
    embed_model.eval()
    model, tokenizer = load_generator(args, dtype)
    dataset_label = instances[0].get("dataset", "memop") if instances else "memop"
    hf_device_map = getattr(model, "hf_device_map", None)
    meta = {
        "run_id": args.run_id,
        "created_at": utc_now(),
        "task": "memop",
        "dataset": dataset_label,
        "baseline": "dense_rag",
        "model_label": args.model_label,
        "model_path": args.model_path,
        "embedding_model": args.embedding_model,
        "instances": str(args.instances),
        "instance_count": len(instances),
        "top_k": args.top_k,
        "chunking": "sentence_level_with_token_guard_over_memop_context",
        "max_chunk_tokens": args.max_chunk_tokens,
        "retrieval_policy": "question embedding over current instance context corpus only; no gold answer or evidence ids are used for retrieval",
        "context_policy": "retrieved_context_top_k_sentence_chunks",
        "device": args.device,
        "embedding_device": args.embedding_device,
        "device_map": args.device_map,
        "max_memory": args.max_memory,
        "hf_device_map": {str(k): str(v) for k, v in hf_device_map.items()} if isinstance(hf_device_map, dict) else hf_device_map,
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
    }
    args.output.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with args.output.open("w", encoding="utf-8") as handle:
        for index, instance in enumerate(instances, start=1):
            started = time.time()
            retrieved, corpus_chunk_count = retrieve(instance, embed_model, embed_tokenizer, args)
            prompt = prompt_for(instance, retrieved)
            audit_issues = audit_retrieval_prompt(instance, prompt)
            if audit_issues and args.fail_on_audit_issue:
                raise RuntimeError(f"DenseRAG audit failed for {instance['instance_id']}: {audit_issues}")
            raw_output, prompt_tokens, query_latency = generate_answer(model, tokenizer, prompt, args.device, args.max_new_tokens)
            record = {
                "run_id": args.run_id,
                "date": utc_now(),
                "task": "memop",
                "task_type": instance.get("task_type"),
                "dataset": instance.get("dataset"),
                "split": instance.get("split"),
                "setting": instance.get("setting"),
                "operation": instance.get("operation"),
                "subtask": instance.get("subtask"),
                "baseline": "dense_rag",
                "model_label": args.model_label,
                "model_path": args.model_path,
                "embedding_model": args.embedding_model,
                "instance_index": index,
                "instance_count": len(instances),
                "instance_id": instance["instance_id"],
                "source_sample_id": instance.get("source_sample_id"),
                "question": instance["question"],
                "context_policy": "retrieved_context",
                "retrieval_policy": "question_only_over_memop_context_corpus",
                "corpus_chunk_count": corpus_chunk_count,
                "retrieved_chunks": retrieved,
                "retrieved_chunk_count": len(retrieved),
                "prompt_tokens": prompt_tokens,
                "latency_sec": round(time.time() - started, 3),
                "query_latency_sec": query_latency,
                "raw_output": raw_output,
                "generation_config": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
                "audit_issues": audit_issues,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                json.dumps(
                    {
                        "baseline": "dense_rag",
                        "model_label": args.model_label,
                        "index": index,
                        "total": len(instances),
                        "instance_id": instance["instance_id"],
                        "chunks": corpus_chunk_count,
                        "latency_sec": record["latency_sec"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return args.output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", default="memop_dense_rag")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-chunk-tokens", type=int, default=256)
    parser.add_argument("--embedding-max-tokens", type=int, default=512)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--embedding-device", default="cuda:0")
    parser.add_argument("--device-map", default="single", choices=["single", "auto"])
    parser.add_argument("--max-memory", nargs="*", default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--fail-on-audit-issue", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
