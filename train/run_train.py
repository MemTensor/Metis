#!/usr/bin/env python
"""Metis unified multi-task memory training.

Supports three training stages with dynamic mixing:
  Task 0: Fact recall / reconstruction  (remember + explicit/implicit)
  Task 1: Memory operations             (remember/forget/update/reflection + explicit/implicit)
  Task 2: Long-term memory with distractors

Usage::
    torchrun --nproc_per_node=8 train/run_train.py \\
        --model_path /path/to/backbone \\
        --data_dir data/train_data \\
        --output_dir checkpoint/experiment_1 \\
        --num_epochs 200 --batch_size 2 --gradient_accumulation_steps 60
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Sampler
from transformers import TrainingArguments

# Ensure the project root is on sys.path so that both 'metis' and 'train'
# imports work regardless of whether this file is run as a script or as a module.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from metis.weight_utils import load_metis_from_backbone
from metis.checkpoint_utils import load_metis_model_from_checkpoint

from train.dataset import MemoryDataset, TokenizedMemoryDataset
from train.collator import build_collate_fn
from train.task_scheduler import TaskScheduler
from train.trainer import MetisMemoryTrainer, MetisEvalCallback
from train.train_utils import (
    FileLoggingCallback,
    MasterWeightAdamW,
    apply_lora,
    count_params,
    dump_sampler_tree,
    freeze_backbone,
    save_code_snapshot,
    setup_file_logging,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def _startup_probe(message: str) -> None:
    if os.environ.get("METIS_STARTUP_PROBE", "1") in {"0", "false", "False"}:
        return
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "-1"))
    local_rank = os.environ.get("LOCAL_RANK", "-1")
    print(
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} | STARTUP | "
        f"[rank {rank} local {local_rank} pid {os.getpid()}] {message}",
        file=sys.stderr, flush=True,
    )


def _resolve_resume_checkpoint(resume_arg: str, output_dir: str) -> str:
    """Resolve --resume_from_checkpoint into a concrete checkpoint directory.

    Accepts an explicit path, or "auto"/"latest" to pick the highest-numbered
    ``checkpoint-N`` under ``output_dir`` that contains a ``trainer_state.json``
    (checkpoint-0 and interrupted saves without trainer state are skipped).
    Returns "" when auto finds nothing resumable, so the caller starts fresh.
    """
    if not resume_arg:
        return ""
    if resume_arg not in {"auto", "latest"}:
        checkpoint = Path(resume_arg)
        if not (checkpoint / "trainer_state.json").is_file():
            raise ValueError(
                f"--resume_from_checkpoint {resume_arg} has no trainer_state.json; "
                "it is not a resumable trainer checkpoint."
            )
        return str(checkpoint)

    candidates: list[tuple[int, Path]] = []
    root = Path(output_dir)
    if root.is_dir():
        for entry in root.iterdir():
            name = entry.name
            if not (entry.is_dir() and name.startswith("checkpoint-")):
                continue
            suffix = name.removeprefix("checkpoint-")
            if suffix.isdigit() and (entry / "trainer_state.json").is_file():
                candidates.append((int(suffix), entry))
    if not candidates:
        logger.warning(
            "resume_from_checkpoint=%s: no resumable checkpoint under %s — starting fresh.",
            resume_arg, output_dir,
        )
        return ""
    step, path = max(candidates)
    logger.info("resume_from_checkpoint=%s resolved → %s (step %d)", resume_arg, path, step)
    return str(path)


def _patch_torch_load_safety_for_trusted_resume(resume_checkpoint: str) -> None:
    """Allow HF Trainer to torch.load optimizer/rng files from a trusted local checkpoint.

    transformers 5.x gates every resume-time ``torch.load`` behind
    ``check_torch_load_is_safe`` + ``weights_only=True``; MasterWeightAdamW /
    rng state files need the legacy pickle path.  This patch no-ops the gate
    and retries ``weights_only=False`` only for files inside the resume
    checkpoint directory.  Gated by
    ``METIS_TRUST_LOCAL_TORCH_LOAD=1`` so it is never active implicitly.
    """
    if not resume_checkpoint:
        return
    enabled = os.environ.get("METIS_TRUST_LOCAL_TORCH_LOAD", "0").lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return

    checkpoint = Path(resume_checkpoint)
    if not checkpoint.is_dir():
        raise ValueError(
            "METIS_TRUST_LOCAL_TORCH_LOAD=1 was set, but resume_from_checkpoint "
            f"is not a local directory: {checkpoint}"
        )

    trusted_root = checkpoint.resolve()
    original_torch_load = torch.load

    def _trusted_local_checkpoint_noop() -> None:
        return None

    def _is_inside_trusted_checkpoint(target) -> bool:
        try:
            path = Path(os.fspath(target)).resolve()
        except (TypeError, ValueError):
            return False
        try:
            path.relative_to(trusted_root)
            return True
        except ValueError:
            return False

    def _trusted_torch_load(*load_args, **load_kwargs):
        try:
            return original_torch_load(*load_args, **load_kwargs)
        except pickle.UnpicklingError:
            if not load_args or not _is_inside_trusted_checkpoint(load_args[0]):
                raise
            if load_kwargs.get("weights_only") is not True:
                raise
            retry_kwargs = dict(load_kwargs)
            retry_kwargs["weights_only"] = False
            logger.warning(
                "Retrying torch.load(weights_only=False) for trusted local resume file: %s",
                load_args[0],
            )
            return original_torch_load(*load_args, **retry_kwargs)

    import transformers.trainer as hf_trainer
    import transformers.trainer_utils as hf_trainer_utils
    import transformers.utils.import_utils as hf_import_utils

    hf_import_utils.check_torch_load_is_safe = _trusted_local_checkpoint_noop
    hf_trainer.check_torch_load_is_safe = _trusted_local_checkpoint_noop
    hf_trainer_utils.check_torch_load_is_safe = _trusted_local_checkpoint_noop
    torch.load = _trusted_torch_load
    logger.warning(
        "METIS_TRUST_LOCAL_TORCH_LOAD=1: bypassing transformers torch.load "
        "version gate for trusted local resume checkpoint %s",
        checkpoint,
    )


def _enable_zero3_memory_efficient_linear() -> None:
    """Enable DeepSpeed ZeRO-3's memory-efficient linear wrapper."""
    try:
        from deepspeed.runtime.zero.linear import zero3_linear_wrap
    except Exception as exc:
        raise RuntimeError(
            "--deepspeed was set, but DeepSpeed's ZeRO-3 linear wrapper "
            "could not be imported."
        ) from exc

    if torch.nn.functional.linear is zero3_linear_wrap:
        return
    torch.nn.functional.linear = zero3_linear_wrap
    logger.info("Enabled DeepSpeed ZeRO-3 memory_efficient_linear wrapper")


# ── Task-weighted length sampler ───────────────────────────────────

class TaskWeightedLengthDistributedSampler(Sampler):
    """Build task-weighted global batches with similar total token lengths."""

    def __init__(self, dataset: MemoryDataset, num_replicas: int, rank: int,
                 batch_size: int, task_scheduler: TaskScheduler | None = None,
                 total_epochs: int = 1, epoch_getter=None,
                 shuffle: bool = True, seed: int = 0):
        assert rank < num_replicas
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.task_scheduler = task_scheduler
        self.total_epochs = total_epochs
        self.epoch_getter = epoch_getter
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for i in range(len(dataset)):
            key = (
                dataset.task_of(i),
                dataset.num_chunks_of(i),
                dataset.eval_chunk_idx_of(i),
            )
            buckets[key].append(i)
        self.buckets = dict(buckets)

        self._global_bs = batch_size * num_replicas
        self._total_global_batches = sum(len(idxs) // self._global_bs for idxs in self.buckets.values())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        if self.epoch_getter is not None:
            self.epoch = int(self.epoch_getter() or self.epoch)
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        task_batches: dict[int, list[list[int]]] = defaultdict(list)
        for (task_id, _num_chunks, _eval_idx), idxs in self.buckets.items():
            if len(idxs) < self._global_bs:
                continue

            tie_breakers = torch.rand(len(idxs), generator=g).tolist()
            ties = {idx: tie_breakers[pos] for pos, idx in enumerate(idxs)}
            idxs = sorted(
                idxs,
                key=lambda idx: (self.dataset.total_tokens_of(idx), ties[idx]),
            )

            usable = (len(idxs) // self._global_bs) * self._global_bs
            for start in range(0, usable, self._global_bs):
                task_batches[task_id].append(idxs[start:start + self._global_bs])

        all_global_batches = self._sample_task_batches(task_batches, g)

        if self.shuffle:
            order = torch.randperm(len(all_global_batches), generator=g).tolist()
            all_global_batches = [all_global_batches[i] for i in order]

        for gb in all_global_batches:
            start = self.rank * self.batch_size
            end = start + self.batch_size
            for idx in gb[start:end]:
                yield idx

    def __len__(self) -> int:
        return self._total_global_batches * self.batch_size

    def _sample_task_batches(
        self,
        task_batches: dict[int, list[list[int]]],
        g: torch.Generator,
    ) -> list[list[int]]:
        available_tasks = sorted(t for t, batches in task_batches.items() if batches)
        if not available_tasks:
            return []

        total_batches = sum(len(task_batches[t]) for t in available_tasks)
        if self.task_scheduler is None:
            weights = {t: 1.0 for t in available_tasks}
        else:
            weights = self.task_scheduler.get_weights(self.epoch, self.total_epochs)

        weight_sum = sum(max(float(weights.get(t, 0.0)), 0.0) for t in available_tasks)
        if weight_sum <= 0:
            norm_weights = {t: 1.0 / len(available_tasks) for t in available_tasks}
        else:
            norm_weights = {
                t: max(float(weights.get(t, 0.0)), 0.0) / weight_sum
                for t in available_tasks
            }

        counts = {t: int(total_batches * norm_weights[t]) for t in available_tasks}
        remaining = total_batches - sum(counts.values())
        fractions = sorted(
            available_tasks,
            key=lambda t: (total_batches * norm_weights[t]) - counts[t],
            reverse=True,
        )
        for t in fractions[:remaining]:
            counts[t] += 1

        selected: list[list[int]] = []
        for task_id in available_tasks:
            batches = task_batches[task_id]
            if self.shuffle:
                order = torch.randperm(len(batches), generator=g).tolist()
                batches = [batches[i] for i in order]

            count = counts[task_id]
            if count <= len(batches):
                selected.extend(batches[:count])
                continue

            selected.extend(batches)
            extra = count - len(batches)
            picks = torch.randint(len(batches), (extra,), generator=g).tolist()
            selected.extend(batches[i] for i in picks)

        return selected


def _allocate_eval_counts(total: int, task_ids: list, capacities: dict) -> dict:
    if total <= 0 or not task_ids:
        return {}

    counts: dict[int, int] = {}
    base = total // len(task_ids)
    remainder = total % len(task_ids)
    for pos, task_id in enumerate(task_ids):
        target = base + (1 if pos < remainder else 0)
        counts[task_id] = min(target, capacities.get(task_id, 0))

    leftover = total - sum(counts.values())
    while leftover > 0:
        progressed = False
        for task_id in task_ids:
            if counts[task_id] >= capacities.get(task_id, 0):
                continue
            counts[task_id] += 1
            leftover -= 1
            progressed = True
            if leftover == 0:
                break
        if not progressed:
            break
    return counts


def _eval_op_style_key(sample: dict) -> str:
    operation = str(sample.get("operation") or "unknown")
    style = str(sample.get("style") or "unknown")
    return f"{operation}/{style}"


def _sample_eval_indices_by_op_style(dataset, indices: list[int], count: int, rng: random.Random) -> list[int]:
    if count <= 0:
        return []
    if count >= len(indices):
        selected = list(indices)
        rng.shuffle(selected)
        return selected

    by_op_style: dict[str, list[int]] = defaultdict(list)
    for idx in indices:
        by_op_style[_eval_op_style_key(dataset[idx])].append(idx)

    strata = sorted(by_op_style)
    rng.shuffle(strata)
    capacities = {stratum: len(by_op_style[stratum]) for stratum in strata}
    counts = _allocate_eval_counts(count, strata, capacities)

    selected: list[int] = []
    for stratum in strata:
        n = counts.get(stratum, 0)
        if n <= 0:
            continue
        selected.extend(rng.sample(by_op_style[stratum], n))
    rng.shuffle(selected)
    return selected


def _eval_sample_summary(samples: list[dict]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for sample in samples:
        counts[_eval_op_style_key(sample)] += 1
    return ", ".join(f"{key}={counts[key]}" for key in sorted(counts))


def _build_eval_sample_groups(
    dataset,
    total_samples: int,
    samples_per_task: int,
    task_ids: list[int],
    seed: int,
) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """Select the fixed generation-eval subset.

    Fully deterministic for a given (dataset, seed, counts): stratified by
    op/style with a dedicated ``random.Random(seed + 1009)`` instance, never
    the global RNG. Also returns a manifest (dataset indices + op/style per
    group) so the exact eval subset can be audited and compared across runs.
    """
    by_task: dict[int, list[int]] = defaultdict(list)
    task_filter = set(task_ids)
    for idx in range(len(dataset)):
        task_id = int(dataset.task_of(idx))
        if task_id in task_filter:
            by_task[task_id].append(idx)

    available_tasks = [task_id for task_id in task_ids if by_task.get(task_id)]
    if not available_tasks:
        return {}, {}

    capacities = {task_id: len(by_task[task_id]) for task_id in available_tasks}
    if samples_per_task > 0:
        counts = {
            task_id: min(samples_per_task, capacities[task_id])
            for task_id in available_tasks
        }
    else:
        counts = _allocate_eval_counts(total_samples, available_tasks, capacities)

    rng = random.Random(seed + 1009)
    sample_groups: dict[str, list[dict]] = {}
    manifest: dict[str, list[dict]] = {}
    for task_id in available_tasks:
        count = counts.get(task_id, 0)
        if count <= 0:
            continue
        indices = _sample_eval_indices_by_op_style(dataset, by_task[task_id], count, rng)
        sample_groups[f"task{task_id}"] = [dataset[i] for i in indices]
        manifest[f"task{task_id}"] = [
            {
                "dataset_index": int(i),
                "operation": str(dataset[i].get("operation")),
                "style": str(dataset[i].get("style")),
                "num_chunks": int(dataset[i].get("num_chunks", 0)),
            }
            for i in indices
        ]
    return sample_groups, manifest


def _load_memory_dataset(
    *,
    data_dir: str,
    tokenized_data_dir: str,
    tokenizer,
    task_ids: list[int],
    max_samples_per_task: int,
    max_total_tokens: int,
):
    if tokenized_data_dir:
        return TokenizedMemoryDataset(
            tokenized_data_dir,
            tasks=task_ids,
            max_samples_per_task=max_samples_per_task,
            max_total_tokens=max_total_tokens,
        )
    return MemoryDataset(
        data_dir,
        tokenizer,
        tasks=task_ids,
        max_samples_per_task=max_samples_per_task,
        max_total_tokens=max_total_tokens,
    )


def _infer_valid_data_dir(data_dir: str) -> str:
    path = Path(data_dir)
    valid_sibling = path.parent / "valid"
    if path.name == "train" and valid_sibling.is_dir():
        return str(valid_sibling)
    return data_dir


def _infer_valid_tokenized_data_dir(tokenized_data_dir: str) -> str:
    if not tokenized_data_dir:
        return ""

    path = Path(tokenized_data_dir)
    candidates = []
    text = str(path)
    if "-train-" in text:
        candidates.append(Path(text.replace("-train-", "-valid-", 1)))
    if path.name == "train":
        candidates.append(path.parent / "valid")

    for candidate in candidates:
        if (candidate / "manifest.json").is_file():
            return str(candidate)
    return ""


# ── Main ───────────────────────────────────────────────────────────

def train(args):
    _startup_probe("train() enter")

    if args.resume_from_checkpoint and args.init_from_checkpoint:
        raise ValueError("Use only one of --resume_from_checkpoint or --init_from_checkpoint.")
    args.resume_from_checkpoint = _resolve_resume_checkpoint(
        args.resume_from_checkpoint, args.output_dir,
    )
    _patch_torch_load_safety_for_trusted_resume(args.resume_from_checkpoint)

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_main = (local_rank <= 0)

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        log_path = setup_file_logging(logger, args.output_dir, args.log_file)
        logger.info(f"Training log → {log_path}")
        save_code_snapshot(args.output_dir, project_root=Path(__file__).resolve().parent.parent)

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    if local_rank >= 0:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device)

    if args.deepspeed:
        _enable_zero3_memory_efficient_linear()

    # ── Model ─────────────────────────────────────────────────
    checkpoint_to_load = args.init_from_checkpoint or args.resume_from_checkpoint
    if checkpoint_to_load:
        # Build the model from the checkpoint's own config (delta checkpoints
        # rebuild the frozen backbone from --model_path, then overlay the
        # trained Metis weights).  The architecture recorded in the checkpoint
        # is the source of truth; CLI --metis_* flags are ignored here.
        mode = "resume" if args.resume_from_checkpoint else "init"
        if is_main:
            logger.info(
                "Loading Metis %s checkpoint: %s  base_model=%s",
                mode, checkpoint_to_load, args.model_path,
            )
        model = load_metis_model_from_checkpoint(
            checkpoint_to_load,
            model_path=args.model_path,
            backbone_type=args.backbone_type,
            device=device,
            dtype=dtype,
        )
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_to_load, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        if is_main:
            logger.info(f"Loading backbone: {args.model_path}  type={args.backbone_type}")
        model, tokenizer = load_metis_from_backbone(
            args.model_path,
            backbone_type=args.backbone_type,
            device=device,
            dtype=dtype,
            metis_block_type=args.metis_block_type,
            metis_hyper_memory_type=args.metis_hyper_memory_type,
            metis_local_memory_type=args.metis_local_memory_type,
            update_ratio=args.update_ratio,
            commit_hidden_offset=args.commit_hidden_offset,
            mem_norm_init=args.mem_norm_init,
            uniform_num_selected=args.uniform_num_selected,
            stride_interval=args.stride_interval,
            pool_temperature=args.pool_temperature,
            gumbel_topk_noise=args.gumbel_topk_noise,
            alpha_top_p=args.alpha_top_p,
            alpha_min_tokens=args.alpha_min_tokens,
            alpha_max_tokens=args.alpha_max_tokens,
            alpha_max_fraction=args.alpha_max_fraction,
            gated_delta_alpha_init=args.gated_delta_alpha_init,
            gated_delta_beta_init=args.gated_delta_beta_init,
            qk_kernel_type=args.qk_kernel_type,
            metis_reweight_gamma=args.metis_reweight_gamma,
        )
    if is_main:
        logger.info(
            f"memory_configs: "
            f"block={args.metis_block_type} hyper={args.metis_hyper_memory_type} "
            f"local={args.metis_local_memory_type} "
            f"update_ratio={args.update_ratio} "
            f"commit_hidden_offset={args.commit_hidden_offset} "
            f"stride_interval={args.stride_interval} "
            f"pool_temperature={args.pool_temperature} "
            f"gated_delta_alpha_init={args.gated_delta_alpha_init} "
            f"gated_delta_beta_init={args.gated_delta_beta_init} "
            f"mem_norm_init={args.mem_norm_init}"
        )

    hm_count = freeze_backbone(model)
    logger.info(f"Unfroze {hm_count} hyper-memory parameter tensors")

    use_lora = args.lora_r > 0
    if use_lora:
        targets = [t.strip() for t in args.lora_target_modules.split(",")]
        replaced = apply_lora(model, targets, args.lora_r, args.lora_alpha, args.lora_dropout)
        logger.info(f"LoRA r={args.lora_r} alpha={args.lora_alpha} -> {len(replaced)} modules")
    else:
        logger.info("LoRA disabled — training hyper-memory only")

    total, trainable = count_params(model)
    logger.info(f"Params  total={total:,}  trainable={trainable:,} ({100*trainable/max(total,1):.2f}%)")

    model.gradient_checkpointing_disable()
    disable_act_ckpt = os.environ.get("METIS_DISABLE_ACT_CKPT", "0").lower() in {
        "1", "true", "yes", "on",
    }
    if args.deepspeed and not disable_act_ckpt:
        gc_blocks = 0
        for block in model.model.metis_blocks:
            dec = getattr(block, "backbone_decoder", None)
            if dec is not None:
                dec._use_gradient_checkpointing = True
                gc_blocks += 1
        logger.info(
            "Enabled Metis selective activation checkpointing on %d blocks "
            "(ZeRO-3 backbone-weight memory fix)",
            gc_blocks,
        )
    elif args.deepspeed and disable_act_ckpt:
        logger.warning(
            "METIS_DISABLE_ACT_CKPT=1 -> selective activation checkpointing disabled "
            "(control experiment; 27B ZeRO-3 may OOM)"
        )

    # ── Data ──────────────────────────────────────────────────
    task_ids = [int(t) for t in args.tasks.split(",")]
    dataset = _load_memory_dataset(
        data_dir=args.data_dir,
        tokenized_data_dir=args.tokenized_data_dir,
        tokenizer=tokenizer,
        task_ids=task_ids,
        max_samples_per_task=args.max_samples_per_task,
        max_total_tokens=args.max_total_tokens,
    )
    if len(dataset) == 0:
        source = args.tokenized_data_dir or args.data_dir
        raise RuntimeError(f"No samples loaded from {source}. Check --tasks and data files.")

    # ── Task scheduler ────────────────────────────────────────
    task1_weight_start = args.task1_weight_start
    task1_weight_end = args.task1_weight_end
    if args.task1_weight is not None:
        task1_weight_start = args.task1_weight
        task1_weight_end = args.task1_weight

    scheduler = TaskScheduler(
        task0_weight_start=args.task0_weight_start,
        task0_weight_end=args.task0_weight_end,
        task1_weight_start=task1_weight_start,
        task1_weight_end=task1_weight_end,
        task2_weight_start=args.task2_weight_start,
        task2_weight_end=args.task2_weight_end,
        task3_weight_start=args.task3_weight_start,
        task3_weight_end=args.task3_weight_end,
        task4_weight_start=args.task4_weight_start,
        task4_weight_end=args.task4_weight_end,
    )
    if is_main:
        logger.info(scheduler.schedule_description)

    # ── Wandb ─────────────────────────────────────────────────
    if args.wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_run_name:
            os.environ["WANDB_NAME"] = args.wandb_run_name

    # ── TrainingArguments ─────────────────────────────────────
    train_sampling_kwargs = {}

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        # max_steps > 0 takes precedence over num_train_epochs (HF semantics);
        # 0 or negative means "train by epochs".
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        lr_scheduler_type="constant_with_warmup",
        logging_steps=args.log_steps,
        eval_strategy="no",   # eval is handled via the callback
        save_strategy="steps",
        save_steps=args.save_steps,
        save_only_model=False,
        bf16=(args.dtype == "bfloat16"),
        fp16=(args.dtype == "float16"),
        report_to=["wandb"] if args.wandb else ["none"],
        run_name=args.wandb_run_name,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        ddp_backend="nccl",
        ddp_find_unused_parameters=False,
        deepspeed=args.deepspeed,
        seed=args.seed,
        **train_sampling_kwargs,
    )

    # ── Build eval samples (from a small held-out subset) ─────
    eval_samples = None
    eval_callback = None
    if args.eval_steps > 0:
        eval_data_dir = args.eval_data_dir or _infer_valid_data_dir(args.data_dir)
        eval_tokenized_data_dir = (
            args.eval_tokenized_data_dir
            if args.eval_tokenized_data_dir is not None
            else _infer_valid_tokenized_data_dir(args.tokenized_data_dir)
        )
        eval_dataset = _load_memory_dataset(
            data_dir=eval_data_dir,
            tokenized_data_dir=eval_tokenized_data_dir,
            tokenizer=tokenizer,
            task_ids=task_ids,
            max_samples_per_task=args.eval_max_samples_per_task,
            max_total_tokens=args.eval_max_total_tokens
            if args.eval_max_total_tokens >= 0
            else args.max_total_tokens,
        )
        if len(eval_dataset) == 0:
            source = eval_tokenized_data_dir or eval_data_dir
            raise RuntimeError(f"No eval samples loaded from {source}. Check eval data args.")
        if is_main:
            logger.info(
                "Using generation eval dataset: %s",
                eval_tokenized_data_dir or eval_data_dir,
            )
        eval_samples, eval_manifest = _build_eval_sample_groups(
            eval_dataset,
            total_samples=args.eval_samples,
            samples_per_task=args.eval_samples_per_task,
            task_ids=task_ids,
            seed=args.seed,
        )
        if is_main:
            logger.info(
                "Generation eval samples by task: "
                + ", ".join(f"{name}={len(samples)}" for name, samples in sorted(eval_samples.items()))
            )
            for name, samples in sorted(eval_samples.items()):
                logger.info("  %s eval op/style: %s", name, _eval_sample_summary(samples))
            # Persist the exact eval subset (dataset indices + op/style) so it
            # can be audited and byte-compared across runs.
            manifest_path = os.path.join(args.output_dir, "eval_samples_manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "eval_dataset": eval_tokenized_data_dir or eval_data_dir,
                        "seed": args.seed,
                        "groups": eval_manifest,
                    },
                    f, indent=2, ensure_ascii=False,
                )
            logger.info("Eval sample manifest → %s", manifest_path)
        eval_callback = MetisEvalCallback(
            model, tokenizer, eval_samples, device,
            eval_config={
                "max_new_tokens": args.max_new_tokens_eval,
                "max_samples": args.eval_samples,
                "num_examples": 3,
            },
            output_dir=args.output_dir,
            eval_steps=args.eval_steps,
            gen_eval_steps=args.gen_eval_steps,
            zero3_sync_eval=bool(args.deepspeed),
        )

    # ── Trainer ───────────────────────────────────────────────
    def _build_lr_groups(model, base_lr: float):
        """Optional per-module learning rates for Metis components.

        Every group defaults to ``base_lr``, so training dynamics are
        unchanged unless a METIS_*_LR env var is explicitly set:

            METIS_POOL_SCORE_LR  token-selection scorer (pool_score)
            METIS_QUERY_LR       learned read query (query_proj / query_norm)
            METIS_MEM_NORM_LR    memory branch RMSNorm (mem_norm)
            METIS_MEMORY_W_LR    write projections (W_k / W_v)
        """
        def _float_env(name: str, default: float) -> float:
            raw = os.environ.get(name)
            if raw is None or raw == "":
                return default
            try:
                return float(raw)
            except ValueError as exc:
                raise ValueError(f"{name} must be a float, got {raw!r}") from exc

        groups = {
            "pool_score": {"params": [], "lr": _float_env("METIS_POOL_SCORE_LR", base_lr)},
            "query": {"params": [], "lr": _float_env("METIS_QUERY_LR", base_lr)},
            "mem_norm": {"params": [], "lr": _float_env("METIS_MEM_NORM_LR", base_lr)},
            "memory_w": {"params": [], "lr": _float_env("METIS_MEMORY_W_LR", base_lr)},
            "default": {"params": [], "lr": base_lr},
        }

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "pool_score" in name:
                groups["pool_score"]["params"].append(p)
            elif "query_proj" in name or "query_norm" in name:
                groups["query"]["params"].append(p)
            elif "mem_norm" in name:
                groups["mem_norm"]["params"].append(p)
            elif ".W_k." in name or ".W_v." in name:
                groups["memory_w"]["params"].append(p)
            else:
                groups["default"]["params"].append(p)

        lr_groups = [g for g in groups.values() if g["params"]]
        if is_main:
            logger.info(
                "Metis LR groups: %s",
                ", ".join(
                    f"{key}({sum(p.numel() for p in g['params']):,} params, lr={g['lr']:.1e})"
                    for key, g in groups.items()
                    if g["params"]
                ),
            )
        return lr_groups

    optimizer_cls_and_kwargs = None
    if args.dtype == "bfloat16" and not args.deepspeed:
        optimizer_cls_and_kwargs = (
            MasterWeightAdamW,
            {
                "params": _build_lr_groups(model, args.learning_rate),
                "lr": args.learning_rate,
                "betas": (args.adam_beta1, args.adam_beta2),
                "eps": args.adam_epsilon,
            },
        )
        logger.info("Using MasterWeightAdamW for bf16 trainable parameters")

    trainer = MetisMemoryTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=build_collate_fn(tokenizer.pad_token_id),
        metis_tokenizer=tokenizer,
        checkpoint_save_mode=args.checkpoint_save_mode,
        base_model_path=args.model_path,
        callbacks=[FileLoggingCallback()] + ([eval_callback] if eval_callback else []),
        optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
    )

    if local_rank >= 0:
        sampler_replicas = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        sampler_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    else:
        sampler_replicas = 1
        sampler_rank = 0
    trainer.set_bucket_sampler_factory(
        lambda ds: TaskWeightedLengthDistributedSampler(
            ds,
            num_replicas=sampler_replicas,
            rank=sampler_rank,
            batch_size=args.batch_size,
            task_scheduler=scheduler,
            total_epochs=args.num_epochs,
            epoch_getter=lambda: trainer.state.epoch,
            shuffle=True,
            seed=args.seed,
        )
    )

    logger.info(
        f"Training  epochs={args.num_epochs}  batch_size={args.batch_size}  "
        f"accum={args.gradient_accumulation_steps}  lr={args.learning_rate}"
    )

    # ── Save step-0 checkpoint ────────────────────────────────
    step0_dir = os.path.join(args.output_dir, "checkpoint-0")
    if args.resume_from_checkpoint:
        logger.info("Skipping step-0 checkpoint while resuming from %s", args.resume_from_checkpoint)
    elif trainer.is_world_process_zero():
        os.makedirs(step0_dir, exist_ok=True)
        if args.deepspeed:
            # DeepSpeed engine is only built inside trainer.train(); pre-train
            # checkpoint-0 should save from the still-unwrapped module.
            trainer._save(step0_dir, state_dict=trainer.model.state_dict())
        else:
            # trainer._save respects checkpoint_save_mode (delta/full) and also
            # saves the tokenizer.
            trainer._save(step0_dir)
        logger.info(f"Step-0 checkpoint saved → {step0_dir}")

    # ── Log task weights per epoch ────────────────────────────
    for ep in range(args.num_epochs):
        w = scheduler.get_weights(ep, args.num_epochs)
        logger.info(
            f"  Epoch {ep+1}/{args.num_epochs} task_weights: "
            + " ".join(f"Task{task_id}={w[task_id]:.3f}" for task_id in sorted(w))
        )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    trainer.save_model(os.path.join(args.output_dir, "checkpoint_final"))
    logger.info("Training complete.")


# ── CLI ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Metis unified multi-task training")

    g = p.add_argument_group("Model")
    g.add_argument("--model_path", required=True)
    g.add_argument("--backbone_type", default="qwen3_5", choices=["qwen3_5", "qwen3", "llama"])
    g.add_argument("--metis_block_type", default="NormedReweightLearnedQueryMetisBlock")
    g.add_argument("--metis_hyper_memory_type", default="StraightThroughAlphaTopPGatedDeltaRuleMetisHyperMemory")
    g.add_argument("--metis_local_memory_type", default="NormalizedDeltaNetMetisLocalMemory")
    g.add_argument("--update_ratio", type=float, default=0.9)
    g.add_argument("--commit_hidden_offset", type=int, default=0, choices=[0, 1])
    g.add_argument("--mem_norm_init", type=float, default=1.0)
    g.add_argument("--uniform_num_selected", type=int, default=16)
    g.add_argument("--stride_interval", type=int, default=8)
    g.add_argument("--pool_temperature", type=float, default=1.0)
    g.add_argument("--gumbel_topk_noise", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--alpha_top_p", type=float, default=0.9)
    g.add_argument("--alpha_min_tokens", type=int, default=1)
    g.add_argument("--alpha_max_tokens", type=int, default=0)
    g.add_argument("--alpha_max_fraction", type=float, default=0.0)
    g.add_argument("--gated_delta_alpha_init", type=float, default=1.0)
    g.add_argument("--gated_delta_beta_init", type=float, default=1.0)
    g.add_argument("--qk_kernel_type", default="elu_plus_one",
                   choices=["elu_plus_one", "relu_square", "softplus"])
    g.add_argument("--metis_reweight_gamma", type=float, default=0.9)

    g = p.add_argument_group("Data")
    g.add_argument("--data_dir", default="data/train_data")
    g.add_argument("--tokenized_data_dir", default="", help="Optional pre-tokenized dataset cache directory")
    g.add_argument("--eval_data_dir", default="",
                   help="Raw validation data directory for generation eval. If empty and --data_dir ends in train, uses sibling valid.")
    g.add_argument("--eval_tokenized_data_dir", default=None,
                   help="Optional pre-tokenized validation cache for generation eval. If unset, inferred from --tokenized_data_dir when available; empty string forces raw eval_data_dir.")
    g.add_argument("--tasks", default="0,1,2,3,4", help="Comma-separated task ids to train")
    g.add_argument("--max_samples_per_task", type=int, default=0, help="0 = all")
    g.add_argument("--max_total_tokens", type=int, default=0, help="0 = no length filter")
    g.add_argument("--eval_max_samples_per_task", type=int, default=0,
                   help="0 = all validation samples before eval sampling")
    g.add_argument("--eval_max_total_tokens", type=int, default=-1,
                   help="-1 = use --max_total_tokens for validation; 0 = no validation length filter")

    g = p.add_argument_group("Training")
    g.add_argument("--batch_size", type=int, default=2)
    g.add_argument("--num_epochs", type=int, default=200)
    g.add_argument("--max_steps", type=int, default=0,
                   help="If > 0, stop after this many optimizer steps (overrides --num_epochs).")
    g.add_argument("--gradient_accumulation_steps", type=int, default=60)
    g.add_argument("--learning_rate", type=float, default=1e-4)
    g.add_argument("--weight_decay", type=float, default=0.01)
    g.add_argument("--warmup_steps", type=int, default=1160)
    g.add_argument("--max_grad_norm", type=float, default=1.0)
    g.add_argument("--adam_beta1", type=float, default=0.9)
    g.add_argument("--adam_beta2", type=float, default=0.999)
    g.add_argument("--adam_epsilon", type=float, default=1e-8)

    g = p.add_argument_group("Task Scheduler")
    g.add_argument("--task0_weight_start", type=float, default=0.25)
    g.add_argument("--task0_weight_end", type=float, default=0.1)
    g.add_argument("--task1_weight_start", type=float, default=0.35)
    g.add_argument("--task1_weight_end", type=float, default=0.25)
    g.add_argument("--task1_weight", type=float, default=None,
                   help="Deprecated: if set, uses one fixed weight for task1.")
    g.add_argument("--task2_weight_start", type=float, default=0.2)
    g.add_argument("--task2_weight_end", type=float, default=0.3)
    g.add_argument("--task3_weight_start", type=float, default=0.1)
    g.add_argument("--task3_weight_end", type=float, default=0.2)
    g.add_argument("--task4_weight_start", type=float, default=0.1)
    g.add_argument("--task4_weight_end", type=float, default=0.15)

    g = p.add_argument_group("LoRA")
    g.add_argument("--lora_r", type=int, default=0)
    g.add_argument("--lora_alpha", type=int, default=32)
    g.add_argument("--lora_dropout", type=float, default=0.1)
    g.add_argument("--lora_target_modules", default="q_proj,v_proj,o_proj")

    g = p.add_argument_group("Device")
    g.add_argument("--device", default="cuda:0")
    g.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])

    g = p.add_argument_group("Logging / Save / Eval")
    g.add_argument("--log_file", default="train.log")
    g.add_argument("--log_steps", type=int, default=5)
    g.add_argument("--save_steps", type=int, default=500)
    g.add_argument("--eval_steps", type=int, default=0,
                   help="Frequency of eval (0 = off). Cheap loss eval runs every this many steps.")
    g.add_argument("--gen_eval_steps", type=int, default=0,
                   help="Frequency of the expensive generation eval, decoupled from --eval_steps "
                        "(0 = run generation every eval; >0 = only when step %% gen_eval_steps == 0). "
                        "Should be a multiple of --eval_steps to take effect.")
    g.add_argument("--eval_samples", type=int, default=20)
    g.add_argument("--eval_samples_per_task", type=int, default=0,
                   help="If >0, sample this many generation-eval examples per task. Otherwise split --eval_samples evenly across tasks.")
    g.add_argument("--max_new_tokens_eval", type=int, default=256)
    g.add_argument("--checkpoint_save_mode", default="delta", choices=["delta", "full"],
                   help="delta: save only trainable Metis weights (+manifest, default); "
                        "full: legacy full save_pretrained dump.")

    g = p.add_argument_group("Wandb")
    g.add_argument("--wandb", action="store_true")
    g.add_argument("--wandb_project", default="metis_training")
    g.add_argument("--wandb_run_name", default=None)

    g = p.add_argument_group("Reproducibility")
    g.add_argument("--seed", type=int, default=42)

    g = p.add_argument_group("Output")
    g.add_argument("--output_dir", default="checkpoint/experiment_1")
    g.add_argument("--resume_from_checkpoint", default="",
                   help="Resume a full training state (weights + optimizer + scheduler + RNG + step). "
                        "Pass a checkpoint dir, or 'auto'/'latest' to pick the newest resumable "
                        "checkpoint-N under --output_dir (starts fresh if none found).")
    g.add_argument("--init_from_checkpoint", default="",
                   help="Warm-start weights only from a checkpoint (delta or full); optimizer, "
                        "scheduler and step counter start fresh. Mutually exclusive with resume.")
    g.add_argument("--deepspeed", default=None, help="Path to DeepSpeed config JSON")
    g.add_argument("--local_rank", type=int, default=-1)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train(args)
