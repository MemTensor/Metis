"""Custom HuggingFace Trainer for Metis multi-chunk memory training.

Key difference from standard Trainer: ``compute_loss`` runs through all
dialogue chunks sequentially, committing memory at each step, with loss
computed only on the eval (query) chunk.  Gradients flow through the
entire multi-chunk computation graph.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import Trainer, TrainerCallback, TrainingArguments

from .losses import IGNORE_INDEX, compute_loss as compute_task_loss
from .metrics import Metrics

from metis.checkpoint_utils import (
    is_delta_checkpoint,
    load_metis_delta_into_model,
    save_metis_delta_checkpoint,
)

logger = logging.getLogger(__name__)


class MetisMemoryTrainer(Trainer):
    """Custom Trainer for Metis multi-chunk memory training.

    In ``compute_loss`` the model processes all chunks of a sample
    sequentially.  Memory is committed after every chunk (via the
    ``commit_memory=True`` flag in the forward pass).  CE loss is only
    computed on the eval chunk's labelled tokens.
    """

    def __init__(
        self,
        *args,
        metis_tokenizer=None,
        checkpoint_save_mode: str = "full",
        base_model_path: str = "",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._metis_tokenizer = metis_tokenizer
        # "delta": save only trainable Metis weights (backbone is frozen, so a
        # full dump mostly duplicates immutable backbone weights). "full":
        # legacy save_pretrained dump, loadable via from_pretrained directly.
        self._checkpoint_save_mode = checkpoint_save_mode
        self._base_model_path = base_model_path
        self._last_alpha_stats_log_step = -1
        self._last_query_norm_log_step = -1
        self._bucket_sampler_factory = None
        self._task_loss_sums: dict[int, float] = defaultdict(float)
        self._task_loss_tokens: dict[int, int] = defaultdict(int)
        self._task_loss_batches: dict[int, int] = defaultdict(int)

    def set_bucket_sampler_factory(self, factory):
        self._bucket_sampler_factory = factory

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if self._bucket_sampler_factory is None:
            return super().get_train_dataloader()

        sampler = self._bucket_sampler_factory(self.train_dataset)
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def floating_point_ops(self, inputs):
        input_ids = inputs.get("input_ids") if isinstance(inputs, dict) else None
        if isinstance(input_ids, list):
            return 0
        return super().floating_point_ops(inputs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # `unwrapped` is only for non-autograd attribute access (reset, stats).
        # The forward pass MUST go through `model` (the DDP/DeepSpeed wrapper):
        # calling model.module(...) skips the wrapper's per-iteration reducer
        # setup, so gradients are silently never synchronized across ranks.
        unwrapped = model.module if hasattr(model, "module") else model

        unwrapped.reset()

        num_chunks = inputs["num_chunks"]
        eval_idx = inputs["eval_chunk_idx"]
        tasks = inputs["task"]
        operations = inputs.get("operation", [None] * len(tasks))
        task_id = tasks[0]  # all samples in a batch share the same task

        total_loss = torch.tensor(0.0, device=self.args.device)

        for t in range(num_chunks):
            ids = inputs["input_ids"][t]
            attn = inputs["attention_mask"][t]

            # Label-free chunks exist only to write memory; their logits are
            # never read. Skip the full lm_head projection for them
            # (logits_to_keep=1 keeps a single position): with V≈248k the
            # lm_head is ~11-13% of forward FLOPs and the full [B,S,V] logits
            # tensor dominates transient activation memory (~2.5GB at bs5).
            lbls = inputs["labels"][t]
            has_labels = int((lbls[..., 1:] != IGNORE_INDEX).sum().item()) > 0

            outputs = model(
                input_ids=ids,
                attention_mask=attn,
                commit_memory=True,
                use_cache=False,
                logits_to_keep=0 if has_labels else 1,
            )

            if not has_labels:
                continue

            loss = compute_task_loss(
                outputs.logits, lbls,
                task_id=task_id,
                operation=operations[0] if operations else None,
            )
            self._record_task_loss(task_id, loss, lbls)
            total_loss = total_loss + loss

        self._maybe_log_alpha_stats(unwrapped)
        self._maybe_log_query_norm(unwrapped)

        return (total_loss, None) if return_outputs else total_loss

    def _record_task_loss(self, task_id: int, loss: torch.Tensor, labels: torch.Tensor) -> None:
        if not torch.isfinite(loss):
            return
        label_tokens = int((labels[..., 1:] != IGNORE_INDEX).sum().item())
        if label_tokens <= 0:
            return
        task_id = int(task_id)
        self._task_loss_sums[task_id] += float(loss.detach().item()) * label_tokens
        self._task_loss_tokens[task_id] += label_tokens
        self._task_loss_batches[task_id] += 1

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        logs = dict(logs)
        if "loss" in logs:
            logs.update(self._drain_task_loss_logs())
        super().log(logs, start_time=start_time)

    def _drain_task_loss_logs(self) -> dict[str, float]:
        if not self._task_loss_tokens:
            return {}

        task_ids = [0, 1, 2, 3, 4]
        stats = torch.tensor(
            [
                [
                    self._task_loss_sums.get(task_id, 0.0),
                    float(self._task_loss_tokens.get(task_id, 0)),
                    float(self._task_loss_batches.get(task_id, 0)),
                ]
                for task_id in task_ids
            ],
            dtype=torch.float64,
            device=self.args.device,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        logs: dict[str, float] = {}
        for row, task_id in zip(stats.detach().cpu().tolist(), task_ids):
            loss_sum, token_count, batch_count = row
            if token_count <= 0:
                continue
            prefix = f"train/task{task_id}"
            logs[f"{prefix}_loss"] = loss_sum / token_count
            logs[f"{prefix}_loss_tokens"] = token_count
            logs[f"{prefix}_loss_batches"] = batch_count

        self._task_loss_sums.clear()
        self._task_loss_tokens.clear()
        self._task_loss_batches.clear()
        return logs

    def _maybe_log_alpha_stats(self, model) -> None:
        if self.args.process_index != 0:
            return
        step = int(self.state.global_step)
        if step == self._last_alpha_stats_log_step:
            return
        if self.args.logging_steps > 0 and step % self.args.logging_steps != 0:
            return

        stats_by_key: dict[str, list[float]] = {}
        for block in getattr(model.model, "metis_blocks", []):
            hyper = getattr(block, "hyper_memory", None)
            stats = getattr(hyper, "last_alpha_stats", None)
            if not stats:
                continue
            for key, value in stats.items():
                stats_by_key.setdefault(key, []).append(float(value))

        if not stats_by_key:
            return

        logs = {
            f"alpha_top_p/{key}": sum(values) / len(values)
            for key, values in stats_by_key.items()
            if values
        }
        self.log(logs)
        self._last_alpha_stats_log_step = step

    def _maybe_log_query_norm(self, model) -> None:
        if not getattr(model, "training", False):
            return
        step = int(self.state.global_step)
        if step == self._last_query_norm_log_step:
            return
        if self.args.logging_steps > 0 and step % self.args.logging_steps != 0:
            return

        values: list[float] = []
        for block in getattr(model.model, "metis_blocks", []):
            norm = getattr(block, "last_memory_query_norm", None)
            if norm is not None:
                values.append(float(norm))

        stats = torch.tensor(
            [sum(values), float(len(values))],
            dtype=torch.float64,
            device=self.args.device,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        self._last_query_norm_log_step = step
        if self.args.process_index != 0:
            return

        total, count = stats.detach().cpu().tolist()
        if count <= 0:
            return
        self.log({"memory_query/norm": total / count})

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss, None, None)

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        unwrapped = self.model.module if hasattr(self.model, "module") else self.model

        if state_dict is None and getattr(self, "is_deepspeed_enabled", False):
            state_dict = self.accelerator.get_state_dict(self.model_wrapped)

        if self.args.should_save:
            if self._checkpoint_save_mode == "delta":
                save_metis_delta_checkpoint(
                    unwrapped,
                    output_dir,
                    tokenizer=self._metis_tokenizer,
                    base_model_path=self._base_model_path,
                    state_dict=state_dict,
                )
            elif self._checkpoint_save_mode == "full":
                unwrapped.save_pretrained(output_dir, state_dict=state_dict)
                if self._metis_tokenizer is not None:
                    self._metis_tokenizer.save_pretrained(output_dir)
            else:
                raise ValueError(f"Unsupported checkpoint_save_mode={self._checkpoint_save_mode!r}")
            self._save_training_logs(output_dir)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        if is_delta_checkpoint(resume_from_checkpoint):
            load_metis_delta_into_model(model or self.model, resume_from_checkpoint)
            return
        return super()._load_from_checkpoint(resume_from_checkpoint, model=model)

    def _save_training_logs(self, output_dir):
        info = {
            "global_step": self.state.global_step,
            "epoch": self.state.epoch,
            "best_metric": self.state.best_metric,
            "best_model_checkpoint": self.state.best_model_checkpoint,
            "log_history": self.state.log_history,
        }
        with open(os.path.join(output_dir, "training_info.json"), "w") as f:
            json.dump(info, f, indent=2, default=str)
        logger.info(f"Saved training logs (step {self.state.global_step}, epoch {self.state.epoch:.2f})")


class MetisEvalCallback(TrainerCallback):
    """Runs generation-based evaluation every ``eval_steps`` optimizer steps.

    Metrics are appended to ``{output_dir}/eval_metrics.jsonl`` and
    optionally logged to wandb.
    """

    def __init__(self, model, tokenizer, eval_samples, device,
                 eval_config: dict, output_dir: str = ".", eval_steps: int = 0,
                 gen_eval_steps: int = 0, zero3_sync_eval: bool = False):
        self._model = model
        self._tokenizer = tokenizer
        self._eval_sample_groups = self._normalize_eval_samples(eval_samples)
        self._device = device
        self._eval_config = eval_config
        self._output_dir = output_dir
        self._metrics = Metrics()
        self._eval_steps = int(eval_steps)
        # Frequency of the expensive generation eval, decoupled from the cheap
        # loss eval. Generation eval dominates eval wall-clock (autoregressive
        # decoding, one forward per token). 0 = run generation every time loss
        # eval runs (legacy behaviour). >0 = only run generation when
        # step % gen_eval_steps == 0; loss eval still runs every eval_steps.
        # global_step is identical across ranks, so the gate is collective-safe.
        self._gen_eval_steps = int(gen_eval_steps)
        # ZeRO-3 generation/loss eval must execute on every rank so parameter
        # all-gather collectives stay aligned. Plain DDP keeps the legacy
        # rank-0-only eval path.
        self._zero3_sync_eval = bool(zero3_sync_eval)
        self._last_eval_step = -1

    def on_step_end(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if self._eval_steps <= 0 or step <= 0:
            return control
        if step == self._last_eval_step or step % self._eval_steps != 0:
            return control

        self._last_eval_step = step
        try:
            if self._should_run_eval_on_this_rank(state) and self._has_eval_samples():
                self._evaluate_and_log(
                    state,
                    should_log=state.is_world_process_zero,
                    run_generation=self._should_run_generation(step),
                )
        finally:
            self._barrier()
        return control

    def on_evaluate(self, args, state, control, **kwargs):
        try:
            if self._should_run_eval_on_this_rank(state) and self._has_eval_samples():
                self._evaluate_and_log(
                    state,
                    should_log=state.is_world_process_zero,
                    run_generation=self._should_run_generation(int(state.global_step)),
                )
        finally:
            self._barrier()
        return control

    def _should_run_generation(self, step: int) -> bool:
        """Whether to run the expensive generation eval at this step.

        Must return the SAME value on every rank (depends only on step and the
        configured cadence) so generation either runs on all ranks or none;
        otherwise ZeRO-3 param all-gathers can desync.
        """
        if self._gen_eval_steps <= 0:
            return True
        return step % self._gen_eval_steps == 0

    def _should_run_eval_on_this_rank(self, state) -> bool:
        return self._zero3_sync_eval or state.is_world_process_zero

    def _evaluate_and_log(self, state, should_log: bool = True,
                          run_generation: bool = True) -> None:
        if should_log:
            tag = "loss+gen" if run_generation else "loss only"
            logger.info("Running eval at step %s (%s)", state.global_step, tag)
        loss_results = evaluate_loss_groups(
            self._model, self._eval_sample_groups, self._device,
            log_results=should_log,
        )
        generation_results = {}
        if run_generation:
            generation_results = evaluate_generation_groups(
                self._model, self._tokenizer, self._eval_sample_groups,
                self._device, metrics=self._metrics,
                log_results=should_log,
                **self._eval_config,
            )
        if not should_log:
            return
        results = {**loss_results, **generation_results}
        summary = {k: v for k, v in results.items() if "/" not in k}
        logger.info("Eval metrics: " + ", ".join(f"{k}={v:.4f}" for k, v in summary.items()))
        record = {"step": state.global_step, "epoch": state.epoch, **results}
        metrics_path = os.path.join(self._output_dir, "eval_metrics.jsonl")
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

        try:
            import wandb
            if wandb.run is not None:
                wandb.log(
                    {f"eval/{k}": v for k, v in results.items()},
                    step=state.global_step,
                )
        except Exception:
            pass

    @staticmethod
    def _normalize_eval_samples(eval_samples) -> dict[str, list[dict]]:
        if isinstance(eval_samples, dict):
            return {
                str(name): list(samples)
                for name, samples in eval_samples.items()
                if samples
            }
        return {"all": list(eval_samples or [])}

    def _has_eval_samples(self) -> bool:
        return any(self._eval_sample_groups.values())

    @staticmethod
    def _barrier() -> None:
        try:
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
        except Exception:
            pass


@torch.no_grad()
def evaluate_loss_groups(
    model,
    eval_sample_groups: dict[str, list[dict]],
    device: torch.device,
    log_results: bool = True,
) -> dict[str, float]:
    """Compute teacher-forcing eval loss for each sample group and overall."""
    results: dict[str, float] = {}
    total_loss_sum = 0.0
    total_tokens = 0.0
    total_samples = 0.0

    for group_name in sorted(eval_sample_groups):
        samples = eval_sample_groups[group_name]
        if not samples:
            continue

        group = evaluate_loss(
            model,
            samples,
            device,
            log_prefix=group_name,
            log_results=log_results,
        )
        for key, value in group.items():
            results[f"{group_name}/{key}"] = value

        total_loss_sum += group["loss"] * group["loss_num_tokens"]
        total_tokens += group["loss_num_tokens"]
        total_samples += group["loss_num_samples"]

    results["loss"] = total_loss_sum / total_tokens if total_tokens > 0 else 0.0
    results["loss_num_tokens"] = total_tokens
    results["loss_num_samples"] = total_samples
    return results


@torch.no_grad()
def evaluate_loss(
    model,
    eval_samples: list[dict],
    device: torch.device,
    log_prefix: str = "",
    log_results: bool = True,
) -> dict[str, float]:
    """Compute token-weighted teacher-forcing loss on encoded eval samples."""
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    token_count = 0
    sample_count = 0

    try:
        for idx, sample in enumerate(eval_samples):
            eval_idx = sample["eval_chunk_idx"]
            chunks = sample["chunks"]
            task_id = int(sample.get("task", 0))
            operation = sample.get("operation")

            model.reset()
            for t in range(eval_idx + 1):
                ids = chunks[t]["input_ids"].unsqueeze(0).to(device)
                attn = torch.ones_like(ids, device=device)
                outputs = model(
                    input_ids=ids,
                    attention_mask=attn,
                    commit_memory=True,
                    use_cache=False,
                    # Only the eval chunk's logits are read below.
                    logits_to_keep=0 if t == eval_idx else 1,
                )

                if t != eval_idx:
                    continue

                labels = chunks[t]["labels"].unsqueeze(0).to(device)
                labels_for_loss = int((labels[..., 1:] != IGNORE_INDEX).sum().item())
                if labels_for_loss <= 0:
                    continue

                loss = compute_task_loss(
                    outputs.logits,
                    labels,
                    task_id=task_id,
                    operation=operation,
                )
                if not torch.isfinite(loss):
                    if log_results:
                        logger.warning("eval loss is not finite on sample %s: %s", idx, loss.item())
                    continue

                loss_sum += float(loss.item()) * labels_for_loss
                token_count += labels_for_loss
                sample_count += 1

        loss_value = loss_sum / token_count if token_count > 0 else 0.0
        tag = f"{log_prefix} " if log_prefix else ""
        if log_results:
            logger.info(
                "%sEval loss: loss=%.4f samples=%d label_tokens=%d",
                tag,
                loss_value,
                sample_count,
                token_count,
            )
        return {
            "loss": loss_value,
            "loss_num_samples": float(sample_count),
            "loss_num_tokens": float(token_count),
        }
    finally:
        if was_training:
            model.train()
        model.reset()


@torch.no_grad()
def evaluate_generation_groups(
    model,
    tokenizer,
    eval_sample_groups: dict[str, list[dict]],
    device: torch.device,
    max_new_tokens: int = 256,
    max_samples: int = 20,
    num_examples: int = 3,
    metrics: Metrics | None = None,
    log_results: bool = True,
) -> dict[str, float]:
    """Run generation eval for each sample group and return group + overall metrics."""
    if metrics is None:
        metrics = Metrics()

    results: dict[str, float] = {}
    weighted_sums: dict[str, float] = defaultdict(float)
    total_count = 0.0

    grouped_eval = set(eval_sample_groups) != {"all"}
    for group_name in sorted(eval_sample_groups):
        samples = eval_sample_groups[group_name]
        if not samples:
            continue
        sample_cap = len(samples) if grouped_eval else min(max_samples, len(samples))

        group_results = evaluate_generation(
            model,
            tokenizer,
            samples,
            device,
            max_new_tokens=max_new_tokens,
            max_samples=sample_cap,
            num_examples=num_examples,
            metrics=metrics,
            log_prefix=group_name,
            log_results=log_results,
        )
        group_count = float(group_results.get("num_samples", 0.0))
        results[f"{group_name}/num_samples"] = group_count

        for key, value in group_results.items():
            if key == "num_samples":
                continue
            results[f"{group_name}/{key}"] = value
            if group_count > 0:
                weighted_sums[key] += value * group_count

        total_count += group_count

    for key, value in weighted_sums.items():
        results[key] = value / total_count if total_count > 0 else 0.0
    results["num_samples"] = total_count
    return results


@torch.no_grad()
def evaluate_generation(
    model,
    tokenizer,
    eval_samples: list[dict],
    device: torch.device,
    max_new_tokens: int = 256,
    max_samples: int = 20,
    num_examples: int = 3,
    metrics: Metrics | None = None,
    log_prefix: str = "",
    log_results: bool = True,
) -> dict[str, float]:
    """Run generation-based evaluation on a list of pre-encoded samples.

    Each sample is a dict with ``chunks``, ``num_chunks``, ``eval_chunk_idx``.
    Memory is built from all chunks before the eval chunk, then the model
    generates from the eval chunk's prompt.

    Args:
        eval_samples: list of samples in the same format as dataset items.
        max_new_tokens: max tokens to generate per sample.
        max_samples: cap the number of evaluated samples.
        num_examples: number of example outputs to log.
        metrics: Metrics instance for computing scores.
    """
    if metrics is None:
        metrics = Metrics()

    was_training = model.training
    model.eval()
    try:
        indices = list(range(len(eval_samples)))
        if len(indices) > max_samples:
            # Deterministic prefix, NOT random.sample: the upstream group
            # builder already stratified + shuffled with a fixed seed, and
            # sampling from the global RNG here would pick a different eval
            # subset every call (the RNG state advances during training),
            # making generation metrics non-comparable across steps/runs.
            indices = indices[:max_samples]

        accumulated: dict[str, list[float]] = defaultdict(list)
        examples: list[tuple[str, str, dict]] = []
        evaluated_count = 0
        pad_id = tokenizer.pad_token_id

        for idx in indices:
            sample = eval_samples[idx]
            eval_idx = sample["eval_chunk_idx"]
            chunks = sample["chunks"]

            model.reset()

            # Encode: commit all chunks up to (but not including) the eval chunk.
            # These forwards only write memory; none of their logits are read.
            for t in range(eval_idx):
                ids = chunks[t]["input_ids"].unsqueeze(0).to(device)
                model(
                    input_ids=ids,
                    commit_memory=True,
                    use_cache=False,
                    logits_to_keep=1,
                )

            # Build prompt: eval chunk up to the last assistant message.
            eval_ids = chunks[eval_idx]["input_ids"]
            eval_labels = chunks[eval_idx]["labels"]
            labelled_positions = (eval_labels != IGNORE_INDEX).nonzero(as_tuple=True)[0]
            if len(labelled_positions) == 0:
                continue
            prompt_end = labelled_positions[0].item()
            prompt_ids = eval_ids[:prompt_end].unsqueeze(0).to(device)
            ref_ids = eval_ids[labelled_positions[0]:labelled_positions[-1] + 1]
            ref_text = tokenizer.decode(ref_ids, skip_special_tokens=True)

            try:
                generated = model.generate(
                    input_ids=prompt_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=pad_id,
                )
            except Exception as e:
                if log_results:
                    logger.warning(f"generate() failed on sample {idx}: {e}")
                continue

            gen_ids = generated[0][prompt_ids.size(1):].tolist()
            if gen_ids and gen_ids[-1] == tokenizer.eos_token_id:
                gen_ids = gen_ids[:-1]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

            scores = metrics.calculate(gen_text, ref_text)
            for k, v in scores.items():
                accumulated[k].append(v)
            evaluated_count += 1

            if len(examples) < num_examples:
                examples.append((ref_text, gen_text, scores))

        if log_results:
            for i, (ref, gen, scores) in enumerate(examples):
                tag = f"{log_prefix} Example {i}" if log_prefix else f"Example {i}"
                logger.info(
                    f"\n[{tag}] " +
                    " ".join(f"{k}={v:.4f}" for k, v in scores.items())
                )
                logger.info(f"  REF: {ref[:300]}")
                logger.info(f"  GEN: {gen[:300]}")

        results = {
            k: sum(v) / len(v) if v else 0.0
            for k, v in accumulated.items()
        }
        results["num_samples"] = float(evaluated_count)
        tag = f"{log_prefix} " if log_prefix else ""
        if log_results:
            logger.info(
                f"{tag}Eval (n={evaluated_count}): " +
                " | ".join(f"{k}={v:.4f}" for k, v in results.items())
            )
        return results
    finally:
        if was_training:
            model.train()
        model.reset()
