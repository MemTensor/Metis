<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/metis_hero_dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/metis_hero_light.svg">
    <img src="assets/metis_hero_light.svg" alt="Metis — Memory Foundation Model" width="100%">
  </picture>

  <p>
    <!-- TODO: replace PAPER_URL below with the public technical report link (e.g. arXiv) -->
    <a href="https://statics.memtensor.com.cn/paper/metis_v1.pdf"><img src="assets/badges/paper.svg" alt="Preview"></a>
    <a href="https://huggingface.co/collections/IAAR-Shanghai/metis"><img src="assets/badges/models.svg" alt="HuggingFace"></a>
  </p>

  <p>
    <a href="https://arxiv.org/abs/XXXX.XXXXX">Paper</a> ·
    <a href="#overview">Overview</a> ·
    <a href="#quick-start">Quick Start</a> ·
    <a href="#training">Training</a> ·
    <a href="#evaluation">Evaluation</a> ·
    <a href="#citation">Citation</a>
  </p>
</div>

## Research Preview

To address the limitations of external memory, we introduce **memory foundation models** that empower large foundation models with **native memory**. This converts memory from an external module into an internal mechanism of the backbone, directly involved in forward computation. Memory foundation models are natively stateful across multiple inferences: they formulate, maintain, and utilize memory states inside the backbone from prior interactions. Based on this formulation, we propose **Metis**, the first prototype of a memory foundation model.

## Overview

Metis equips a foundation model with a persistent, layer-wise memory state and learns how to **remember, update, forget, reflect, and selectively use** stored information during forward computation.

![From external memory to native memory](assets/native_memory_vs_external.png)

The project explores three central ideas:

- **Native memory state.** Dynamic parametric states live inside the backbone and participate directly in later forward passes.
- **Native memory procedures.** Storage and utilization are learned from data instead of being implemented as separate retrieval, reranking, and prompt-construction rules.
- **Fixed-size session state.** Historical information is compressed into a compact state, so later queries do not need to replay the original text history.

### Architecture

A **Metis Block** is inserted into Transformer layers and contains two components:

- The **Local Memory Block** maintains the dynamic memory matrix and normalization state that persist across interaction steps.
- The **Hyper Memory Block** learns token selection, memory key/value projections, a dedicated memory query, and the state-update procedure.

After each memory step, Metis selects informative hidden states and updates the local memory. During a later query, memory attention reads that state and fuses the result with the original attention branch. The default implementation uses a **Gated Delta Network (GDN)** update. In the Qwen3.5 hybrid implementation, Metis is attached to full-attention layers, while linear-attention layers keep their original computation path.

Metis is an early research system rather than a complete replacement for external memory. Hybrid native–external memory remains an important direction.

## Quick Start

### 1. Create the environment

```bash
conda env create -f environment.yml
conda activate metis
```

### 2. Run inference

Run inference from a Metis delta or full checkpoint:

```bash
bash infer.sh /path/to/checkpoint \
  --prompt "Hello" \
  --max_new_tokens 128
```

For multi-turn inference with persistent memory, repeat `--prompt`:

```bash
python run_inference.py \
  --checkpoint_path /path/to/checkpoint \
  --prompt "Remember that my code name is Polaris." \
  --prompt "What is my code name?" \
  --commit_mode exchange
```

`--commit_mode none` leaves memory unchanged, `--commit_mode user` commits only the user message, and `--commit_mode exchange` commits both the user message and the model response.

### Runtime Memory Lifecycle

```python
model.reset()

# Write one interaction step into the native memory state.
model(
    **memory_inputs,
    commit_memory=True,
    use_cache=False,
    logits_to_keep=1,
)

# Query without replaying the original memory text.
answer_ids = model.generate(
    **query_inputs,
    max_new_tokens=32,
    do_sample=False,
)

# Start a new independent session.
model.reset()
```

## Data Format

Each line of a Metis JSONL dataset contains one object. `messages` is a list of interaction chunks, each chunk is a list of chat messages, and `query_turn_id` selects the chunk whose final assistant response contributes to the loss.

```json
{
  "sample_id": "sample-001",
  "messages": [
    [
      {"role": "user", "content": "Remember that my code name is Polaris."},
      {"role": "assistant", "content": "I will remember it."}
    ],
    [
      {"role": "user", "content": "What is my code name?"},
      {"role": "assistant", "content": "Your code name is Polaris."}
    ]
  ],
  "query_turn_id": 1,
  "metadata": {"type": "remember", "style": "explicit"}
}
```

## Training

### 1. Prepare the dataset

The Metis training dataset can be downloaded from **[URL to be released]**. Prepare separate training and validation splits in the JSONL format described above.

### 2. Tokenize the dataset

Pre-tokenize the training and validation splits with the tokenizer of the target backbone:

```bash
python scripts/tokenize_dataset.py \
  --data_dir /path/to/metis-dataset/train \
  --output_dir /path/to/tokenized/train \
  --model_path /path/to/backbone \
  --max_total_tokens 1024

python scripts/tokenize_dataset.py \
  --data_dir /path/to/metis-dataset/valid \
  --output_dir /path/to/tokenized/valid \
  --model_path /path/to/backbone \
  --max_total_tokens 1024
```

All training tasks are included by default. Add `--overwrite` when intentionally replacing an existing tokenized cache.

### 3. Launch training

A minimal tokenized-data run is:

```bash
bash scripts/train.sh \
  --model-path /path/to/backbone \
  --name metis-run \
  --train-data /path/to/tokenized/train \
  --valid-data /path/to/tokenized/valid
```

The launcher freezes the backbone, disables LoRA, and trains the native-memory parameters. Its main arguments are:

| Argument | Description |
| --- | --- |
| `--model-path` | Local backbone path or Hugging Face model ID. |
| `--name` | Identifier for the training run. |
| `--train-data` | Training cache, or a raw JSONL directory when `--data-format raw` is used. |
| `--valid-data` | Validation cache or raw validation data. |
| `--data-format` | `tokenized` by default; use `raw` to tokenize samples online. |
| `--backbone-type` | `qwen3_5`, `qwen3`, or `llama`. |
| `--cuda-visible-devices` | Comma-separated CUDA devices. |
| `--nproc-per-node` | Number of data-parallel `torchrun` workers. |
| `--batch-size` | Per-device batch size. |
| `--grad-accum` | Gradient accumulation steps. The effective batch size is workers × per-device batch × accumulation. |
| `--deepspeed` | Enable DeepSpeed with the specified configuration; `--no-deepspeed` disables it. |
| `--resume-from-checkpoint` | Resume model, optimizer, scheduler, RNG, and training step; `auto` selects the newest checkpoint. |
| `--init-from-checkpoint` | Load checkpoint weights but start a fresh optimizer and training state. |

The default memory recipe is:

```text
NormedReweightLearnedQueryMetisBlock
+ StraightThroughAlphaTopPGatedDeltaRuleMetisHyperMemory
+ NormalizedDeltaNetMetisLocalMemory
```

Append one of the following options to the launcher command to switch the hyper-memory variant:

```text
# Last-token gated-delta update.
--metis-hyper-memory-type LastTokenGatedDeltaRuleMetisHyperMemory

# AlphaTopP token selection without the gated-delta update.
--metis-hyper-memory-type StraightThroughAlphaTopPKeyNormMetisHyperMemory
```

`--metis-block-type`, `--metis-hyper-memory-type`, and `--metis-local-memory-type` can be combined to study block fusion, token aggregation/update, and local-state variants respectively.

Common optimization and validation controls are environment variables:

| Variable | Meaning |
| --- | --- |
| `LR` | Learning rate; default `2e-4`. |
| `NUM_EPOCHS` / `MAX_STEPS` | Epoch-based or step-limited training. |
| `WARMUP_STEPS` | Constant-with-warmup scheduler warmup. |
| `SAVE_STEPS` | Checkpoint interval. |
| `EVAL_STEPS` | Validation interval; set to `0` to disable validation. |
| `GEN_EVAL_STEPS` | Generation-evaluation interval; `0` runs generation at every validation point. |
| `EVAL_SAMPLES` / `EVAL_SAMPLES_PER_TASK` | Size and per-task balance of the fixed validation subset. |
| `TASKS` | Comma-separated training-task subset; all five tasks are enabled by default. |

For example, prefix the launcher with `LR=1e-4 NUM_EPOCHS=3 EVAL_STEPS=1000` to override those defaults. For settings exposed by both interfaces, an explicit launcher option takes precedence over its environment variable.

### Training Tasks

Metis organizes the objectives into five sampling tasks:

| Task | Training behavior |
| --- | --- |
| 0 | Reconstruction and explicit/implicit fact recall. |
| 1 | Explicit/implicit remember, forget, update, and reflection operations. |
| 2 | Distractor and long-context variants of the memory operations. |
| 3 | Mixed and LLM-snippet memory interactions. |
| 4 | Normal and no-query interactions that regularize memory pollution. |

The sampler anneals each task from `TASK0_WEIGHT_START` … `TASK4_WEIGHT_START` to the corresponding `*_END` value across training. Memory reconstruction establishes high-fidelity storage, operation supervision teaches instruction-driven state changes, and the later tasks reduce interference, collateral forgetting, and memory leakage.

## Evaluation

Training-time evaluation uses the validation split supplied to `train.sh`. `EVAL_STEPS` controls loss evaluation, while `GEN_EVAL_STEPS` controls the more expensive generation evaluation. A standalone evaluation package will be released separately.

The technical report evaluates Metis without replaying the original evidence at query time:

- Metis-27B reaches **73.77** on the Metis Test set and **50.82** on NextMem under the no-context setting.
- In a controlled single-A800, batch-size-1 sweep, Metis-4B achieves an **11.37× query-latency speedup** over Full Context at 128K history for 32 generated tokens.
- A rank-64 Metis-4B state occupies **2.11 MB per session** and retains **99.9%** of the full-state average score in the report's compression study.

These results are configuration-specific. The full benchmark tables, prompts, baselines, and evaluation protocol are provided in the [technical report](https://arxiv.org/abs/XXXX.XXXXX).

## Roadmap

![Roadmap for memory foundation models](assets/memory_foundation_model_roadmap.png)

The paper frames native memory as a progression from **stateful capability** to **self-managing memory**, **experience-driven learning**, **persistent cognition**, and ultimately **self-evolving capability**. Upcoming repository releases will add public checkpoints, training data, and standalone evaluation tooling.

## Citation

A machine-readable `CITATION.cff` is included at the repository root.

```bibtex
@misc{zhang2026metis,
  title        = {Metis: Memory Foundation Model},
  author       = {Zhang, Zeyu and Guo, Ziliang and Sun, Yihang and Zhang, Xichong and
                  Hao, Xixuan and Lin, Zehao and Zhang, Yang and Zhao, Xiaoyan and
                  Shen, Tong and Tang, Bo and Xu, Zhi-Qin John and Yan, Junchi and
                  Wang, Haofen and Chen, Xu and Xiong, Feiyu and Li, Zhiyu and
                  Chua, Tat-Seng},
  year         = {2026},
  howpublished = {Technical report}
}
```

## License

This project uses separate licenses for the paper and the repository software:

- **Paper:** The Metis technical report and original paper materials are licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](LICENSE-PAPER).
- **Repository software:** The source code in this repository is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE).

Commercial use of the repository software is not permitted under the PolyForm Noncommercial License. For commercial licensing inquiries, please contact the authors.

Unless explicitly stated otherwise, model weights, datasets, benchmark assets, trademarks, and third-party materials are not covered by the licenses above. Their applicable terms will be provided with the corresponding releases.

## Contact

Research correspondence: [lizy@memtensor.cn](mailto:lizy@memtensor.cn) and [xu.chen@ruc.edu.cn](mailto:xu.chen@ruc.edu.cn).
