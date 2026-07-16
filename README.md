<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/metis_hero_dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/metis_hero_light.svg">
    <img src="assets/metis_hero_light.svg" alt="Metis — Memory Foundation Model" width="100%">
  </picture>

  <p>
    <!-- TODO: replace PAPER_URL below with the public technical report link (e.g. arXiv) -->
    <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="assets/badges/paper.svg" alt="Technical report"></a>
    <img src="assets/badges/status.svg" alt="Research preview">
    <img src="assets/badges/backbone.svg" alt="Qwen3.5 backbone">
    <a href="#model-checkpoints"><img src="assets/badges/models.svg" alt="Models coming soon"></a>
  </p>

  <p>
    <a href="https://arxiv.org/abs/XXXX.XXXXX">Paper</a> ·
    <a href="#quick-start">Quick Start</a> ·
    <a href="#model-checkpoints">Models</a> ·
    <a href="#citation">Citation</a> ·
    <a href="README_zh.md">中文</a>
  </p>
</div>

> [!IMPORTANT]
> **Research preview.** This repository currently provides the Metis architecture, Qwen3/Qwen3.5/Llama integration, multi-step mid-training code, inference tooling, DeepSpeed configuration, and checkpoint utilities. Public model checkpoints, training data, and a standalone evaluation harness are not included yet.

## Overview

**Metis** is the first prototype of a **memory foundation model**: a foundation model whose memory state and memory procedures are native parts of the model rather than external retrieval workflows.

Metis keeps a persistent, layer-wise memory state across interaction steps. It learns how to **remember, update, forget, reflect, and selectively use** stored information during the model's forward computation.

![From external memory to native memory](assets/native_memory_vs_external.png)

The project explores three ideas:

- **Native memory state.** Dynamic parametric states live inside the backbone and participate directly in later forward passes.
- **Native memory procedures.** Storage and utilization are learned from data instead of being implemented as separate retrieval, reranking, and prompt-construction rules.
- **Fixed-size session state.** Historical information is compressed into a compact state, so later queries do not need to replay the original text history.

Metis is an early research system, not a complete replacement for external memory. Hybrid native–external memory remains an important direction.

## Architecture

![Metis architecture](assets/metis_architecture.png)

A **Metis Block** is inserted into Transformer layers and contains two components:

- **Local Memory Block:** maintains the dynamic memory matrix and normalization state that persist across interaction steps.
- **Hyper Memory Block:** learns token selection, memory key/value projections, a dedicated memory query, and the state-update procedure.

After each memory step, Metis selects informative hidden states and updates the local memory. During a later query, memory attention reads that state and fuses the result with the original attention branch. The paper implementation uses a **Gated Delta Network (GDN)** update.

In the current Qwen3.5 hybrid implementation, Metis is attached to full-attention layers; linear-attention layers keep their original computation path.

## Quick Start

### 1. Create the environment

```bash
conda env create -f environment.yml
conda activate metis
```

The checked-in environment is a research lock file based on Python 3.10, PyTorch 2.4.1 + CUDA 11.8, Transformers 5.4.0, DeepSpeed, and Flash Linear Attention. CUDA extensions may require platform-specific adjustment.

### 2. Verify the source tree

```bash
python -m compileall -q metis train
python train/run_train.py --help
```

### 3. Run inference

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

`--commit_mode` supports:

| Mode | Behavior |
| --- | --- |
| `none` | Do not update memory |
| `user` | Commit the user message |
| `exchange` | Commit the user message and model response |

For a moved delta checkpoint, override the backbone with `--model_path /path/to/backbone` or the `MODEL_PATH` environment variable. Run `python run_inference.py --help` for all options.

### Runtime memory lifecycle

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

Metis accepts JSONL data in a flat layout:

```text
data/train/remember_explicit.jsonl
```

or a nested layout:

```text
data/train/remember/explicit_data.jsonl
```

Each line must contain one JSON object. `messages` is a list of memory chunks, and each chunk is a list of chat messages. `query_turn_id` selects the chunk whose final assistant response contributes to the loss.

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

Supported task IDs:

| Task | Data |
| --- | --- |
| 0 | Reconstruction and explicit/implicit recall |
| 1 | Remember, forget, update, and reflection operations |
| 2 | Distractor and long-context examples |
| 3 | Samples marked as `task3` |
| 4 | Samples marked as `task4` |

## Preprocessing

Pre-tokenize the training and validation sets separately:

```bash
python scripts/tokenize_dataset.py \
  --data_dir data/train \
  --output_dir data/tokenized/train \
  --model_path /path/to/backbone \
  --tasks 0,1,2,3,4 \
  --max_total_tokens 1024

python scripts/tokenize_dataset.py \
  --data_dir data/valid \
  --output_dir data/tokenized/valid \
  --model_path /path/to/backbone \
  --tasks 0,1,2,3,4 \
  --max_total_tokens 1024
```

Use `--overwrite` to replace an existing tokenized cache.

## Training

```bash
bash scripts/train.sh \
  --model-path /path/to/backbone \
  --name metis-run \
  --train-data data/tokenized/train \
  --valid-data data/tokenized/valid \
  --output-dir checkpoints
```

Tokenized data is used by default. Add `--data-format raw` to train directly from JSONL files.

Common options:

```text
--backbone-type qwen3_5|qwen3|llama
--cuda-visible-devices 0,1,2,3
--nproc-per-node 4
--batch-size 2
--grad-accum 10
--metis-hyper-memory-type LastTokenGatedDeltaRuleMetisHyperMemory
--deepspeed configs/ds_zero3.json
```

Run `bash scripts/train.sh --help` for the complete launcher interface.

Training runs in the background. Logs are written to `logs/`, and model outputs are written to `<output-dir>/<name>/`.

### Validation

When `EVAL_STEPS` is greater than zero, training performs loss and generation evaluation on a fixed validation subset.

```text
<output-dir>/<name>/eval_samples_manifest.json
<output-dir>/<name>/eval_metrics.jsonl
```

Configure validation with `EVAL_STEPS`, `GEN_EVAL_STEPS`, `EVAL_SAMPLES`, and `EVAL_SAMPLES_PER_TASK`.

## Data and Optimization

Metis is mid-trained on temporally ordered, multi-step interactions. Earlier steps transform the native memory state; later query steps provide supervised responses.

| Memory behavior | Training pattern |
|---|---|
| Remember | Store a fact and answer a later query |
| Update | Replace an earlier value with a newer one |
| Forget | Revoke information before a later query |
| Reflection | Compose multiple stored facts |
| Robustness | Handle distractors, multiple entities, selective forgetting, and memory-irrelevant dialogue |

The training objective combines:

1. **Memory reconstruction** to warm up high-fidelity storage and recovery.
2. **Memory operations** to learn instruction-driven remember, update, forget, and reflection behavior.
3. **Regularization** to reduce interference, collateral forgetting, and memory pollution.

The backbone is frozen during mid-training; only native-memory parameters are optimized. See Sections 4–5 of the [technical report](https://arxiv.org/abs/XXXX.XXXXX) for the complete data construction pipeline, sampling curriculum, and objectives.

## Paper Highlights

The report evaluates Metis without replaying the original evidence at query time.

- Metis-27B reaches **73.77** on the Metis Test set and **50.82** on NextMem under the no-context setting.
- In a controlled single-A800, batch-size-1 context-length sweep with the 4B model, Metis query latency remains nearly independent of stored history length and achieves an **11.37× query-latency speedup** over Full Context at 128K history for 32 generated tokens.
- A rank-64 Metis-4B state occupies **2.11 MB per session** and retains **99.9%** of the full-state average score in the report's compression study.

These results are configuration-specific. The full benchmark tables, prompts, baselines, and evaluation protocol are provided in the [technical report](https://arxiv.org/abs/XXXX.XXXXX).

## Model Checkpoints

Public model checkpoints are not released yet. Links will be added here when they become available.

## Roadmap

![Roadmap for memory foundation models](assets/memory_foundation_model_roadmap.png)

The paper frames native memory as a progression from **stateful capability** to **self-managing memory**, **experience-driven learning**, **persistent cognition**, and ultimately **self-evolving capability**.

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

## Acknowledgements

Metis builds on PyTorch, Hugging Face Transformers, DeepSpeed, Flash Linear Attention, and Qwen3.5, together with prior research on fast weight programming, memory-augmented neural networks, test-time training, long-context modeling, and agent memory.

## Contact

Research correspondence: [lizy@memtensor.cn](mailto:lizy@memtensor.cn) and [xu.chen@ruc.edu.cn](mailto:xu.chen@ruc.edu.cn).
