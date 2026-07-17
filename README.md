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

> [!IMPORTANT]
> **Research preview.** This repository provides the Metis architecture, data format documentation, inference examples, a multi-step mid-training and evaluation harness, and official model weights. Training data is not included yet.

## Overview

**Metis is the first prototype of a memory foundation model, equipping foundation models with a persistent and dynamically evolving native memory state.**

It learns to autonomously store and utilize information through model computation, compressing historical context into native memory and accessing it through memory attention. At inference time, all model weights remain frozen, while memory is updated through gradient-free forward computation.

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


## 🚀 Quick Start

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

## 🏋️ Training

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

The launcher freezes the backbone, disables LoRA, and trains the native-memory parameters. Complete argument descriptions, default values, validation controls, and the task schedule are documented in [`scripts/train.sh`](scripts/train.sh); run `bash scripts/train.sh --help` for the command-line interface.

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

Common optimization and validation controls are environment variables. For example, the following command reproduces the main settings of a 4-GPU Qwen3.5-4B run with per-device batch size 4, gradient accumulation 2, three epochs, and last-token gated-delta memory updates (effective batch size 32):

```bash
LR=2e-4 NUM_EPOCHS=3 \
bash scripts/train.sh \
  --model-path /path/to/Qwen3.5-4B \
  --name metis-qwen35-4b-agg-lta-gdu-4gpu-bs4-ga5-3ep \
  --train-data /path/to/tokenized/train \
  --valid-data /path/to/tokenized/valid \
  --backbone-type qwen3_5 \
  --cuda-visible-devices 0,1,2,3 \
  --nproc-per-node 4 \
  --batch-size 4 \
  --grad-accum 5 \
  --metis-hyper-memory-type LastTokenGatedDeltaRuleMetisHyperMemory
```

## 📊 Evaluation


## Roadmap

![Roadmap for memory foundation models](assets/memory_foundation_model_roadmap.png)

The paper frames native memory as a progression from **stateful capability** to **self-managing memory**, **experience-driven learning**, **persistent cognition**, and ultimately **self-evolving capability**.

Metis is an early research system rather than a complete replacement for external memory. Hybrid native–external memory remains an important direction.


## 📝 Citation

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

## ⚖️ License

This project uses separate licenses for the paper and the repository software:

- **Paper:** The Metis technical report and original paper materials are licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](LICENSE-PAPER).
- **Repository software:** The source code in this repository is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE).

Commercial use of the repository software is not permitted under the PolyForm Noncommercial License. For commercial licensing inquiries, please contact the authors.

Unless explicitly stated otherwise, model weights, datasets, benchmark assets, trademarks, and third-party materials are not covered by the licenses above. Their applicable terms will be provided with the corresponding releases.

## 📬 Contact

Research correspondence: [lizy@memtensor.cn](mailto:lizy@memtensor.cn) and [xu.chen@ruc.edu.cn](mailto:xu.chen@ruc.edu.cn).
