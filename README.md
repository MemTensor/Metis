# Metis

## Overview

Metis adds trainable HyperMemory and LocalMemory modules to a frozen language-model backbone, enabling persistent memory across conversational turns.

## Installation

```bash
conda env create -f environment.yml
conda activate metis
```

The provided environment uses Python 3.10, PyTorch 2.4.1, and CUDA 11.8.

## Inference

Run inference from a Metis delta or full checkpoint:

```bash
bash infer.sh /path/to/checkpoint \
  --prompt "Hello" \
  --max_new_tokens 128
```

For multi-turn inference, repeat `--prompt`:

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

For a moved delta checkpoint, override the backbone with `--model_path /path/to/backbone` or the `MODEL_PATH` environment variable.

Run `python run_inference.py --help` for all options.

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

## Validation

When `EVAL_STEPS` is greater than zero, training performs loss and generation evaluation on a fixed validation subset.

```text
<output-dir>/<name>/eval_samples_manifest.json
<output-dir>/<name>/eval_metrics.jsonl
```

Configure validation with `EVAL_STEPS`, `GEN_EVAL_STEPS`, `EVAL_SAMPLES`, and `EVAL_SAMPLES_PER_TASK`.
