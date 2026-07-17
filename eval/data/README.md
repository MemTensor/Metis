# Evaluation data

No raw or normalized evaluation data is distributed with this repository.
Users obtain the source datasets from their owners, place them under the
git-ignored `eval/data/raw/` layout below, and run the tracked converters.

```text
eval/data/raw/
  locomo/locomo10.json
  nextmem/stm_{hotpot,squad,locomo,longmemeval}_test.json
  metis_test/                                      # owner-provided source directory
  metisops/memoryops_v23_30topic_artifacts.zip     # owner-provided artifact
  atm/
    data/atm-bench/atm-bench.json
    data/processed_memory/image_batch_results.json
    data/processed_memory/video_batch_results.json
    data/raw_memory/email/emails.json
  memdaily/memdaily.json
```

Public sources:

- LoCoMo: `https://github.com/snap-research/LoCoMo`; use
  `data/locomo10.json`.
- NextMem: `https://github.com/nuster1128/NextMem` at revision `df63035`;
  obtain the Task 2 `stm_*_test.json` files referenced by its README.
- ATM-Bench: `https://huggingface.co/datasets/Jingbiao/ATM-Bench` at revision
  `78e826dc07e97466b2f54443831ef9a83ab8b27c`; only the four files shown above
  are required.
- MemDaily: `https://github.com/nuster1128/MemSim` at revision
  `db9d5d552d6cb1d859f692eb7e6c0fd6d61d3815`; copy
  `data_generation/final_dataset/memdaily.json`.

MetisTest and MetisOps are not downloaded by this repository. Under the
current no-data-release policy, users need source files obtained separately
from the data owner. This limitation means the complete MemOP table cannot be
reproduced from public downloads alone.

Build one family or all available families:

```bash
python -m eval.data.prepare --dataset locomo
python -m eval.data.prepare --dataset nextmem
python -m eval.data.prepare --dataset atm --dataset memdaily
python -m eval.data.prepare --dataset metis_test --dataset metisops
python -m eval.data.prepare --dataset all
```

`prepare` verifies the selected families automatically. After all six families
have been built, `python -m eval.data.verify` verifies the complete seven-file
payload.

The converters write the exact paths consumed by the launchers:

```text
eval/data/
  memqa/*.jsonl
  memops/*.jsonl
  ood/{atm,memdaily}.jsonl
```

`verify` checks row count, order boundaries, and an evaluation-content hash.
It also reports whether bytes exactly match the frozen paper payload. The
content hash excludes only machine-local build metadata (`source_dir` and an
optional diagnostic tokenizer count); model inputs, answers, scoring fields,
row order, and all other metadata remain covered. A runner will not silently
select a similarly named historical split.
