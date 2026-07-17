# Metis paper evaluation

This directory reproduces the four evaluation workflows reported in the Metis
paper: the MemOP/MemQA main results, Metis-4B ablations, ATM/MemDaily OOD, and
Metis-4B LowRankMemory. It deliberately excludes training code, historical
candidate matrices, cached models, datasets, checkpoints, logs, and generated
per-example results.

## 1. Environment and assets

```bash
conda env create -f eval/environments/paper-eval-minimal-cu118.yml
conda activate metis-paper-eval-cu118
python -m eval.data.prepare --dataset locomo
```

Evaluation data is not distributed. See `eval/data/README.md` for the pinned
upstream sources, raw layout, deterministic converters, and the two
owner-provided MemOP sources that are unavailable from public downloads.

Download checkpoints and adapters to the relative locations in
`eval/configs/assets.json`. For another layout, copy that registry to the
ignored `eval/artifacts/assets.local.json`, edit the copy, and pass
`--assets eval/artifacts/assets.local.json` to a matrix. Never put credentials
in a config file; judge credentials are read only from the environment
variable selected by the CLI.

delta-Mem additionally needs its official runtime at the pinned revision in
`eval/methods/delta_mem/README.md`. The ATM official metric subset is already
vendored with its MIT license under `eval/third_party/atm_bench/`.

## 2. Protocol checks

These commands expand the three declarative matrices without loading a model:

```bash
python -m eval.experiments.main_tables.matrix --output-dir eval/outputs/main --dry-run
python -m eval.experiments.ood.matrix --output-dir eval/outputs/ood --dry-run
python -m eval.experiments.low_rank.matrix --output-dir eval/outputs/lowrank --dry-run
```

The complete main, OOD, and LowRankMemory grids contain 77, 14, and 28 cells.
The ablation matrix contains the full Metis-4B checkpoint and the six paper
ablations over the four Table 7 benchmarks. Paper-reported aggregate scores and
checkpoint hashes are recorded in `configs/paper/reported_scores.json`.

The ablation preflight is different: it reads and hashes every selected dataset
and checkpoint, so run it only after preparing those assets:

```bash
python -m eval.experiments.ablation.run --stage audit
```

Remove `--dry-run` to execute main, OOD, or LowRankMemory. The main and OOD
launchers accept repeated `--method`/`--benchmark` or `--dataset` filters for a
small smoke. LowRankMemory accepts `--rank` and `--benchmark` filters. Generated
outputs belong under ignored `eval/outputs/`. Memory-based runners resume
completed raw rows where supported, but the main matrix does not skip completed
cells; use method/benchmark filters as explicit shards and do not relaunch a
finished shard. See each experiment README for complete commands.

## Result boundary

The score reference contains table aggregates, not generated answers or cached
judge responses. See `VALIDATION.md` for isolated-checkout test evidence,
`RELEASE_CHECKLIST.md` before release, and `INTEGRATION.md` before copying this
directory into another compatible Metis checkout.
