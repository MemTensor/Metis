# LowRankMemory

`matrix.py` runs Metis-4B step-14000 over ranks
`1, 4, 16, 64, 128, 256, full` and the four Table 10 datasets: 28 cells. The
numeric-rank policy projects LocalMemory state after every commit; `full`
disables projection.

`run.py` is the per-model inference/score engine. `audit.py` checks coverage,
IDs, row counts, query/runtime issues, delta-load metadata, low-rank config and
debug data, numeric strict scores, `api_median` provenance, judge metadata, and
MetisTest operation counts. The paper's `full` column uses the main-table
aggregate; an independent LowRankMemory run also executes `full` so its raw
predictions and judge variation can be audited directly.

```bash
python -m eval.experiments.low_rank.matrix --output-dir eval/outputs/lowrank --dry-run
python -m eval.experiments.low_rank.matrix --output-dir eval/outputs/lowrank-smoke --rank 1 --benchmark locomo_tps16 --limit 1
python -m eval.experiments.low_rank.matrix --output-dir eval/outputs/lowrank
```

Use `--raw-workers` and `--gpu-ids` to control raw-inference parallelism.
Scoring remains serialized per result file so resumable judge output is not
written concurrently.
