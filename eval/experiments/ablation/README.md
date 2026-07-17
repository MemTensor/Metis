# Data and structure ablation

`run.py` is the Table 7 evaluation engine with repository paths updated for
this package. The exact protocol matrix is
`configs/ablation_matrix.json`. `--checkpoint` selects checkpoint IDs from that
matrix. It contains only the seven paper checkpoint packages; superseded and
confounded audit-only entries are excluded. The default paper path selects four
datasets; MemOps Full remains available only via `--include-appendix`.
Normalized data is read from `eval/data/`. Checkpoint IDs resolve through the
shared `eval/configs/assets.json`; pass `--assets eval/artifacts/assets.local.json`
for another local layout.

Stages are `audit`, `raw`, `score`, `result-audit`, and `summary`. The original
runner intentionally has no synthetic `all` stage.

Use `--stage audit` first. `--dry-run` applies to raw/score subprocess launches
and does not contact a Judge service.

The two full paper axes use separate result roots. A single GPU example is:

```bash
python -m eval.experiments.ablation.run --stage audit

python -m eval.experiments.ablation.run --stage raw --axis data --mode full --run-dir eval/outputs/ablation-data --gpu 0
python -m eval.experiments.ablation.run --stage score --axis data --mode full --run-dir eval/outputs/ablation-data --judge-concurrency 64
python -m eval.experiments.ablation.run --stage result-audit --axis data --mode full --run-dir eval/outputs/ablation-data
python -m eval.experiments.ablation.run --stage summary --axis data --mode full --run-dir eval/outputs/ablation-data

python -m eval.experiments.ablation.run --stage raw --axis structure --mode full --run-dir eval/outputs/ablation-structure --gpu 0
python -m eval.experiments.ablation.run --stage score --axis structure --mode full --run-dir eval/outputs/ablation-structure --judge-concurrency 64
python -m eval.experiments.ablation.run --stage result-audit --axis structure --mode full --run-dir eval/outputs/ablation-structure
python -m eval.experiments.ablation.run --stage summary --axis structure --mode full --run-dir eval/outputs/ablation-structure
```

Use repeated `--gpu-map CHECKPOINT_ID=GPU` assignments for parallel raw
inference. The raw and score stages skip cells that already pass their
completeness checks.
