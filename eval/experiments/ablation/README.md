# Data and structure ablation

`run.py` is the Table 7 evaluation engine with repository paths updated for
this package. The exact protocol matrix is
`configs/ablation_matrix.json`. `--checkpoint` selects checkpoint IDs from that
matrix. It contains only the seven paper checkpoint packages; superseded and
confounded audit-only entries are excluded. The default paper path selects four
datasets; MemOps Full remains available only via `--include-appendix`.
Normalized data is read from `eval/data/`. Checkpoint IDs resolve through the
shared `eval/configs/assets.json`; pass `--assets artifacts/assets.local.json`
for another local layout.

Stages are `audit`, `raw`, `score`, `result-audit`, and `summary`. The original
runner intentionally has no synthetic `all` stage.

Use `--stage audit` first. `--dry-run` applies to raw/score subprocess launches
and does not contact a Judge service.
