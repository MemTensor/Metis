# ATM and MemDaily OOD

`matrix.py` expands seven methods over the two Table 8 datasets into 14 cells.
`run.py` performs inference and scoring for one method; `audit.py` verifies
row/ID coverage, runtime metadata, official ATM scorer provenance, judge
metadata where ATM open-ended rows require it, and the final aggregate.
The vendored ATM subset is pinned by file hash in
`eval/third_party/atm_bench/MANIFEST.json`.

```bash
python -m eval.experiments.ood.matrix --output-dir eval/outputs/ood --dry-run
python -m eval.experiments.ood.matrix --output-dir eval/outputs/ood-smoke --method metis4b --dataset atm --limit 1
python -m eval.experiments.ood.matrix --output-dir eval/outputs/ood
```

MemDaily uses its deterministic choice metric and does not contact the judge.
ATM open-ended questions use the configured OpenAI-compatible judge with three
strict repeats. Set the API key only in the environment; do not put it in an
asset registry or run config.
