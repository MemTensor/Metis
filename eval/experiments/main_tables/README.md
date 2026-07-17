# Main MemOP and MemQA tables

`matrix.py` expands the canonical 77 cells from
`eval/configs/paper/table_5_6_main.json`. `run.py` dispatches one cell to its
benchmark/method implementation. DenseRAG is intentionally absent from
NextMem, where retrieval is not applicable; the other four datasets contain
all 16 method rows.

The protocol includes Qwen no/full context at 4B/9B/27B, DenseRAG at those
scales, delta-Mem 4B, Temp-LoRA at 4B/9B/27B, and Metis at 4B/9B/27B using
steps 14000/8000/14000. Assets are resolved through
`eval/configs/assets.json`; no historical checkpoint is silently substituted.
Temp-LoRA 27B and Metis 27B are both two-GPU cells, but use their distinct
paper-audited loading policies from the matrix (`balanced` and
`paired_layers`, respectively).

Dry-run the complete matrix, run a one-row smoke, or run all cells:

```bash
python -m eval.experiments.main_tables.matrix --output-dir eval/outputs/main --dry-run
python -m eval.experiments.main_tables.matrix --output-dir eval/outputs/main-smoke --method metis4b_v24_s14000 --benchmark locomo_tps16 --limit 1
python -m eval.experiments.main_tables.matrix --output-dir eval/outputs/main
```

The matrix launches cells sequentially and does not skip a completed cell.
For an interruptible full run, use repeated `--method` and `--benchmark`
filters as explicit shards, keep one output directory per shard, and record
which shards completed before relaunching.
