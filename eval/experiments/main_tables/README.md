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
