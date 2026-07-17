# MemOps and Metis Test

Inference code for MetisTest no-mixed, MetisOps full-segment, and MetisOps
gold-turn protocols. Normalized payload paths, counts, bytes, and hashes are in
`eval/data/manifest.json`.

Memory-only queries are audited so evidence, gold answers, and scoring metadata
do not leak into query payloads. MemOP OOM policy is `fail` in paper workflows;
placeholder rows are not accepted by final audits.
