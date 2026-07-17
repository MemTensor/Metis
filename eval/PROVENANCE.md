# Provenance

This evaluation package was assembled on 2026-07-16 against:

- public Metis base commit: `fe459c169131dcbc5dbfa32e3216d53fa66d700d`
- ATM-Bench scorer subset: upstream revision
  `d463445614ad78a48736b98ab901795f7ecaf3da`
- delta-Mem runtime used by the baseline: upstream revision
  `5cd5d9153c7f408764728d953565201e198c39e2`

Only the benchmark, method, loader, score, audit, and protocol modules needed
by paper Tables 5-8, 10, and 11 were carried into the public base. Historical
builders, update-scaling studies, caches, data, checkpoints, logs, and source
result directories were not carried over.

The canonical dataset row counts, byte sizes, SHA-256 digests, and boundary IDs
are recorded in `eval/data/manifest.json`. Checkpoint hashes used by the paper
are recorded in the protocol configs and `reported_scores.json`. Generated run
configs record the resolved data, asset registry, commands, and cell count.
