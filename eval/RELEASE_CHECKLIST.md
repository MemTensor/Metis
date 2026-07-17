# Evaluation release checklist

Before publishing the evaluation package:

1. Keep evaluation data out of Git. Rebuild each available source with
   `python -m eval.data.prepare`, verify all seven normalized files with
   `python -m eval.data.verify`, and keep the public limitation for the two
   owner-provided MemOP sources explicit.
2. Publish the three paper Metis checkpoints, six additional ablation
   checkpoints, and the delta-Mem adapter at the paths documented in
   `eval/configs/assets.json`. For each Metis evaluation package, retain the
   delta/config/manifest, tokenizer files, and provenance metadata; exclude
   optimizer, scheduler, and RNG states. Replace private base-model paths in
   checkpoint JSON with public model IDs, update the declared release config
   hashes, and verify every delta hash after packaging. Record immutable
   revisions for Qwen, BGE-M3, delta-Mem, and every released Metis repository;
   do not make the paper recipe depend on a mutable `main` snapshot.
3. Run the 77-cell main matrix and 14-cell OOD matrix to completion. Run the
   selected ablation and LowRankMemory checks, then retain their generated
   audits outside Git.
4. Compare aggregate scores with `eval/configs/paper/reported_scores.json` and
   investigate deviations together with raw-answer equality, row coverage,
   load reports, and judge metadata. Aggregate agreement alone is not enough.
5. Scan the complete destination repository and Git history for secrets,
   internal endpoints, personal paths, caches, logs, datasets, checkpoints, and
   result payloads. The scan must include files outside `eval/`. If generated
   evidence is published separately, sanitize server-local paths and physical
   GPU identifiers while preserving asset IDs, hashes, and score provenance.
6. Complete the separately owned license, attribution, dataset-card, and model-
   card reviews for all released assets and third-party dependencies.

Do not commit `.env`, source/normalized data, model weights, raw predictions,
scored rows, or judge logs.
