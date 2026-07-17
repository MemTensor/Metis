# Validation record

This release candidate was validated from an isolated checkout of public Metis
commit `fe459c169131dcbc5dbfa32e3216d53fa66d700d`. No pre-existing result
directory was modified.

## Static and protocol validation

- All Python sources compiled in the existing paper environment.
- All 15 test functions passed when invoked directly (the environment has no
  `pytest` command).
- All seven data files passed exact byte-size, SHA-256, row-count, and boundary
  ID verification.
- The ablation audit verified five datasets and seven checkpoint packages,
  including config, delta, and trainer-state metadata where declared.
- Main, OOD, and LowRankMemory expand to the paper's 77, 14, and 28 declared
  cells. Generated child-CLI dry runs completed across the local and server
  copies, including the distinct Temp-LoRA main/OOD seeds and 27B two-GPU
  loading arguments.
- The vendored ATM deterministic scorer matched existing official number and
  list-recall rows for metric name, score, and normalized prediction.
- Final main/OOD/LowRank audits re-run the tracked byte/hash/row/boundary data
  verification and require the declared judge protocol. OOD additionally
  checks the vendored scorer hash and pinned ATM revision.

## Real GPU samples

Deterministic inference was rerun on one or two leading LoCoMo, ATM, or
MetisTest instances and compared with the corresponding frozen raw evidence.
The following method paths matched on instance/question, raw answer, prompt
tokens, context/query policy, committed-step count where applicable, and query
audit state:

- Qwen3.5-4B no-context: 1/1.
- Qwen3.5-4B full-context: 1/1.
- DenseRAG Qwen3.5-4B: 1/1.
- TempLoRA Qwen3.5-4B official-like profile: 1/1.
- delta-Mem Qwen3-4B TSW memory-only profile: 1/1.
- Metis 4B v2.4 step 14000 main-table path: 2/2.
- Metis 4B v2.4 step 14000 OOD/ATM path: 2/2.
- Metis 4B query-BQ ablation step 14000: 1/1.
- LowRankMemory Metis 4B step 14000, rank 1: 1/1, including projection-event
  count and projection summary.
- Metis 9B main-table path: 1/1.
- Metis 27B paired-layer path: instance, question, memory steps, and semantics
  matched; the generated date used the equivalent form `May 1st` instead of
  `May 1`.

Only implementation module paths differed in the LowRankMemory metadata,
because the code moved from the experiment repository into `eval/`; the
algorithmic and output fields matched.

## Validation boundary

Representative OOD inference and three-repeat semantic judging completed with
zero judge failures. Complete 77-cell main and 14-cell OOD reruns are still in
progress; this file must be updated with their audit paths and score deltas
before publication. Aggregate agreement alone is not sufficient: final review
also requires exact row/ID coverage, raw sampling, query audits, model load
reports, Temp-LoRA seed/device-map evidence, and judge-source metadata.
