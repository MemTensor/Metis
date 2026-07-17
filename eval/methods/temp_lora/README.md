# Temp-LoRA

This adapter implements the temporary per-session LoRA baseline through the
shared `write`, `query`, and `reset` contract. Supply a public model identifier
or local model directory with `--model`; generated adapters are runtime state
and are not saved in the repository.

The paper's 27B cells require two GPUs. The tracked main/OOD matrices use the
audited `balanced` Hugging Face device map with a 76 GiB cap on each of two
visible devices. Use the matrix config as-is; for a single-cell launch, expose
exactly two GPUs or preserve the tracked `--max-memory 0:76GiB 1:76GiB`
arguments. Runtime metadata records the resolved device map for audit.

The paper launchers also fixed randomness. Main-table cells use seed
`20260702`; OOD cells use seed `20260714`. These values are declared per method
in the paper configs and propagated into runtime metadata. Do not substitute a
single global seed across both protocols.
