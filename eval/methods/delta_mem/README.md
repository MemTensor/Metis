# delta-Mem

The formal baseline uses the separately installed official `deltamem` runtime
from `https://github.com/declare-lab/delta-Mem` at revision
`5cd5d9153c7f408764728d953565201e198c39e2`, the
`Qwen/Qwen3-4B-Instruct-2507` base, and the
`declare-lab/delta-mem_qwen3_4b-instruct` adapter. The adapter is a delta-Mem
TSW artifact, not a PEFT LoRA.

Clone that revision and install/expose its `deltamem` package, then download the
adapter into the path configured by `delta_mem_4b_adapter` in
`eval/configs/assets.json`. The runner deliberately clears chat history and KV
cache after each write while preserving online delta state, which is the
paper's memory-only protocol rather than delta-Mem's full-history replay mode.
