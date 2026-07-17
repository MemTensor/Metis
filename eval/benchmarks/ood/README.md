# OOD: ATM-Bench and MemDaily

The current OOD protocol contains only ATM and MemDaily normalized gold inputs.
RHELM is retired. Exact payload facts are in `eval/data/manifest.json`.

ATM deterministic metrics and the official judge prompt load from the pinned,
MIT-licensed subset in `eval/third_party/atm_bench/`; there is no ignored
official-code dependency. Semantic calls use an OpenAI-compatible endpoint and
an API-key environment variable chosen by the CLI. No secret or internal
endpoint is stored in configs.
