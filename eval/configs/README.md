# Asset registry

`assets.json` is the only indirection layer between paper protocols and model
locations. Public backbone IDs may be replaced by local Hugging Face snapshots.
Metis and delta-Mem release artifacts live under ignored `eval/artifacts/` paths.
For a host-specific layout, copy the registry to
`eval/artifacts/assets.local.json` and pass that file with `--assets`; keep the
tracked paper registry unchanged.

The registry intentionally contains no server paths, usernames, API endpoints,
or credentials. The paper main table uses only the Metis 4B/9B/27B checkpoints
at steps 14000/8000/14000; older main-table candidates are not registered.

Metis delta checkpoints record the original base-model path in their own
manifest. The release loader first checks that path, then
`eval/artifacts/models/<basename>` and every directory in
`METIS_BASE_MODEL_ROOTS`, and finally the pinned public Qwen ID for known
basenames. This permits both offline mirrors and normal Hugging Face caching
without rewriting checkpoint metadata or changing its hash.
