# Integrating into another Metis checkout

The required code change is additive: copy this `eval/` directory beside the
existing `metis/` package. The evaluation code imports the public Metis
interfaces (`metis.configuration_metis`, `metis.modeling_metis`, and the
current delta-checkpoint loader behavior) but does not modify model source.

For a source-only transfer:

```bash
rsync -a --delete eval/ /path/to/other/Metis/eval/
cd /path/to/other/Metis
python -m compileall -q eval
python -m pytest -q eval/tests
```

Keep the nested `eval/.gitignore`; it covers normalized JSONL, raw source
data, `eval/results/**` (except its README), `eval/artifacts/`, and
`eval/outputs/`. The destination root should continue to ignore `.env`.
Generated manifests and score metadata may contain machine-local paths, so the
whole result tree is local evidence rather than source. If the destination has a broad
`data` rule, add explicit negations for `eval/data/` plus `__init__.py`,
`README.md`, `prepare.py`, `manifest.json`, `verify.py`, and the complete
`processors/` package; otherwise Git may hide the data contract together with
local payloads.

The `eval/` directory works with `python -m ...` from the repository root
without editable installation. Paths to data, models, and
checkpoints are resolved at runtime; no symlink or server filesystem layout is
assumed. Its own `.gitignore` protects user-supplied data, local assets, and the
complete in-tree result evidence even when only `eval/` is copied.

For an offline host, place Qwen snapshots under
`eval/artifacts/models/Qwen3.5-4B`, `Qwen3.5-9B`, and `Qwen3.5-27B`, or set
`METIS_BASE_MODEL_ROOTS` to an OS-path-separated list of mirror roots. The
loader falls back to public Hugging Face IDs only after local resolution fails.

Before GPU execution, compare the destination's Metis loader against the source
commit recorded in `PROVENANCE.md`, run data verification, run the ablation
static audit, and expand all three declarative matrices. A source-compatible
checkout should not require changes under `metis/`.
