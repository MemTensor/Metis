# ATM-Bench evaluator subset

- Upstream: `https://github.com/ByteDance-Seed/ATM-Bench`
- Revision: `d463445614ad78a48736b98ab901795f7ecaf3da`
- License: MIT; the upstream license is preserved in `LICENSE`.

Only the deterministic QA metric, normalizer, question-type helper, and judge
prompt required by the Metis ATM scorer are vendored. Upstream model agents,
data, retrieval code, caches, and command-line utilities are excluded.

Metis evaluation patch: `memqa/utils/evaluator/config.py` no longer imports the upstream
repository-wide `global_config.py`. That module searched local API-key files,
declared localhost endpoints, and created directories at import time; none of
those behaviors are used by the imported metric functions. The official metric
functions and judge prompt remain unchanged.

The upstream CLI-only `requests` and `tqdm` imports are also optional in this
subset so importing deterministic functions does not initialize unrelated CLI
dependencies. The release environment still installs both packages.
