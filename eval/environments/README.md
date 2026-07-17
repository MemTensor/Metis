# Evaluation environments

- `paper-eval-original-cu118.yml` and
  `paper-eval-original-pip-freeze.txt` are sanitized exports from the original
  paper evaluation environment. They preserve a broad historical snapshot and
  are evidence, not a minimal dependency recommendation.
- `paper-eval-minimal-cu118.yml` is the small, human-maintained evaluation
  environment. Its core versions match the original environment; packages
  absent from that environment but required by the reorganized release are
  bounded explicitly.

Optional clean-host recipe (not used for the recorded validation):

```bash
conda env create -f eval/environments/paper-eval-minimal-cu118.yml
conda activate metis-paper-eval-cu118
```

The minimal file includes the Qwen3.5 linear-attention runtime used by the
paper checkpoints. It is maintained as a readable installation recipe, but a
fresh-host installation remains a release check; the broad original export is
the exact environment record used for the validation described here.

Run evaluation commands from the repository root so that the checked-out
`metis` and `eval` packages are importable. This repository currently has no
Python packaging metadata or `eval` extra, so no editable-install step is
assumed.

The original environment did **not** contain `deltamem` or `pytest`.
DenseRAG uses the existing Transformers runtime directly. delta-Mem is
intentionally exposed from its official repository at the
revision recorded in `eval/methods/delta_mem/README.md`; it is not silently
substituted with a similarly named package.

The recorded release validation deliberately reused the existing paper
environment. Tests were invoked directly where `pytest` was absent, and the
official delta-Mem checkout was exposed read-only on `PYTHONPATH`. No package
was installed into that environment.

CUDA driver and hardware compatibility are host responsibilities. The
`cu118` snapshot is the measured paper environment, not a promise that every
new GPU should use CUDA 11.8.
