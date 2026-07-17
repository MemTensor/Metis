# Evaluation data

The normalized evaluation payload is deliberately not stored in Git. Download
the public Metis evaluation dataset into this directory while preserving the
paths in `manifest.json`:

```bash
python -m eval.data.download --repo-id ORGANIZATION/Metis-Eval
python -m eval.data.verify
```

You may instead download the dataset by another client. The resulting layout
must be:

```text
eval/data/
  memqa/*.jsonl
  memops/*.jsonl
  ood/{atm,memdaily}.jsonl
```

`python -m eval.data.verify` checks every file's byte size, SHA-256 digest, row
count, and boundary instance IDs. A runner will not silently select a similarly
named historical split.

The final Hugging Face dataset repository ID is intentionally not hard-coded
until it exists. Set `METIS_EVAL_DATASET_REPO` or pass `--repo-id`. Before the
payload is published, every row in `manifest.json` marked
`license_review_required` must receive a completed redistribution review and a
dataset card containing upstream attribution and terms.
