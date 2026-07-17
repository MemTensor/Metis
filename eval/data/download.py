#!/usr/bin/env python3
"""Download the normalized Metis evaluation payload from Hugging Face."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_DATA_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("METIS_EVAL_DATASET_REPO"),
        help="Hugging Face dataset repository, for example ORGANIZATION/Metis-Eval.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    if not args.repo_id:
        parser.error("pass --repo-id or set METIS_EVAL_DATASET_REPO")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required; create the environment from "
            "eval/environments/paper-eval-minimal-cu118.yml"
        ) from exc

    args.data_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.data_dir,
        allow_patterns=["memqa/*.jsonl", "memops/*.jsonl", "ood/*.jsonl"],
    )
    from eval.data.verify import verify

    report = verify(args.data_dir)
    if not report["ok"]:
        raise RuntimeError("Downloaded payload failed eval/data/manifest.json verification")
    print(f"verified {report['files']} files ({report['bytes']} bytes)")


if __name__ == "__main__":
    main()
