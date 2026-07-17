#!/usr/bin/env python3
"""Build the paper evaluation JSONL files from user-supplied source data."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path
from typing import Callable

from eval.data.processors import locomo, metis_test, metisops, nextmem, ood
from eval.data.verify import verify


DEFAULT_RAW_ROOT = Path(__file__).resolve().parent / "raw"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent
DATASET_IDS = {
    "locomo": {"locomo_tps16"},
    "nextmem": {"nextmem_stm"},
    "metis_test": {"metis_test_nomixed"},
    "metisops": {"metisops_v23_full_segments", "metisops_v23_gold_turns"},
    "atm": {"atm"},
    "memdaily": {"memdaily"},
}


def build_locomo(raw_root: Path, output_dir: Path) -> None:
    source = raw_root / "locomo/locomo10.json"
    output = output_dir / "memqa/locomo_evidence_sessions_tps16_20260626.jsonl"
    records, _ = locomo.convert(
        Namespace(
            input=source,
            output=output,
            review=None,
            turns_per_step=16,
            include_adversarial=False,
            limit=0,
        )
    )
    locomo.write_jsonl(output, records)


def build_nextmem(raw_root: Path, output_dir: Path) -> None:
    records, _ = nextmem.collect_records(raw_root / "nextmem", tokenizer=None)
    normalized = nextmem.build_split(records, split_kind="full", per_dataset=0)
    nextmem.write_jsonl(
        output_dir / "memqa/nextmem_stm_official_task2_20260622.jsonl",
        normalized,
    )


def build_metis_test(raw_root: Path, output_dir: Path) -> None:
    records, _ = metis_test.build_records(raw_root / "metis_test")
    records = [
        row
        for row in records
        if sum(len(step["content"]) for step in row["memory_steps"]) < 50_000
        and len(str(row["answer"])) < 1_000
        and row["operation"] != "mixed"
    ]
    for row in records:
        row["metadata"]["source_dir"] = "user-provided"
    metis_test.write_jsonl(
        output_dir / "memops/metis_test_v2_4_memoryops_test_nomixed_20260707.jsonl",
        records,
    )


def build_metisops(raw_root: Path, output_dir: Path) -> None:
    full, gold, _ = metisops.build_records(
        raw_root / "metisops/memoryops_v23_30topic_artifacts.zip"
    )
    metisops.write_jsonl(
        output_dir / "memops/metisops_v23_full_segments_20260626.jsonl",
        full,
    )
    metisops.write_jsonl(
        output_dir / "memops/metisops_v23_gold_turns_20260626.jsonl",
        gold,
    )


def build_atm(raw_root: Path, output_dir: Path) -> None:
    ood.write_jsonl(output_dir / "ood/atm.jsonl", ood.build_atm(raw_root / "atm"))


def build_memdaily(raw_root: Path, output_dir: Path) -> None:
    ood.write_jsonl(
        output_dir / "ood/memdaily.jsonl",
        ood.build_memdaily(raw_root / "memdaily/memdaily.json"),
    )


BUILDERS: dict[str, Callable[[Path, Path], None]] = {
    "locomo": build_locomo,
    "nextmem": build_nextmem,
    "metis_test": build_metis_test,
    "metisops": build_metisops,
    "atm": build_atm,
    "memdaily": build_memdaily,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        choices=[*BUILDERS, "all"],
        help="Dataset family to build; repeat as needed. The default is all.",
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    selected = list(BUILDERS) if not args.dataset or "all" in args.dataset else args.dataset
    expected_ids: set[str] = set()
    for name in selected:
        BUILDERS[name](args.raw_root, args.output_dir)
        expected_ids.update(DATASET_IDS[name])
        print(json.dumps({"event": "built", "dataset": name}, ensure_ascii=False))

    report = verify(args.output_dir, only=expected_ids)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit("prepared data failed eval/data/manifest.json verification")


if __name__ == "__main__":
    main()
