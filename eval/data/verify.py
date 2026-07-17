#!/usr/bin/env python3
"""Verify the normalized evaluation payload against its tracked manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_DATA_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = DEFAULT_DATA_DIR / "manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_sha256(path: Path) -> str:
    """Hash evaluation content while excluding machine-local build metadata."""

    digest = hashlib.sha256()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row.get("metadata")
            if isinstance(metadata, dict):
                metadata = dict(metadata)
                metadata.pop("source_dir", None)
                metadata.pop("reference_token_count", None)
                row = {**row, "metadata": metadata}
            normalized = json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            digest.update((normalized + "\n").encode("utf-8"))
    return digest.hexdigest()


def scan_jsonl(path: Path) -> tuple[int, str | None, str | None]:
    rows = 0
    first = None
    last = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            instance_id = row.get("instance_id")
            first = instance_id if first is None else first
            last = instance_id
            rows += 1
    return rows, first, last


def verify(
    data_dir: Path,
    manifest_path: Path = DEFAULT_MANIFEST,
    only: set[str] | None = None,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    for item in manifest["files"]:
        if only is not None and item["id"] not in only:
            continue
        path = data_dir / item["path"]
        checks: dict[str, bool] = {"present": path.is_file()}
        observed: dict[str, Any] = {}
        if path.is_file():
            rows, first, last = scan_jsonl(path)
            observed = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "content_sha256": content_sha256(path),
                "rows": rows,
                "first_instance_id": first,
                "last_instance_id": last,
            }
            exact_payload = (
                observed["bytes"] == item["bytes"]
                and observed["sha256"] == item["sha256"]
            )
            content_match = observed["content_sha256"] == item["content_sha256"]
            observed["exact_payload"] = exact_payload
            checks.update(
                {
                    "payload": exact_payload or content_match,
                    "rows": observed["rows"] == item["rows"],
                    "first_instance_id": observed["first_instance_id"]
                    == item["first_instance_id"],
                    "last_instance_id": observed["last_instance_id"]
                    == item["last_instance_id"],
                }
            )
        records.append({"id": item["id"], "path": item["path"], "checks": checks, "observed": observed})
    return {
        "ok": all(all(record["checks"].values()) for record in records),
        "files": len(records),
        "bytes": sum(record["observed"].get("bytes", 0) for record in records),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--json", action="store_true", help="Print the complete machine-readable report.")
    args = parser.parse_args()
    report = verify(args.data_dir, args.manifest)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for record in report["records"]:
            state = "OK" if all(record["checks"].values()) else "FAIL"
            failed = [name for name, value in record["checks"].items() if not value]
            print(f"{state:4} {record['id']}: {', '.join(failed) if failed else record['path']}")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
