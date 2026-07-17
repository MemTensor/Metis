#!/usr/bin/env python3
"""Check whether the official delta-mem runtime is importable."""

from __future__ import annotations

import json
from eval.methods.delta_mem.delta_mem_baseline import DeltaMemBaseline


def main() -> None:
    print(json.dumps(DeltaMemBaseline.runtime_check(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
