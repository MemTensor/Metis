"""Shared contract for parameterized-memory baselines.

The contract deliberately separates information/write inputs from query/read
inputs so lane-specific runners can audit context policy without depending on a
particular model implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class QueryResult:
    raw_output: str
    prompt_tokens: int = 0
    latency_sec: float = 0.0
    debug: dict[str, Any] = field(default_factory=dict)


class MemoryBaseline(Protocol):
    """Minimal interface shared by MemQA, MemOP, and future lanes."""

    method_id: str

    def reset(self) -> None:
        """Clear per-instance online state."""

    def write(self, step: dict[str, Any]) -> dict[str, Any]:
        """Consume one information-phase memory step."""

    def query(self, question: str) -> QueryResult:
        """Answer from memory state and the question only."""
