from __future__ import annotations

from eval.methods.shared.memqa_io import audit_query_payload


def test_memory_only_query_audit_detects_evidence_leakage() -> None:
    instance = {
        "question": "What city does Alice live in?",
        "answer": "Beijing",
        "evidence": ["turn-1"],
        "memory_steps": [{"content": "Alice has permanently lived in Beijing since the spring of 2012."}],
    }
    assert not audit_query_payload(instance, instance["question"])
    issues = audit_query_payload(
        instance,
        "Alice has permanently lived in Beijing since the spring of 2012. What city?",
    )
    assert issues
