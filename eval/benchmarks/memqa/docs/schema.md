# MemQA Schema

Last updated: 2026-06-18.

The normalized MemQA instance schema is JSONL, one record per QA.

```json
{
  "task_type": "memqa",
  "dataset": "locomo",
  "split": "single_session",
  "instance_id": "locomo_<sample_id>__session_<n>__qa_<i>",
  "source_sample_id": "...",
  "session": {
    "session_id": "session_3",
    "date_time": "2023-06-09",
    "speaker_a": "Caroline",
    "speaker_b": "Melanie"
  },
  "context": [
    {
      "dia_id": "D3:1",
      "speaker": "Caroline",
      "text": "...",
      "blip_caption": null,
      "img_url": null,
      "query": null
    }
  ],
  "memory_steps": [
    {
      "step_id": 1,
      "turn_start": "D3:1",
      "turn_end": "D3:8",
      "content": "DATE: ..."
    }
  ],
  "question": "...",
  "answer": "...",
  "evidence": ["D3:1"],
  "metadata": {
    "raw_category": 2,
    "category_name": null,
    "category_name_status": "unconfirmed",
    "is_adversarial": false,
    "source_file": "locomo10.json",
    "evidence_session": 3,
    "evidence_turns": [1]
  }
}
```
## Model Output Schema

Baseline runners should output JSONL with:

```json
{
  "run_id": "...",
  "baseline": "base_no_context | base_full_context | dense_rag | metis | ttt",
  "model_label": "...",
  "model_path": "...",
  "instance_id": "...",
  "raw_output": "...",
  "prompt_tokens": 0,
  "latency_sec": 0.0,
  "generation_config": {},
  "context_policy": "..."
}
```

Additional baseline-specific fields are allowed, but the above fields should be
present for scoring.

## Scored Output Schema

Scored JSONL should add:

```json
{
  "score": {
    "normalized_f1": 0.0,
    "exact_match": false,
    "llm_judge_score": 0.0,
    "llm_judge_pass": false,
    "judge_source": "api_median",
    "attempt_scores": [0.0, 0.0, 0.0]
  }
}
```
