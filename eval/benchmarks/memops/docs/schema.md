# MemOP Schema

Last updated: 2026-06-26.

MemOP uses JSONL, one record per question.

```json
{
  "task_type": "memop",
  "dataset": "metis_test | metisops_v23",
  "split": "test",
  "setting": "turn_commit | full_segments | gold_turns",
  "instance_id": "...",
  "source_sample_id": "...",
  "operation": "remember | update | forget | reflect | mixed",
  "subtask": "...",
  "context": [
    {
      "turn_id": "T0",
      "segment_id": "S1",
      "messages": [{"role": "user", "content": "..."}],
      "content": "USER: ...\nASSISTANT: ..."
    }
  ],
  "memory_steps": [
    {
      "step_id": 1,
      "turn_start": "T0",
      "turn_end": "T0",
      "commit_granularity": "one_user_assistant_turn",
      "content": "USER: ...\nASSISTANT: ..."
    }
  ],
  "question": "...",
  "answer": "...",
  "rubric": {},
  "evidence": ["T0"],
  "metadata": {}
}
```

The minimum scorer contract is compatible with MemQA-style scoring:

- `question`
- `answer`
- model output from a baseline runner
- optional `rubric` for LLM judge
- grouping metadata such as `dataset`, `setting`, `operation`, and `subtask`

## Query Policy

For `memory_only` baseline runs, the query prompt must not include `context`,
`memory_steps`, `evidence`, `answer`, or `rubric`. The write/information phase
consumes `memory_steps`; the query phase consumes `question` plus the baseline's
internal memory state.

MetisOps source artifacts contain `trajectory_ops`, but those tasks are excluded
from the formal MemOP evaluation line and should not appear in normalized
release files.
