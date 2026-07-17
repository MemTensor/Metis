# MemQA Scoring And Judge

Last updated: 2026-06-18.

## Main Metrics

Each answer should report:

- normalized F1
- exact match
- LLM-as-Judge score

The deterministic F1/EM follows the LoCoMo official spirit:

- lowercase
- remove punctuation
- remove articles-like stop words used by LoCoMo (`a`, `an`, `the`, `and`)
- compare token overlap

Category 5/adversarial handling should stay separate from answerable QA unless
the user explicitly asks to include it.

## LLM Judge

The released code defaults to an OpenAI-compatible endpoint and permits an
explicit endpoint/model override:

```text
base_url: https://api.openai.com
model: gpt-4.1-mini
temperature: 0.0
repeats: 3
aggregation: median score
```

The judge should see:

- question
- gold answer
- model output
- raw category
- baseline label

The judge should not see full hidden metadata that was unavailable to the model
unless it is necessary for scoring.

## Judge Prompt

System:

```text
You are a strict but fair evaluator for a memory question-answering benchmark. Accept semantically equivalent short answers. Do not require exact wording. Penalize hallucinated entities, wrong dates, wrong relationships, and answers that contradict the gold answer. Return JSON only.
```

User instruction:

```text
Grade model_output against gold_answer for the question. Return JSON with keys: score (0 to 1), pass (boolean), matched_points (array of strings), missed_points (array of strings), and rationale (short string). Use partial credit only when the answer is partly correct. If the model says the answer is unknown or unavailable when the gold answer is answerable, score 0.
```

## Guardrails

Do not add broad deterministic guardrails by default. If exact-value guardrails
are added later, they must be:

- narrow
- auditable
- recorded in the scored output
- listed in the run report

Any deterministic guardrail used for a published result must be migrated with
its original scorer and validated against the corresponding historical output;
no generic replacement guardrail is applied by this package.
