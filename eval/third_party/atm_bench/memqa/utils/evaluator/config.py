#!/usr/bin/env python3
"""Minimal vendored configuration for the ATM QA metric.

The upstream evaluator imports a repository-wide configuration module that
looks for local API-key files and creates output directories on import. Metis evaluation
uses only the official rubric and deterministic metric functions, so those
unrelated side effects are deliberately removed here.
"""

ABSTENTION_PHRASES = [
    "unknown",
    "abstention",
    "no information",
    "not available",
    "no evidence",
    "no evidnece",
    "insufficient information",
]

LLM_JUDGE_PROMPT = (
    "You are an evaluator, and you are given a task to evaluate a model predictions "
    "with a given question. Let's follow the instructions step by step to make a "
    "judgement.  "
    "1. As the first step, you need to check whether the prediction was really "
    "answering the question.  "
    "2. If the model prediction does provide a meaningful answer, judge whether "
    "the model Prediction matches the ground truth answer by reasoning according "
    "to the following steps:  "
    "2.1: Always assume the ground truth is correct.  "
    "2.2: Pay attention to theses special cases:  "
    'a. If the ground truth answer contains numbers, the value of "accuracy" '
    "is true only if numbers in ground truth and numbers in model predictions "
    'match very well; in case of math questions, "accuracy" is true only if '
    "the numbers in model predictions EXACTLY matches the numbers in ground truth;  "
    'b. If the ground truth answer contains time, and/or time range, "accuracy" '
    'is "true" only if if times and time ranges in ground truth and model '
    "predictions match very well.  "
    'c. If the ground truth answer contains a set of objects, "accuracy" is '
    '"true" if the model prediction covers most of the objects in the ground truth; '
    'however, "accuracy" if "false" if the  model prediction has a lot of objects '
    "that are not in the ground truth.  "
    'd. If the ground truth is something similar to "I don\'t know", "accuracy" '
    'is "true" only if the model prediction also implies the similar thing.  '
    "2.3: Even if the prediction statement is reasonable, if it conflicts with or does "
    'not match the ground truth, "accuracy" should be "false".  '
    '2.4. "Accuracy" is true if the ground truth information is covered by the prediction. '
    "The prediction is allowed to provide more information but should not be against "
    "the ground truth. If it is hard to  decide whether the prediction matches ground "
    'truth, "accuracy" should be "false".  '
    "Think step by step following the instructions above, and then make a judgment. "
    'Respond with only a single JSON blob with an "explanation" field that has your '
    'short(less than 100 word) reasoning  steps and an "accuracy" field which is '
    '"true" or "false".  '
    "Question: {{question}}  "
    "Ground truth: {{answer}}  "
    "Prediction: {{prediction}}"
)

EVALUATOR_CONFIG = {"abstention_phrases": ABSTENTION_PHRASES}

__all__ = [
    "ABSTENTION_PHRASES",
    "LLM_JUDGE_PROMPT",
    "EVALUATOR_CONFIG",
]
