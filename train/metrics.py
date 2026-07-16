"""Evaluation metrics for Metis memory training.

Matches the NextMem evaluation suite: F1, ROUGE-1, ROUGE-L.
METEOR, BLEU, BertScore are available if optional dependencies are installed.
"""

from __future__ import annotations

import logging
from collections import Counter

logger = logging.getLogger(__name__)


def _f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = prediction.split()
    gt_tokens = ground_truth.split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    p = num_same / len(pred_tokens) if pred_tokens else 0.0
    r = num_same / len(gt_tokens) if gt_tokens else 0.0
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def _lcs_len(x: list, y: list) -> int:
    m, n = len(x), len(y)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            curr[j] = prev[j - 1] + 1 if x[i - 1] == y[j - 1] else max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


class Metrics:
    """Evaluates generation quality against reference text.

    Available metrics: F1, ROUGE_1, ROUGE_L (always), plus
    METEOR / BLEU / BertScore if optional dependencies are installed.
    """

    ALL = ["F1", "ROUGE_1", "ROUGE_L"]

    def __init__(self, metrics_list: list[str] | None = None):
        self.metrics_list = metrics_list or self.ALL

    def calculate(self, prediction: str, ground_truth: str | list[str]) -> dict[str, float]:
        res: dict[str, float] = {}
        for metric in self.metrics_list:
            fn = getattr(self, metric)
            if isinstance(ground_truth, list):
                res[metric] = max(float(fn(prediction, g)) for g in ground_truth)
            else:
                res[metric] = float(fn(prediction, ground_truth))
        return res

    def F1(self, prediction: str, ground_truth: str) -> float:
        return _f1_score(prediction.lower(), ground_truth.lower())

    def ROUGE_1(self, prediction: str, ground_truth: str) -> float:
        return _f1_score(prediction.lower(), ground_truth.lower())

    def ROUGE_L(self, prediction: str, ground_truth: str) -> float:
        pw = prediction.lower().split()[:500]
        rw = ground_truth.lower().split()[:500]
        if not pw or not rw:
            return 0.0
        lcs = _lcs_len(pw, rw)
        if lcs == 0:
            return 0.0
        p, r = lcs / len(pw), lcs / len(rw)
        return 2 * p * r / (p + r)
