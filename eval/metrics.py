"""Scoring metrics.

The paper reports "accuracy" and, for NOTA, "balanced accuracy." It also notes outputs were
often unparseable and partly hand-scored. Here we compute both plain and balanced accuracy
AND surface the parse-failure rate, because a scorer that silently maps parse failures to
"wrong" will understate/overstate systems differently (WARNINGS.md §5).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass
class ScoreReport:
    n: int
    n_scored: int            # excludes parse failures
    accuracy: float          # correct / n_scored
    balanced_accuracy: float # mean per-class recall over gold labels
    parse_fail_rate: float   # parse failures / n
    abstain_rate: float      # fraction where the model abstained

    def as_row(self) -> dict:
        return {
            "n": self.n,
            "n_scored": self.n_scored,
            "accuracy": round(self.accuracy, 4),
            "balanced_accuracy": round(self.balanced_accuracy, 4),
            "parse_fail_rate": round(self.parse_fail_rate, 4),
            "abstain_rate": round(self.abstain_rate, 4),
        }


def score(predictions: list[dict]) -> ScoreReport:
    """Score a list of per-question records.

    Each record: {gold_index:int|None, pred_index:int|None, parse_failed:bool,
                  abstained:bool, gold_label:any}
    `gold_index` is the 1-based correct option (for NOTA/FAKE the dataset convention decides
    what counts as correct — see datasets.py).
    """
    n = len(predictions)
    if n == 0:
        return ScoreReport(0, 0, 0.0, 0.0, 0.0, 0.0)

    parse_fail = sum(1 for p in predictions if p.get("parse_failed"))
    abstain = sum(1 for p in predictions if p.get("abstained"))

    scored = [p for p in predictions if not p.get("parse_failed")]
    correct = sum(1 for p in scored if p.get("pred_index") == p.get("gold_index"))
    accuracy = correct / len(scored) if scored else 0.0

    # Balanced accuracy = mean recall across gold classes.
    per_class_total: dict = defaultdict(int)
    per_class_correct: dict = defaultdict(int)
    for p in scored:
        g = p.get("gold_label", p.get("gold_index"))
        per_class_total[g] += 1
        if p.get("pred_index") == p.get("gold_index"):
            per_class_correct[g] += 1
    recalls = [per_class_correct[c] / per_class_total[c] for c in per_class_total if per_class_total[c]]
    balanced = sum(recalls) / len(recalls) if recalls else 0.0

    return ScoreReport(
        n=n,
        n_scored=len(scored),
        accuracy=accuracy,
        balanced_accuracy=balanced,
        parse_fail_rate=parse_fail / n,
        abstain_rate=abstain / n,
    )
