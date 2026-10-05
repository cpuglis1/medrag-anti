"""Robust extraction of the model's answer.

The paper asked the model to emit JSON {'cop', 'cop index', 'why correct'} but reports that
outputs "did not follow that format" and required "a mixture of text processing techniques
and manual review." This module is a best-effort automated reconstruction of that step.

Tip: always log `raw`, and report the `parse_failed` rate alongside accuracy — a silent parse
failure scored as "wrong" (or "right") corrupts the metric.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

# Phrases that signal the model abstained. On Med-HALT NOTA/FAKE an abstention is effectively
# the *correct* behavior, so abstention detection is as important as option extraction.
_ABSTAIN_PATTERNS = [
    r"i do not know",
    r"i don'?t know",
    r"none of the above",
    r"cannot (?:be )?determin",
    r"no correct answer",
    r"fictional",
    r"nonsensical",
    r"absurd",
]


@dataclass
class ParsedAnswer:
    cop: str | None          # the chosen option text
    cop_index: int | None    # 1-based index if recoverable, else None
    why_correct: str | None  # rationale text
    abstained: bool          # model declined / flagged unanswerable
    parse_failed: bool       # nothing structured could be recovered
    raw: str                 # original generation (always keep)


def _first_json_object(text: str) -> dict | None:
    """Find the first balanced {...} block and try to json.loads it (with light repair)."""
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    blob = text[start : i + 1]
                    for candidate in (blob, blob.replace("'", '"')):
                        try:
                            obj = json.loads(candidate)
                            if isinstance(obj, dict):
                                return obj
                        except json.JSONDecodeError:
                            continue
                    break
        start = text.find("{", start + 1)
    return None


def _coerce_index(value) -> int | None:
    if value is None:
        return None
    m = re.search(r"\d+", str(value))
    return int(m.group()) if m else None


def parse_answer(raw: str) -> ParsedAnswer:
    """Extract a structured answer from a raw generation. Never raises."""
    text = (raw or "").strip()
    abstained = any(re.search(p, text, flags=re.IGNORECASE) for p in _ABSTAIN_PATTERNS)

    obj = _first_json_object(text)
    if obj is not None:
        cop = obj.get("cop") or obj.get("correct option")
        idx = _coerce_index(obj.get("cop index") or obj.get("cop_index"))
        why = obj.get("why correct") or obj.get("why_correct")
        return ParsedAnswer(
            cop=str(cop) if cop is not None else None,
            cop_index=idx,
            why_correct=str(why) if why is not None else None,
            abstained=abstained,
            parse_failed=(cop is None and idx is None),
            raw=raw,
        )

    # Fallback: look for "option 3", "answer is 2", "(4)", a bare leading digit, etc.
    idx = None
    for pat in (
        r"(?:option|answer|choice)\D{0,10}(\d+)",
        r"\(\s*(\d+)\s*\)",
        r"^\s*(\d+)\b",
    ):
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            idx = int(m.group(1))
            break

    return ParsedAnswer(
        cop=None,
        cop_index=idx,
        why_correct=text or None,
        abstained=abstained,
        parse_failed=(idx is None and not abstained),
        raw=raw,
    )
