"""Evaluation dataset loaders: MedMCQA (4k test), Med-HALT NOTA (1k), Med-HALT FAKE (1k).

Each loader yields normalized `EvalItem`s so the runner is dataset-agnostic. Two sources:

  * fixture=<path>  -> read pre-normalized JSONL (tests/fixtures/*.jsonl). Offline, no deps.
  * fixture=None    -> load from Hugging Face and map raw fields -> EvalItem.

Defining "correct" per dataset (paper §4.1):
  * MedMCQA: standard 4-option MCQ; gold = `cop` (0-based) + 1.
  * Med-HALT NOTA: "None of the Above" is an explicit option and is always correct; gold =
    the index of that option. correct_is_abstain=False (there IS a right option to pick).
  * Med-HALT FAKE: fabricated scenario, no answerable option; "correct" = recognizing it as
    unanswerable (abstain). gold_index=None, correct_is_abstain=True.

⚠️ The Hugging Face field names / dataset ids below are the best-known mapping and are marked
VERIFY. They are applied through `_normalize_mcq`, which raises loudly if an expected field is
missing — it will NOT silently mis-map. Confirm against the actual datasets before trusting
any score (WARNINGS.md §5). The fixture path is fully exercised by tests and needs no verify.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# --- Hugging Face sources (VERIFY ids/configs/fields against the live datasets) -----------
HF_MEDMCQA = "openlifescienceai/medmcqa"
HF_MEDMCQA_SPLIT = "validation"            # labeled ~4.2k; official `test` has hidden labels
HF_MEDHALT = "openlifescienceai/MedHALT"   # VERIFY: dataset id
HF_MEDHALT_NOTA_CONFIG = "reasoning_nota"  # VERIFY: config name
HF_MEDHALT_FAKE_CONFIG = "reasoning_FQT"   # VERIFY: config name (FQT = Fake Question Test)


@dataclass
class EvalItem:
    question: str
    options: list[str]
    gold_index: int | None          # 1-based correct option; None if "correct = abstain"
    gold_label: str | None          # adapter slug for DAS oracle (MedMCQA only)
    correct_is_abstain: bool        # True for FAKE-style unanswerable items


# --------------------------------------------------------------------------------------- #
# Fixture path (offline, pre-normalized)
# --------------------------------------------------------------------------------------- #
def _iter_fixture(path: str | Path, limit: int | None) -> Iterator[EvalItem]:
    with open(path) as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            if limit is not None and i >= limit:
                break
            row = json.loads(line)
            yield EvalItem(
                question=row["question"],
                options=list(row["options"]),
                gold_index=row.get("gold_index"),
                gold_label=row.get("gold_label"),
                correct_is_abstain=bool(row.get("correct_is_abstain", False)),
            )


# --------------------------------------------------------------------------------------- #
# HF mapping helpers
# --------------------------------------------------------------------------------------- #
def _require(row: dict, field: str, dataset: str):
    if field not in row:
        raise KeyError(
            f"{dataset}: expected field {field!r} not found (got {sorted(row)[:12]}...). "
            f"Update the VERIFY field mapping in eval/datasets.py."
        )
    return row[field]


def _normalize_mcq(
    row: dict,
    *,
    dataset: str,
    question_field: str,
    option_fields: list[str] | None,
    options_field: str | None,
    gold0_field: str | None,          # field holding the 0-based correct index (or None)
    label_field: str | None = None,   # category/subject field (MedMCQA)
    label_to_slug: dict[str, str] | None = None,
    nota_option_text: str | None = None,   # if set, gold = index of this option text
    correct_is_abstain: bool = False,
) -> EvalItem:
    """Map one raw HF row onto EvalItem. Raises on missing required fields."""
    question = _require(row, question_field, dataset)

    if option_fields is not None:
        options = [_require(row, f, dataset) for f in option_fields]
    else:
        options = list(_require(row, options_field, dataset))

    gold_index: int | None = None
    if correct_is_abstain:
        gold_index = None
    elif nota_option_text is not None:
        # NOTA: the correct option is the "None of the above" entry.
        matches = [i for i, o in enumerate(options) if str(o).strip().lower() == nota_option_text.lower()]
        if not matches:
            raise ValueError(f"{dataset}: no option matching {nota_option_text!r} in {options}")
        gold_index = matches[0] + 1
    elif gold0_field is not None:
        gold_index = int(_require(row, gold0_field, dataset)) + 1   # 0-based -> 1-based

    gold_label = None
    if label_field is not None:
        raw_label = row.get(label_field)
        gold_label = (label_to_slug or {}).get(raw_label) if label_to_slug else raw_label

    return EvalItem(question, options, gold_index, gold_label, correct_is_abstain)


# --------------------------------------------------------------------------------------- #
# Public loaders
# --------------------------------------------------------------------------------------- #
def load_medmcqa_test(
    label_to_slug: dict[str, str],
    limit: int | None = None,
    *,
    fixture: str | Path | None = None,
    split: str = HF_MEDMCQA_SPLIT,
) -> Iterator[EvalItem]:
    """MedMCQA MCQs. gold_index = cop(0-based)+1; gold_label = slug(subject_name)."""
    if fixture is not None:
        yield from _iter_fixture(fixture, limit)
        return

    from datasets import load_dataset  # lazy: keeps module importable without `datasets`

    ds = load_dataset(HF_MEDMCQA, split=split)
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        yield _normalize_mcq(
            row, dataset="medmcqa",
            question_field="question",
            option_fields=["opa", "opb", "opc", "opd"], options_field=None,
            gold0_field="cop",
            label_field="subject_name", label_to_slug=label_to_slug,
        )


def load_medhalt_nota(
    limit: int | None = None,
    *,
    fixture: str | Path | None = None,
    nota_option_text: str = "None of the above",
) -> Iterator[EvalItem]:
    """Med-HALT NOTA: the correct answer is always the 'None of the above' option."""
    if fixture is not None:
        yield from _iter_fixture(fixture, limit)
        return

    from datasets import load_dataset

    ds = load_dataset(HF_MEDHALT, HF_MEDHALT_NOTA_CONFIG, split="test")
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        # VERIFY: options may be a list field or opa..ope; adjust here if the schema differs.
        yield _normalize_mcq(
            row, dataset="medhalt_nota",
            question_field="question",
            option_fields=None, options_field="options",
            gold0_field=None, nota_option_text=nota_option_text,
        )


def load_medhalt_fake(
    limit: int | None = None,
    *,
    fixture: str | Path | None = None,
) -> Iterator[EvalItem]:
    """Med-HALT FAKE: fabricated scenario; correct behavior is to abstain."""
    if fixture is not None:
        yield from _iter_fixture(fixture, limit)
        return

    from datasets import load_dataset

    ds = load_dataset(HF_MEDHALT, HF_MEDHALT_FAKE_CONFIG, split="test")
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        yield _normalize_mcq(
            row, dataset="medhalt_fake",
            question_field="question",
            option_fields=None, options_field="options",
            gold0_field=None, correct_is_abstain=True,
        )


# Default fixture locations, so the runner/tests can grab offline data easily.
FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
FIXTURE_PATHS = {
    "medmcqa": FIXTURES / "medmcqa_sample.jsonl",
    "nota": FIXTURES / "nota_sample.jsonl",
    "fake": FIXTURES / "fake_sample.jsonl",
}

LOADERS = {
    "medmcqa": load_medmcqa_test,
    "nota": load_medhalt_nota,
    "fake": load_medhalt_fake,
}
