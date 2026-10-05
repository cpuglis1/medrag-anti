"""Adapter selection — decide which of the 21 QLoRA adapters to fuse for a question.

⚠️ WARNINGS.md §4: the PAPER used an ORACLE. Each MedMCQA question ships a ground-truth
category label; the paper routed on that label. For Med-HALT FAKE (no categories) it routed
at RANDOM. There was NO learned classifier. Any claim about *deployed* performance on
unlabeled questions must use `ClassifierSelector`, which is an unimplemented stub here —
exactly the "category identification method" the paper lists as future work.
"""
from __future__ import annotations

import random
from typing import Protocol


class Selector(Protocol):
    def select(self, question: str, *, gold_category_slug: str | None = None) -> str:
        """Return the slug of the adapter to fuse."""
        ...


class OracleSelector:
    """Reproduces the paper: trust the dataset's ground-truth category label.

    This is NOT deployable — it requires the label the model is supposed to be tested on.
    Use only to reproduce the paper's DAS numbers on MedMCQA. Reported results assume a
    perfect router.
    """

    def __init__(self, valid_slugs: list[str], fake_strategy: str = "random", seed: int = 0) -> None:
        self._valid = set(valid_slugs)
        self._slugs = list(valid_slugs)
        self._fake_strategy = fake_strategy
        self._rng = random.Random(seed)

    def select(self, question: str, *, gold_category_slug: str | None = None) -> str:
        if gold_category_slug is None:
            # Med-HALT FAKE path: no category exists -> random adapter (paper §4.3/5).
            if self._fake_strategy == "random":
                return self._rng.choice(self._slugs)
            return "unknown"
        if gold_category_slug not in self._valid:
            return "unknown"
        return gold_category_slug


class ClassifierSelector:
    """Deployment router: infer the category from the question text alone (paper future work).

    Uses the loaded base LLM as a zero-shot classifier — it is asked to name the single best
    category from the 21 slugs, and the answer is matched back to a slug (fallback "unknown").
    This makes the system runnable on unlabeled questions, but it is UNVALIDATED against the
    oracle: do not report numbers from this path as if the oracle produced them (WARNINGS §4).
    A lightweight trained classifier would be a stronger, cheaper router — a natural next step.

    Note: for best routing the base model (no adapter fused) should be active when select() is
    called; running it with a stale adapter fused is a minor inaccuracy for a reconstruction.
    """

    def __init__(self, valid_slugs: list[str], backend, fake_strategy: str = "random", seed: int = 0) -> None:
        self._valid = list(valid_slugs)
        self._backend = backend
        self._fake_strategy = fake_strategy
        self._rng = random.Random(seed)

    def _classify(self, question: str) -> str:
        menu = ", ".join(self._valid)
        prompt = (
            "<s>[INST] You are a medical triage router. Choose the single most relevant "
            f"category for the question from this list: {menu}. Reply with ONLY the category "
            f"slug, nothing else.\nQuestion: {question} [/INST]"
        )
        text = self._backend.generate(prompt, max_new_tokens=16, temperature=0.0).text.lower()
        for slug in self._valid:                       # exact slug mention wins
            if slug in text:
                return slug
        for slug in self._valid:                       # else match on the first word of the slug
            if slug.split("_")[0] in text:
                return slug
        return "unknown"

    def select(self, question: str, *, gold_category_slug: str | None = None) -> str:
        # gold_category_slug is ignored by design — this is the unlabeled/deployment path.
        try:
            return self._classify(question)
        except Exception:
            # Never crash routing; fall back like the FAKE path.
            return self._rng.choice(self._valid) if self._fake_strategy == "random" else "unknown"


def build_selector(
    strategy: str,
    valid_slugs: list[str],
    fake_strategy: str = "random",
    backend=None,
) -> Selector:
    if strategy == "oracle":
        return OracleSelector(valid_slugs, fake_strategy=fake_strategy)
    if strategy == "classifier":
        if backend is None:
            raise ValueError("classifier strategy requires a loaded `backend` to route with.")
        return ClassifierSelector(valid_slugs, backend, fake_strategy=fake_strategy)
    raise ValueError(f"Unknown selection strategy {strategy!r} (expected 'oracle'|'classifier').")
