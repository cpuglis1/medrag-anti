"""End-to-end pipeline wiring tests using a fake backend (no models, no weights).

Proves that each of the 4 methods routes correctly: which get an adapter fused, which get
retrieved context injected into the prompt, and that parsing flows through.
"""
from pathlib import Path

import pytest

from medrag_das.adapters.registry import AdapterRegistry
from medrag_das.adapters.selector import OracleSelector
from medrag_das.backends.base import GenerationResult
from medrag_das.pipeline import Pipeline

SLUGS = ["pharmacology", "surgery", "unknown"]


class FakeBackend:
    def __init__(self, text):
        self.text = text
        self.fused = "INIT"
        self.prompts = []

    def load_base(self, *a, **k):
        pass

    def fuse_adapter(self, adapter_path):
        self.fused = None if adapter_path is None else Path(adapter_path).name

    def generate(self, prompt, max_new_tokens, temperature):
        self.prompts.append(prompt)
        return GenerationResult(self.text, 1, 1, 0.0)

    def train_adapter(self, **k):
        return {}


def _registry(backend, tmp_path, make_dirs=True):
    adir = tmp_path / "adapters"
    adir.mkdir(exist_ok=True)
    if make_dirs:
        for s in SLUGS:
            (adir / s).mkdir(exist_ok=True)
    return AdapterRegistry(backend, adir, SLUGS)


def _pipe(be, tmp_path, make_dirs=True, ctx=("CTX",)):
    return Pipeline(
        be,
        registry=_registry(be, tmp_path, make_dirs),
        selector=OracleSelector(SLUGS),
        retriever=lambda q: (list(ctx), 0.01),
    )


def test_baseline_no_adapter_no_context(tmp_path):
    be = FakeBackend('{"cop": "Aspirin", "cop index": 1, "why correct": "x"}')
    out = _pipe(be, tmp_path).run("baseline", "Q?", ["Aspirin", "B"])
    assert out.adapter_slug is None
    assert out.context == []
    assert be.fused is None                               # registry unfused for baseline
    assert out.answer.cop_index == 1
    assert "abstracts context" not in be.prompts[0]


def test_das_fuses_oracle_adapter(tmp_path):
    be = FakeBackend('{"cop": "B", "cop index": 2, "why correct": "x"}')
    out = _pipe(be, tmp_path).run("das", "Q?", ["A", "B"], gold_category_slug="surgery")
    assert out.adapter_slug == "surgery" and be.fused == "surgery"
    assert out.context == []
    assert "abstracts context" not in be.prompts[0]


def test_rag_adds_context_not_adapter(tmp_path):
    be = FakeBackend("I do not know")
    out = _pipe(be, tmp_path, ctx=("A1", "A2")).run("rag", "Q?", ["A", "B"])
    assert out.adapter_slug is None and be.fused is None
    assert out.context == ["A1", "A2"]
    assert "abstracts context" in be.prompts[0]
    assert out.answer.abstained is True


def test_das_rag_does_both(tmp_path):
    be = FakeBackend('{"cop": "A", "cop index": 1, "why correct": "x"}')
    out = _pipe(be, tmp_path).run("das_rag", "Q?", ["A", "B"], gold_category_slug="pharmacology")
    assert out.adapter_slug == "pharmacology" and be.fused == "pharmacology"
    assert out.context == ["CTX"]
    assert "abstracts context" in be.prompts[0]


def test_missing_adapter_raises(tmp_path):
    be = FakeBackend("x")
    pipe = _pipe(be, tmp_path, make_dirs=False)
    with pytest.raises(FileNotFoundError):
        pipe.run("das", "Q?", ["A", "B"], gold_category_slug="surgery")


def test_unknown_method_raises(tmp_path):
    be = FakeBackend("x")
    with pytest.raises(ValueError):
        _pipe(be, tmp_path).run("nonsense", "Q?", ["A", "B"])
