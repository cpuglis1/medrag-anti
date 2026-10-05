"""Import + wiring smoke test that needs NO models, weights, or datasets.

Confirms the pure-logic pieces (config, categories, prompts, parser, search math, selector)
load and behave. Heavy paths (backends, embedder, KB, training) are intentionally untouched.

    python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from medrag_das.config import load_config  # noqa: E402
from medrag_das.generation.prompts import build_prompt  # noqa: E402
from medrag_das.parsing import parse_answer  # noqa: E402
from medrag_das.retrieval.search import cosine_threshold_topk  # noqa: E402
from medrag_das.adapters.selector import build_selector  # noqa: E402


def main() -> int:
    cfg = load_config()
    assert len(cfg.categories) == 21, "expected 21 adapters/categories"
    print(f"[ok] config loaded; {len(cfg.categories)} categories; backend={cfg.backend}")

    p = build_prompt("What inhibits COX?", ["Aspirin", "Water", "Salt", "Air"], context=["abstract"])
    assert "[/INST]" in p and "abstracts context" in p
    print("[ok] prompt assembly")

    a = parse_answer('{"cop": "Aspirin", "cop index": 1, "why correct": "COX inhibition"}')
    assert a.cop_index == 1 and not a.parse_failed
    print("[ok] answer parsing")

    res = cosine_threshold_topk(
        np.array([1.0, 0.0]), np.array([[1.0, 0.0], [0.0, 1.0]]), threshold=0.5
    )
    assert [r.idx for r in res] == [0]
    print("[ok] cosine-threshold search")

    sel = build_selector("oracle", cfg.category_slugs, "random")
    assert sel.select("q", gold_category_slug="surgery") == "surgery"
    print("[ok] oracle selector")

    print("\nSMOKE TEST PASSED — pure-logic layer is wired. Model/data code is implemented "
          "but needs weights + datasets (and a version-pinned env) to run end-to-end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
