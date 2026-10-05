"""Run the 4-method × 3-dataset evaluation matrix (paper Table 1).

Reconstructs the experimental grid. Builds the backend, KB retriever, adapter registry, and
selector from config, then runs each (method, dataset) pair and scores it.

Notes:
  * DAS uses the ORACLE selector by default (what the paper did); set selection.strategy to
    `classifier` to route on question text for unlabeled/deployment use.
  * For Med-HALT FAKE, a correct answer = recognizing it as unanswerable (abstain); the paper
    selects a random adapter there.
  * Reports accuracy, balanced accuracy, and parse-fail/abstain rates per run.
  * Writes per-question raw generations to <out_dir>/raw/*.jsonl so scores are auditable.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from medrag_das.config import REPO_ROOT, Config

from .datasets import LOADERS, EvalItem
from .metrics import score


def build_retriever(cfg: Config):
    """Return a `retriever(question) -> (context, seconds)` closure over the FAISS KB."""
    from medrag_das.retrieval.build_kb import load_kb
    from medrag_das.retrieval.embedder import BioMedLMEmbedder
    from medrag_das.retrieval.search import faiss_search

    index, kb_matrix, meta = load_kb(cfg.retrieval.kb_path, cfg.retrieval.meta_path)
    embedder = BioMedLMEmbedder()

    def retriever(question: str):
        t0 = time.time()
        qv = embedder.embed_query(question)
        hits = faiss_search(qv, index, kb_matrix, cfg.retrieval.cosine_threshold, cfg.retrieval.top_k)
        context = [meta[h.idx]["text"] for h in hits]
        return context, time.time() - t0

    return retriever


def _build_runtime(cfg: Config):
    """Construct backend + retriever + registry + selector. Heavy; needs trained artifacts."""
    from medrag_das.adapters.registry import AdapterRegistry
    from medrag_das.adapters.selector import build_selector
    from medrag_das.backends import get_backend

    backend = get_backend(cfg.backend)
    backend.load_base(cfg.models["base_model"], cfg.models["quantize_bits"])

    adapters_dir = REPO_ROOT / cfg.adapters.dir
    registry = AdapterRegistry(backend, adapters_dir, cfg.category_slugs)

    # The classifier selector (deployment) needs the backend to route; oracle does not.
    selector = build_selector(
        cfg.selection["strategy"], cfg.category_slugs,
        fake_strategy=cfg.selection["fake_strategy"], backend=backend,
    )
    retriever = build_retriever(cfg)
    return backend, registry, selector, retriever


def _order_for_method(items: list[EvalItem], method: str) -> list[EvalItem]:
    """Group by adapter for DAS methods so the backend swaps/reloads as rarely as possible."""
    if method in ("das", "das_rag"):
        return sorted(items, key=lambda it: it.gold_label or "~")
    return items


def run_matrix(
    cfg: Config,
    methods: list[str],
    datasets: list[str],
    limit: int | None = None,
    out_dir: str | Path | None = None,
) -> dict:
    from medrag_das.pipeline import Pipeline

    out_dir = Path(out_dir or (REPO_ROOT / "data" / "results"))
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)

    backend, registry, selector, retriever = _build_runtime(cfg)
    pipe = Pipeline(backend, registry=registry, selector=selector, retriever=retriever,
                    gen_cfg=cfg.generation)
    label_to_slug = cfg.label_to_slug()

    grid: dict[str, dict] = {}
    for ds_name in datasets:
        loader = LOADERS[ds_name]
        items: list[EvalItem] = list(
            loader(label_to_slug, limit) if ds_name == "medmcqa" else loader(limit)
        )
        for method in methods:
            preds = []
            raw_path = out_dir / "raw" / f"{ds_name}__{method}.jsonl"
            with open(raw_path, "w") as raw_fh:
                for it in _order_for_method(items, method):
                    out = pipe.run(method, it.question, it.options, gold_category_slug=it.gold_label)
                    pred_index = out.answer.cop_index
                    if it.correct_is_abstain:
                        # FAKE: correct == abstain. Encode both gold and pred as the sentinel -1.
                        pred_index = -1 if out.answer.abstained else pred_index
                        gold_index = -1
                    else:
                        gold_index = it.gold_index
                    preds.append({
                        "gold_index": gold_index,
                        "pred_index": pred_index,
                        "gold_label": it.gold_label,
                        "parse_failed": out.answer.parse_failed,
                        "abstained": out.answer.abstained,
                    })
                    raw_fh.write(json.dumps({
                        "question": it.question,
                        "adapter": out.adapter_slug,
                        "pred_index": pred_index,
                        "gold_index": gold_index,
                        "abstained": out.answer.abstained,
                        "parse_failed": out.answer.parse_failed,
                        "raw": out.answer.raw,
                    }) + "\n")
            report = score(preds)
            grid.setdefault(ds_name, {})[method] = report.as_row()
            print(f"[{ds_name:8s}] {method:9s} -> {json.dumps(report.as_row())}")

    with open(out_dir / "results.json", "w") as fh:
        json.dump(grid, fh, indent=2)
    print(f"\nWrote {out_dir/'results.json'} (+ per-question raw/ logs).")
    return grid
