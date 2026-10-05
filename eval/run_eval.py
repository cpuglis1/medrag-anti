"""Run the 4-method × 3-dataset evaluation matrix (paper Table 1).

Reconstructs the experimental grid. Builds the backend, KB retriever, adapter registry, and
selector from config, then runs each (method, dataset) pair and scores it.

Honesty hooks baked in (WARNINGS.md §3/§4/§5):
  * DAS uses the ORACLE selector by default (what the paper did) — reported as such.
  * For FAKE, a correct answer = abstention; selection is random.
  * Reports accuracy AND balanced accuracy AND parse-fail/abstain rates, so a high NOTA/FAKE
    score can be read against the abstention rate that likely produced it.
"""
from __future__ import annotations

import json
import time

from medrag_das.config import Config

from .datasets import LOADERS, EvalItem
from .metrics import score


def _build_runtime(cfg: Config):
    """Construct backend + retriever + registry + selector. Heavy; needs trained artifacts."""
    from medrag_das.adapters.registry import AdapterRegistry
    from medrag_das.adapters.selector import build_selector
    from medrag_das.backends import get_backend

    backend = get_backend(cfg.backend)
    backend.load_base(cfg.models["base_model"], cfg.models["quantize_bits"])

    registry = AdapterRegistry(backend, cfg.adapter_path("").parent, cfg.category_slugs)
    selector = build_selector(cfg.selection["strategy"], cfg.category_slugs, cfg.selection["fake_strategy"])

    # Retriever closure over the loaded FAISS KB.
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

    return backend, registry, selector, retriever


def run_matrix(cfg: Config, methods: list[str], datasets: list[str], limit: int | None = None) -> dict:
    from medrag_das.pipeline import Pipeline

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
            for it in items:
                out = pipe.run(method, it.question, it.options, gold_category_slug=it.gold_label)
                pred_index = out.answer.cop_index
                # For FAKE-style unanswerable items, an abstention counts as correct.
                if it.correct_is_abstain:
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
            report = score(preds)
            grid.setdefault(ds_name, {})[method] = report.as_row()
            print(f"[{ds_name:8s}] {method:9s} -> {json.dumps(report.as_row())}")

    return grid
