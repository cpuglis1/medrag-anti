"""End-to-end orchestration of the four experimental methods (paper §4.3).

methods:
  baseline  -> no RAG, no DAS (plain quantized Llama-2-7B)
  das       -> fuse category adapter, no retrieved context
  rag       -> retrieve context, no adapter
  das_rag   -> fuse adapter AND retrieve context  (the final system)

This ties together: retrieval (search), selection+fusion (selector + registry), generation,
and parsing. The heavy pieces (backend, embedder, KB) are injected so the pipeline itself is
testable with fakes.
"""
from __future__ import annotations

from dataclasses import dataclass

from .adapters.registry import AdapterRegistry
from .adapters.selector import Selector
from .generation.generate import generate_answer
from .parsing import ParsedAnswer, parse_answer

METHODS = ("baseline", "das", "rag", "das_rag")


@dataclass
class PipelineOutput:
    answer: ParsedAnswer
    context: list[str]
    adapter_slug: str | None
    retrieval_seconds: float
    generation_seconds: float


class Pipeline:
    def __init__(
        self,
        backend,
        *,
        registry: AdapterRegistry | None = None,
        selector: Selector | None = None,
        retriever=None,           # callable(question) -> (list[str] context, seconds)
        gen_cfg: dict | None = None,
    ) -> None:
        self.backend = backend
        self.registry = registry
        self.selector = selector
        self.retriever = retriever
        self.gen_cfg = gen_cfg or {}

    def run(
        self,
        method: str,
        question: str,
        options: list[str],
        *,
        gold_category_slug: str | None = None,
    ) -> PipelineOutput:
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")

        use_rag = method in ("rag", "das_rag")
        use_das = method in ("das", "das_rag")

        # --- DAS: choose + fuse exactly one adapter ---------------------------------
        slug = None
        if use_das:
            if self.selector is None or self.registry is None:
                raise RuntimeError("DAS method requires a selector and an adapter registry.")
            slug = self.selector.select(question, gold_category_slug=gold_category_slug)
            self.registry.swap_to(slug)
        elif self.registry is not None:
            self.registry.unfuse()        # ensure baseline/rag run on the plain base model

        # --- RAG: retrieve context --------------------------------------------------
        context: list[str] = []
        retrieval_seconds = 0.0
        if use_rag:
            if self.retriever is None:
                raise RuntimeError("RAG method requires a retriever.")
            context, retrieval_seconds = self.retriever(question)

        # --- generate + parse -------------------------------------------------------
        result = generate_answer(
            self.backend, question, options, context=context or None,
            max_new_tokens=self.gen_cfg.get("max_new_tokens", 512),
            temperature=self.gen_cfg.get("temperature", 0.0),
        )
        return PipelineOutput(
            answer=parse_answer(result.text),
            context=context,
            adapter_slug=slug,
            retrieval_seconds=retrieval_seconds,
            generation_seconds=result.seconds,
        )
