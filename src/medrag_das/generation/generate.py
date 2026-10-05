"""Generation Unit — assemble the prompt and call the (fused) backend."""
from __future__ import annotations

from ..backends.base import GenerationResult, LLMBackend
from .prompts import build_prompt


def generate_answer(
    backend: LLMBackend,
    question: str,
    options: list[str],
    context: list[str] | None,
    *,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> GenerationResult:
    """Build the prompt (with or without retrieved context) and generate.

    The caller is responsible for having fused the correct adapter (DAS) and/or supplying
    `context` (RAG) beforehand — this function just formats and generates.
    """
    prompt = build_prompt(question, options, context=context)
    return backend.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
