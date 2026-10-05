"""Backend interface shared by the MLX (primary) and Torch (stub) implementations.

A backend owns: loading the quantized base model, fusing exactly one QLoRA adapter at a
time, generating text, and training an adapter. Keeping this behind a Protocol lets the
rest of the system stay backend-agnostic.

Memory contract (WARNINGS.md §7): `fuse_adapter` must swap — load one adapter, releasing
any previously fused one. Never hold 21 adapters live (≈84 GiB). One at a time ≈ <5 GiB.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float


@runtime_checkable
class LLMBackend(Protocol):
    def load_base(self, model_id: str, quantize_bits: int = 8) -> None:
        """Load (and quantize) the base model into memory."""
        ...

    def fuse_adapter(self, adapter_path: Path | None) -> None:
        """Fuse a single QLoRA adapter with the base model.

        Passing None unfuses / returns to the plain base model. Implementations MUST release
        the previously fused adapter before fusing a new one.
        """
        ...

    def generate(self, prompt: str, max_new_tokens: int, temperature: float) -> GenerationResult:
        """Generate a completion for an already-formatted prompt string."""
        ...

    def train_adapter(self, *, data_dir: Path, out_dir: Path, **hparams) -> dict:
        """Train one QLoRA adapter. Returns a small metrics dict (final loss, iters, ...)."""
        ...
