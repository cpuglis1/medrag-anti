"""Torch backend — PORTABILITY STUB (CUDA + bitsandbytes + PEFT).

NOT part of the paper (the paper used MLX because bitsandbytes has no Apple-Silicon support).
Provided so the system can run on NVIDIA hardware. Its numbers have NOT been reconciled with
the paper — treat any result from this path as a separate experiment (WARNINGS.md §7).

Sketch:
  * load_base: AutoModelForCausalLM.from_pretrained(..., quantization_config=BitsAndBytesConfig(
        load_in_8bit=True)) + AutoTokenizer
  * fuse_adapter: PeftModel.from_pretrained(base, adapter_path); to swap, unload first
        (model.unload() / reload base) — do not stack adapters
  * train_adapter: peft.LoraConfig(r, lora_alpha, target_modules=[...]) + SFTTrainer
"""
from __future__ import annotations

from pathlib import Path

from .base import GenerationResult


class TorchBackend:
    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._fused_adapter: Path | None = None

    def load_base(self, model_id: str, quantize_bits: int = 8) -> None:
        raise NotImplementedError("TorchBackend.load_base: HF + BitsAndBytesConfig(load_in_8bit=True).")

    def fuse_adapter(self, adapter_path: Path | None) -> None:
        self._fused_adapter = adapter_path
        raise NotImplementedError("TorchBackend.fuse_adapter: PeftModel.from_pretrained; unload before swap.")

    def generate(self, prompt: str, max_new_tokens: int, temperature: float) -> GenerationResult:
        raise NotImplementedError("TorchBackend.generate: tokenizer + model.generate.")

    def train_adapter(self, *, data_dir: Path, out_dir: Path, **hparams) -> dict:
        raise NotImplementedError("TorchBackend.train_adapter: peft LoraConfig + SFTTrainer.")
