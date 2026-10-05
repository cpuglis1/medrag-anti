"""MLX backend — PRIMARY path, matches the paper (Apple M2 Pro, MLX, no bitsandbytes).

Implemented against `mlx-lm` (https://github.com/ml-explore/mlx-examples/tree/main/llms).
The methods below are STUBS with precise contracts and the mlx-lm calls sketched in
comments. Fill them in and verify on-device; do not assume they reproduce the paper until
you have.

Paper-specific settings this backend must honor:
  * 8-bit quantized base, ~3.98 GiB live  (configs/default.yaml: quantize_bits)
  * QLoRA: 4 LoRA layers (not 16), batch size 2, early stop on train-loss plateau
  * exactly one adapter fused at a time (WARNINGS.md §7)
"""
from __future__ import annotations

import time
from pathlib import Path

from .base import GenerationResult


class MLXBackend:
    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._base_model_id: str | None = None
        self._fused_adapter: Path | None = None

    def load_base(self, model_id: str, quantize_bits: int = 8) -> None:
        # from mlx_lm import load
        # self._model, self._tokenizer = load(model_id)   # quantize via convert if needed
        raise NotImplementedError(
            "MLXBackend.load_base: load+quantize Llama-2-7B via mlx_lm.load / mlx_lm.convert "
            f"(bits={quantize_bits}). See paper §3.2. Expect ~3.98 GiB live."
        )

    def fuse_adapter(self, adapter_path: Path | None) -> None:
        # mlx-lm applies LoRA weights at load/generate time. To 'swap': reload base and
        # apply the new adapter, or use mlx_lm.tuner.utils.apply_lora_layers then load_adapters.
        # MUST release the previously fused adapter first.
        self._fused_adapter = adapter_path
        raise NotImplementedError(
            "MLXBackend.fuse_adapter: apply a single adapter (mlx_lm load_adapters / fuse). "
            "Release the previously fused one first — never hold 21 live (WARNINGS §7)."
        )

    def generate(self, prompt: str, max_new_tokens: int, temperature: float) -> GenerationResult:
        # from mlx_lm import generate as mlx_generate
        # t0 = time.time()
        # text = mlx_generate(self._model, self._tokenizer, prompt=prompt,
        #                     max_tokens=max_new_tokens, temp=temperature)
        _ = time.time()
        raise NotImplementedError(
            "MLXBackend.generate: call mlx_lm.generate with the fused model + tokenizer."
        )

    def train_adapter(self, *, data_dir: Path, out_dir: Path, **hparams) -> dict:
        # Mirrors `python -m mlx_lm.lora --train` with:
        #   --model <base> --data <data_dir> --adapter-path <out_dir>
        #   --lora-layers 4 --batch-size 2 --iters <max_iters> ...
        # Early stopping on train-loss plateau is NOT built into mlx_lm.lora — wrap the
        # training loop (or poll the loss log) and stop per configs early_stop.* .
        raise NotImplementedError(
            "MLXBackend.train_adapter: run mlx_lm QLoRA with lora_layers=4, batch_size=2, "
            "and custom train-loss-plateau early stopping (paper §3.4.2)."
        )
