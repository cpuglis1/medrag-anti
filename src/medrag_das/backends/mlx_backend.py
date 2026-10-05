"""MLX backend — PRIMARY path, matches the paper (Apple M2 Pro, MLX, no bitsandbytes).

Implemented against `mlx-lm`. The public surface of mlx-lm shifts between releases, so the
version-sensitive calls (generate sampler, convert, lora config keys) are guarded with
fallbacks and clear errors. Pin a known-good mlx-lm in your environment and re-check if an
API call fails.

Paper-faithful behavior:
  * 8-bit quantized base (~3.98 GiB live) — produced by a one-time mlx_lm.convert -q
  * QLoRA: num_layers=4 (not 16), batch_size=2, train-loss-plateau early stopping
  * exactly one adapter fused at a time (WARNINGS.md §7) — fuse reloads base+adapter
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

from ..config import REPO_ROOT
from .base import GenerationResult


def _ensure_mlx_model(model_id: str, quantize_bits: int | None) -> str:
    """Return a local MLX model path, converting+quantizing from HF once if needed.

    If `model_id` is already a local directory, it is used as-is. Otherwise it is treated as
    an HF repo and converted to data/mlx/<repo>-q{bits} via `mlx_lm.convert` (one-time).
    Requires HF auth for gated models (set HF_TOKEN / `huggingface-cli login`).
    """
    p = Path(model_id)
    if p.exists() and p.is_dir():
        return str(p)

    safe = model_id.replace("/", "__")
    suffix = f"-q{quantize_bits}" if quantize_bits else "-fp16"
    target = REPO_ROOT / "data" / "mlx" / f"{safe}{suffix}"
    if target.exists():
        return str(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "mlx_lm.convert", "--hf-path", model_id, "--mlx-path", str(target)]
    if quantize_bits:
        cmd += ["-q", "--q-bits", str(quantize_bits)]
    print(f"[mlx] converting {model_id} -> {target} ({'q%d' % quantize_bits if quantize_bits else 'fp16'})")
    subprocess.run(cmd, check=True)
    return str(target)


def _mlx_generate(model, tokenizer, prompt: str, max_tokens: int, temperature: float) -> str:
    """Call mlx_lm.generate across API variants (sampler kwarg vs legacy temp kwarg)."""
    from mlx_lm import generate as g

    # Modern mlx-lm: temperature is supplied via a sampler object.
    try:
        from mlx_lm.sample_utils import make_sampler

        return g(model, tokenizer, prompt=prompt, max_tokens=max_tokens,
                 sampler=make_sampler(temp=temperature), verbose=False)
    except (ImportError, TypeError):
        # Legacy mlx-lm: temp kwarg directly on generate.
        return g(model, tokenizer, prompt=prompt, max_tokens=max_tokens,
                 temp=temperature, verbose=False)


class MLXBackend:
    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._base_path: str | None = None
        self._fused_adapter: Path | None = None

    # --- lifecycle --------------------------------------------------------------
    def load_base(self, model_id: str, quantize_bits: int = 8) -> None:
        try:
            from mlx_lm import load
        except ImportError as e:  # pragma: no cover - env-dependent
            raise ImportError(
                "mlx-lm not installed. On Apple Silicon: `pip install -e .[mlx]`. "
                "MLX is Apple-Silicon-only; use the torch backend elsewhere."
            ) from e

        self._base_path = _ensure_mlx_model(model_id, quantize_bits)
        self._model, self._tokenizer = load(self._base_path)
        self._fused_adapter = None

    def fuse_adapter(self, adapter_path: Path | None) -> None:
        """Swap to a single adapter by reloading base (+adapter). Releases the previous one."""
        if self._base_path is None:
            raise RuntimeError("call load_base() before fuse_adapter().")
        from mlx_lm import load

        if adapter_path is None:
            self._model, self._tokenizer = load(self._base_path)
            self._fused_adapter = None
            return

        try:
            self._model, self._tokenizer = load(self._base_path, adapter_path=str(adapter_path))
        except TypeError:
            # Older mlx-lm: apply adapters after loading base.
            from mlx_lm.tuner.utils import load_adapters

            self._model, self._tokenizer = load(self._base_path)
            load_adapters(self._model, str(adapter_path))
        self._fused_adapter = adapter_path

    # --- inference --------------------------------------------------------------
    def generate(self, prompt: str, max_new_tokens: int, temperature: float) -> GenerationResult:
        if self._model is None:
            raise RuntimeError("call load_base() first.")
        t0 = time.time()
        text = _mlx_generate(self._model, self._tokenizer, prompt, max_new_tokens, temperature)
        return GenerationResult(
            text=text,
            prompt_tokens=len(self._tokenizer.encode(prompt)),
            completion_tokens=len(self._tokenizer.encode(text)) if text else 0,
            seconds=time.time() - t0,
        )

    # --- training ---------------------------------------------------------------
    def train_adapter(self, *, data_dir: Path, out_dir: Path, **hp) -> dict:
        """QLoRA fine-tune via `mlx_lm.lora`, with train-loss-plateau early stopping."""
        if self._base_path is None:
            raise RuntimeError("call load_base() before train_adapter().")
        out_dir.mkdir(parents=True, exist_ok=True)

        es = hp.get("early_stop", {}) or {}
        cfg = {
            "model": self._base_path,
            "train": True,
            "data": str(data_dir),
            "adapter_path": str(out_dir),
            "batch_size": hp["batch_size"],
            "iters": hp["max_iters"],
            "learning_rate": hp["learning_rate"],
            "num_layers": hp["lora_layers"],       # paper: 4 (older mlx-lm: `lora_layers`)
            "steps_per_report": 10,
            "steps_per_eval": 50,
            "save_every": hp["max_iters"],
            "lora_parameters": {
                "rank": hp["lora_rank"],
                "scale": float(hp["lora_alpha"]),
                "dropout": 0.0,
            },
        }
        cfg_path = out_dir / "lora_config.yaml"
        with open(cfg_path, "w") as fh:
            yaml.safe_dump(cfg, fh)

        metrics = self._run_lora_with_plateau_stop(
            cfg_path,
            monitor=es.get("monitor", "train_loss"),
            patience=int(es.get("patience", 20)),
            min_delta=float(es.get("min_delta", 1e-3)),
        )
        (out_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
        return metrics

    def _run_lora_with_plateau_stop(self, cfg_path: Path, monitor: str, patience: int, min_delta: float) -> dict:
        key = "Val" if monitor == "val_loss" else "Train"
        pat = re.compile(r"Iter\s+(\d+):\s+(Train|Val) loss\s+([0-9.]+)")
        proc = subprocess.Popen(
            [sys.executable, "-m", "mlx_lm.lora", "--config", str(cfg_path)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        best, since, last_iter, final_loss, stopped = float("inf"), 0, 0, None, False
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            m = pat.search(line)
            if not m or m.group(2) != key:
                continue
            last_iter, final_loss = int(m.group(1)), float(m.group(3))
            if final_loss < best - min_delta:
                best, since = final_loss, 0
            else:
                since += 1
                if since >= patience:
                    proc.terminate()
                    stopped = True
                    break
        proc.wait()
        return {"final_loss": final_loss, "iters": last_iter, "early_stopped": stopped,
                "monitor": monitor}
