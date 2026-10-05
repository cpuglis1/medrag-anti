"""Torch backend — PORTABILITY path (CUDA + bitsandbytes + PEFT).

NOT part of the paper (the paper used MLX because bitsandbytes has no Apple-Silicon support).
Provided so the system can run on NVIDIA hardware. Its numbers have NOT been reconciled with
the paper — treat any result from this path as a separate experiment (WARNINGS.md §7).

Needs the `torch` extra: `pip install -e .[torch]`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .base import GenerationResult


class TorchBackend:
    def __init__(self) -> None:
        self._base = None          # untouched base (for clean adapter swaps)
        self._model = None         # active model (base or base+adapter)
        self._tokenizer = None
        self._model_id: str | None = None
        self._device = None

    def _import(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        return torch, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    def load_base(self, model_id: str, quantize_bits: int = 8) -> None:
        torch, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig = self._import()
        self._model_id = model_id
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        qcfg = None
        if quantize_bits == 8:
            qcfg = BitsAndBytesConfig(load_in_8bit=True)
        elif quantize_bits == 4:
            qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._base = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=qcfg,
            device_map="auto" if self._device == "cuda" else None,
            torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
        )
        self._model = self._base

    def fuse_adapter(self, adapter_path: Path | None) -> None:
        if self._base is None:
            raise RuntimeError("call load_base() before fuse_adapter().")
        if adapter_path is None:
            self._model = self._base
            return
        from peft import PeftModel

        # Swap cleanly: always attach the adapter to the pristine base, never stack.
        self._model = PeftModel.from_pretrained(self._base, str(adapter_path))

    def generate(self, prompt: str, max_new_tokens: int, temperature: float) -> GenerationResult:
        if self._model is None:
            raise RuntimeError("call load_base() first.")
        import torch

        t0 = time.time()
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=temperature > 0, temperature=max(temperature, 1e-6),
                pad_token_id=self._tokenizer.pad_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]          # completion tokens only
        text = self._tokenizer.decode(gen, skip_special_tokens=True)
        return GenerationResult(
            text=text,
            prompt_tokens=int(inputs["input_ids"].shape[1]),
            completion_tokens=int(gen.shape[0]),
            seconds=time.time() - t0,
        )

    def train_adapter(self, *, data_dir: Path, out_dir: Path, **hp) -> dict:
        """QLoRA fine-tune with PEFT + transformers Trainer (self-contained, no trl)."""
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import Trainer, TrainingArguments

        out_dir.mkdir(parents=True, exist_ok=True)
        rows = [json.loads(l) for l in open(data_dir / "train.jsonl")]

        def tokenize(row):
            full = row["prompt"] + " " + row["completion"] + self._tokenizer.eos_token
            enc = self._tokenizer(full, truncation=True, max_length=1024, padding="max_length")
            # Mask the prompt tokens so loss is computed only over the completion.
            p_len = len(self._tokenizer(row["prompt"])["input_ids"])
            labels = list(enc["input_ids"])
            for i in range(min(p_len, len(labels))):
                labels[i] = -100
            labels = [-100 if t == self._tokenizer.pad_token_id else t for t in labels]
            enc["labels"] = labels
            return enc

        ds = [tokenize(r) for r in rows]

        lcfg = LoraConfig(
            r=hp["lora_rank"], lora_alpha=hp["lora_alpha"], lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
        )
        model = get_peft_model(self._base, lcfg)

        args = TrainingArguments(
            output_dir=str(out_dir / "_trainer"),
            per_device_train_batch_size=hp["batch_size"],
            learning_rate=hp["learning_rate"],
            max_steps=hp["max_iters"],
            logging_steps=10, save_strategy="no", report_to=[],
            fp16=torch.cuda.is_available(),
        )
        trainer = Trainer(model=model, args=args, train_dataset=ds)
        result = trainer.train()
        model.save_pretrained(str(out_dir))
        metrics = {"final_loss": result.training_loss, "iters": hp["max_iters"], "early_stopped": False}
        (out_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
        self._model = self._base       # detach training head
        return metrics
