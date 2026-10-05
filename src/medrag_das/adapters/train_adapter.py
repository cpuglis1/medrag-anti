"""Train ONE QLoRA adapter on a single MedMCQA category (paper §3.4.2).

Per-category recipe from the paper:
  * data: MedMCQA rows filtered to one `subject_name`, capped at 10,000 samples
  * QLoRA on the 8-bit Llama-2-7B base
  * 4 LoRA layers (not the default 16) to cut back-prop memory
  * batch size 2
  * early stopping when TRAIN loss plateaus — observed 10–60 iters across categories

Unspecified in the paper (see configs/default.yaml, marked PAPER UNSPECIFIED): LoRA rank,
alpha, learning rate, exact plateau criterion.

STUB: data prep contract is written; the fit() call delegates to the backend's QLoRA trainer.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..backends.base import LLMBackend
from ..generation.prompts import build_prompt


def prepare_category_data(
    medmcqa_split,
    medmcqa_label: str,
    out_dir: Path,
    max_samples: int,
    valid_frac: float = 0.1,
) -> Path:
    """Filter MedMCQA to one subject, format as prompt/completion pairs.

    Writes BOTH train.jsonl and valid.jsonl (mlx_lm.lora requires a validation set). MedMCQA
    fields: question, opa, opb, opc, opd, cop (0-based correct index), subject_name, exp
    (explanation). The completion mirrors the required JSON output schema so the adapter learns
    the format the evaluation expects.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[str] = []
    for row in medmcqa_split:
        if row.get("subject_name") != medmcqa_label:
            continue
        options = [row["opa"], row["opb"], row["opc"], row["opd"]]
        cop_idx0 = int(row["cop"])                     # 0-based in MedMCQA
        prompt = build_prompt(row["question"], options, context=None)
        completion = json.dumps({
            "cop": options[cop_idx0],
            "cop index": cop_idx0 + 1,                 # 1-based in our prompt contract
            "why correct": row.get("exp") or "",
        })
        records.append(json.dumps({"prompt": prompt, "completion": completion}))
        if len(records) >= max_samples:
            break
    if not records:
        raise ValueError(f"no MedMCQA rows for subject {medmcqa_label!r}")

    # Hold out a small validation slice (at least 1 row) for mlx_lm's eval loop.
    n_val = max(1, int(len(records) * valid_frac)) if len(records) > 1 else 0
    valid, train = records[:n_val], records[n_val:] or records
    (out_dir / "train.jsonl").write_text("\n".join(train) + "\n")
    if valid:
        (out_dir / "valid.jsonl").write_text("\n".join(valid) + "\n")
    return out_dir / "train.jsonl"


def train_one(
    backend: LLMBackend,
    *,
    data_dir: Path,
    out_dir: Path,
    lora_layers: int,
    lora_rank: int,
    lora_alpha: int,
    batch_size: int,
    learning_rate: float,
    max_iters: int,
    early_stop: dict,
) -> dict:
    """Fit a single adapter. Returns metrics (final loss, iters run, early-stopped?)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    return backend.train_adapter(
        data_dir=data_dir,
        out_dir=out_dir,
        lora_layers=lora_layers,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_iters=max_iters,
        early_stop=early_stop,
    )
