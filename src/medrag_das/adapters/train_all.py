"""Train all 21 category adapters sequentially (paper §3.4.2).

One base model, 21 small adapters — total extra storage is only a few MiB per adapter, which
is the whole point of DAS vs. fusing 21 full models (WARNINGS.md §7).
"""
from __future__ import annotations

from pathlib import Path

from ..backends.base import LLMBackend
from ..config import Config
from .train_adapter import prepare_category_data, train_one


def train_all(backend: LLMBackend, cfg: Config, medmcqa_split, only: list[str] | None = None) -> dict:
    """Train adapters for all categories (or just `only` slugs). Returns per-slug metrics."""
    a = cfg.adapters
    results: dict[str, dict] = {}

    for cat in cfg.categories:
        if only and cat.slug not in only:
            continue
        out_dir = cfg.adapter_path(cat.slug)
        data_dir = out_dir / "data"
        prepare_category_data(medmcqa_split, cat.medmcqa_label, data_dir, a.max_samples_per_category)
        results[cat.slug] = train_one(
            backend,
            data_dir=data_dir,
            out_dir=out_dir,
            lora_layers=a.lora_layers,
            lora_rank=a.lora_rank,
            lora_alpha=a.lora_alpha,
            batch_size=a.batch_size,
            learning_rate=a.learning_rate,
            max_iters=a.max_iters,
            early_stop=a.early_stop,
        )
    return results
