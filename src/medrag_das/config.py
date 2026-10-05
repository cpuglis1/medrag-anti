"""Typed configuration loader.

Loads `configs/default.yaml` (overridable via $MEDRAG_CONFIG) and `configs/categories.yaml`
into validated dataclasses. Keep knobs here, not scattered as literals.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Category:
    slug: str
    medmcqa_label: str


@dataclass
class RetrievalCfg:
    dataset: str
    dataset_config: str
    kb_limit: int
    embed_batch_size: int
    chunk_words: int
    pool: str
    faiss_index: str
    cosine_threshold: float
    top_k: int
    kb_path: str
    meta_path: str


@dataclass
class AdaptersCfg:
    dir: str
    dataset: str
    max_samples_per_category: int
    lora_layers: int
    lora_rank: int
    lora_alpha: int
    batch_size: int
    learning_rate: float
    max_iters: int
    early_stop: dict


@dataclass
class Config:
    backend: str
    models: dict
    retrieval: RetrievalCfg
    adapters: AdaptersCfg
    selection: dict
    generation: dict
    paths: dict
    categories: list[Category] = field(default_factory=list)

    # --- convenience accessors -------------------------------------------------
    @property
    def category_slugs(self) -> list[str]:
        return [c.slug for c in self.categories]

    def label_to_slug(self) -> dict[str, str]:
        return {c.medmcqa_label: c.slug for c in self.categories}

    def adapter_path(self, slug: str) -> Path:
        return REPO_ROOT / self.adapters.dir / slug


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def load_config(config_path: str | os.PathLike | None = None) -> Config:
    """Load and validate configuration. Env $MEDRAG_CONFIG wins if arg is None."""
    cfg_path = Path(config_path or os.environ.get("MEDRAG_CONFIG", REPO_ROOT / "configs/default.yaml"))
    raw = _load_yaml(cfg_path)

    cats_raw = _load_yaml(REPO_ROOT / "configs/categories.yaml")["categories"]
    categories = [Category(**c) for c in cats_raw]
    if len(categories) != 21:
        # The paper trained exactly 21 adapters (Appendix Table 3). Loudly flag drift.
        raise ValueError(f"Expected 21 categories (paper), found {len(categories)}.")

    return Config(
        backend=os.environ.get("MEDRAG_BACKEND", raw["backend"]),
        models=raw["models"],
        retrieval=RetrievalCfg(**raw["retrieval"]),
        adapters=AdaptersCfg(**raw["adapters"]),
        selection=raw["selection"],
        generation=raw["generation"],
        paths=raw["paths"],
        categories=categories,
    )
