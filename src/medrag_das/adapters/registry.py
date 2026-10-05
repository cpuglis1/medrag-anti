"""Adapter registry — map category slug -> adapter directory, and drive swap/fuse.

Thin layer over the backend's `fuse_adapter`. Enforces the one-adapter-at-a-time memory
rule (WARNINGS.md §7) by tracking the currently-fused slug and no-op'ing redundant swaps.
"""
from __future__ import annotations

from pathlib import Path

from ..backends.base import LLMBackend


class AdapterRegistry:
    def __init__(self, backend: LLMBackend, adapters_dir: Path, valid_slugs: list[str]) -> None:
        self._backend = backend
        self._dir = Path(adapters_dir)
        self._valid = set(valid_slugs)
        self._current: str | None = None

    def path_for(self, slug: str) -> Path:
        if slug not in self._valid:
            raise KeyError(f"unknown adapter slug {slug!r}")
        return self._dir / slug

    def available(self) -> list[str]:
        """Slugs that actually have a trained adapter on disk."""
        return sorted(s for s in self._valid if (self._dir / s).exists())

    def swap_to(self, slug: str) -> None:
        """Fuse the adapter for `slug`, releasing the previously fused one. Idempotent."""
        if slug == self._current:
            return
        path = self.path_for(slug)
        if not path.exists():
            raise FileNotFoundError(
                f"adapter for {slug!r} not found at {path} — train it first (`medrag train`)."
            )
        self._backend.fuse_adapter(path)   # backend releases the old adapter
        self._current = slug

    def unfuse(self) -> None:
        self._backend.fuse_adapter(None)
        self._current = None
