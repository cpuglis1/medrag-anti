"""LLM backends. `mlx` is primary (Apple Silicon, matches the paper); `torch` is a stub."""

from .base import GenerationResult, LLMBackend


def get_backend(name: str, **kwargs) -> LLMBackend:
    """Factory: return a backend instance by name ('mlx' | 'torch')."""
    if name == "mlx":
        from .mlx_backend import MLXBackend

        return MLXBackend(**kwargs)
    if name == "torch":
        from .torch_backend import TorchBackend

        return TorchBackend(**kwargs)
    raise ValueError(f"Unknown backend {name!r}. Expected 'mlx' or 'torch'.")


__all__ = ["LLMBackend", "GenerationResult", "get_backend"]
