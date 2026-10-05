"""BioMedLM embeddings for KB abstracts and queries (paper §3.3.1).

Procedure the paper describes:
  1. tokenize each abstract with `stanford-crfm/BioMedLM`
  2. process in batches of 32; split long text into chunks of 1024 words
  3. embed each chunk, then take the MEAN of a chunk's token embeddings
  4. pad to a consistent length and store the per-abstract vector
The SAME model embeds the query at search time (§3.3.2).

BioMedLM is a GPT-2-style decoder, so we mean-pool the last hidden state over non-pad tokens
(a standard sentence-embedding choice for decoder LMs). Runs on MPS (Apple) or CUDA if
available, else CPU.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EmbedConfig:
    model_id: str = "stanford-crfm/BioMedLM"
    batch_size: int = 32
    chunk_words: int = 1024
    pool: str = "mean"
    max_tokens: int = 1024


def chunk_by_words(text: str, chunk_words: int) -> list[str]:
    """Split text into <= chunk_words word chunks (paper processes 1024-word chunks)."""
    words = text.split()
    if not words:
        return []
    return [" ".join(words[i : i + chunk_words]) for i in range(0, len(words), chunk_words)]


class BioMedLMEmbedder:
    def __init__(self, cfg: EmbedConfig | None = None) -> None:
        self.cfg = cfg or EmbedConfig()
        self._model = None
        self._tokenizer = None
        self._device = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        dtype = torch.float16 if self._device in ("cuda", "mps") else torch.float32
        self._model = AutoModel.from_pretrained(self.cfg.model_id, torch_dtype=dtype).to(self._device).eval()

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed chunks -> (len(texts), d). Mean-pool last hidden state over non-pad tokens."""
        self._ensure_loaded()
        import torch

        out: list[np.ndarray] = []
        bs = self.cfg.batch_size
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            enc = self._tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True,
                max_length=self.cfg.max_tokens,
            ).to(self._device)
            with torch.no_grad():
                hidden = self._model(**enc).last_hidden_state          # (b, t, d)
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)  # (b, t, 1)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            pooled = (summed / counts).float().cpu().numpy()           # (b, d)
            out.append(pooled)
        return np.vstack(out).astype(np.float32)

    def embed_abstract(self, abstract: str) -> np.ndarray:
        """Embed one abstract: chunk -> embed chunks -> mean across chunks (paper §3.3.1)."""
        chunks = chunk_by_words(abstract, self.cfg.chunk_words)
        if not chunks:
            raise ValueError("empty abstract")
        chunk_vecs = self.embed_texts(chunks)                          # (num_chunks, d)
        return chunk_vecs.mean(axis=0).astype(np.float32)

    def embed_query(self, question: str) -> np.ndarray:
        """Embed a query with the same model/pooling used for the KB."""
        return self.embed_abstract(question)
