"""BioMedLM embeddings for KB abstracts and queries (paper §3.3.1).

Procedure the paper describes:
  1. tokenize each abstract with `stanford-crfm/BioMedLM`
  2. process in batches of 32; split long text into chunks of 1024 words
  3. embed each chunk, then take the MEAN of a chunk's token/hidden embeddings
  4. pad to a consistent length and store the per-abstract vector
The SAME model embeds the query at search time (§3.3.2).

STUB: the batching/chunking/pooling control flow is written out; the actual forward pass is
left to implement (and to decide which hidden state to pool — BioMedLM is a GPT-2-style
decoder, so last-hidden-state mean-pooling over non-pad tokens is the natural choice).
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

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # from transformers import AutoModel, AutoTokenizer
        # self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id)
        # self._model = AutoModel.from_pretrained(self.cfg.model_id).eval()
        raise NotImplementedError(
            "BioMedLMEmbedder: load stanford-crfm/BioMedLM via transformers (eval mode)."
        )

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a list of chunks -> (len(texts), d). Mean-pool last hidden state over tokens."""
        self._ensure_loaded()
        raise NotImplementedError(
            "BioMedLMEmbedder.embed_texts: tokenize (batch), forward pass, mean-pool last "
            "hidden state over non-pad tokens, L2-stage happens later. Return float32 (n, d)."
        )

    def embed_abstract(self, abstract: str) -> np.ndarray:
        """Embed one abstract: chunk -> embed chunks -> mean across chunks (paper §3.3.1)."""
        chunks = chunk_by_words(abstract, self.cfg.chunk_words)
        if not chunks:
            raise ValueError("empty abstract")
        chunk_vecs = self.embed_texts(chunks)          # (num_chunks, d)
        return chunk_vecs.mean(axis=0).astype(np.float32)

    def embed_query(self, question: str) -> np.ndarray:
        """Embed a query with the same model/pooling used for the KB."""
        return self.embed_abstract(question)
