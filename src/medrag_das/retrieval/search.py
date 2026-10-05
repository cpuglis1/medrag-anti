"""Semantic search over the FAISS KB — the paper's two-stage retrieval (§3.3.2).

The default FAISS IndexFlatL2 returns nearest neighbors by L2 distance, but nearest is not
necessarily *relevant*. The paper adds a cosine-similarity gate first:

    S = { x_i in {x_1..x_n} : cos(x, x_i) >= t }          # relevance gate
    M = min(|S|, 3)
    return argmin-top-M ||x - x_i||  over x_i in S         # closest within the gated set

So: filter by cosine >= t, then take up to 3 of those by smallest L2 distance. If S is
empty, no context is returned (and the generation falls back to the no-context prompt).

This module implements the math directly on numpy arrays so it is testable without a live
FAISS index. `faiss_search` wires it to an on-disk index.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Retrieved:
    idx: int          # row index into the KB
    l2: float         # L2 distance to the query
    cosine: float     # cosine similarity to the query


def cosine_threshold_topk(
    query: np.ndarray,
    kb: np.ndarray,
    threshold: float,
    top_k: int = 3,
) -> list[Retrieved]:
    """Reference implementation of the paper's S / argmin-top-M search.

    Args:
        query: (d,) query embedding.
        kb:    (n, d) matrix of KB embeddings.
        threshold: cosine-similarity gate `t`.
        top_k: cap on returned items (paper: 3).

    Returns:
        Up to min(|S|, top_k) items with cosine >= threshold, sorted by ascending L2.
    """
    query = np.asarray(query, dtype=np.float32).ravel()
    kb = np.asarray(kb, dtype=np.float32)
    if kb.ndim != 2 or kb.shape[1] != query.shape[0]:
        raise ValueError(f"shape mismatch: query {query.shape}, kb {kb.shape}")

    q_norm = np.linalg.norm(query) + 1e-12
    kb_norms = np.linalg.norm(kb, axis=1) + 1e-12
    cos = (kb @ query) / (kb_norms * q_norm)

    gated = np.where(cos >= threshold)[0]           # the set S
    if gated.size == 0:
        return []

    diffs = kb[gated] - query
    l2 = np.linalg.norm(diffs, axis=1)
    order = np.argsort(l2)[: min(top_k, gated.size)]  # argmin-top-M by L2

    return [Retrieved(idx=int(gated[i]), l2=float(l2[i]), cosine=float(cos[gated[i]])) for i in order]


def faiss_search(query: np.ndarray, index, kb_matrix: np.ndarray, threshold: float, top_k: int = 3):
    """Same search backed by an on-disk FAISS index for the L2 stage.

    Strategy: over-fetch a candidate pool with FAISS (fast exact L2), then apply the cosine
    gate on the candidates and keep up to `top_k`. Over-fetch because the cosine gate can
    reject near-but-irrelevant neighbors; pull enough that `top_k` can still be filled.
    """
    query = np.asarray(query, dtype=np.float32).reshape(1, -1)
    pool = max(top_k * 10, 50)
    _, idxs = index.search(query, pool)                 # (1, pool) nearest by L2
    cand = idxs[0][idxs[0] >= 0]
    if cand.size == 0:
        return []
    results = cosine_threshold_topk(query.ravel(), kb_matrix[cand], threshold, top_k)
    # Map local candidate positions back to global KB indices.
    return [Retrieved(idx=int(cand[r.idx]), l2=r.l2, cosine=r.cosine) for r in results]
