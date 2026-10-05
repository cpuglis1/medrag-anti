"""Build the FAISS knowledge base from PubMed abstracts (paper §3.3.1).

Source: TensorFlow Datasets `scientific_papers` (config "pubmed"), OpenAccess, >100k
abstracts. Embed each abstract with BioMedLM, store vectors in a FAISS IndexFlatL2, and
keep a sidecar JSONL of metadata (so retrieved rows can be shown as context).

Keep the KB independent of MedMCQA / Med-HALT content to limit leakage (paper §4.1).
Reference footprint from the paper: 1.02 GiB index (from a 2.34 GiB raw dataset), ~23s/query.

STUB for the model-touching parts; the index/metadata plumbing is real.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .embedder import BioMedLMEmbedder, EmbedConfig


def iter_pubmed_abstracts(limit: int):
    """Yield (doc_id, abstract_text) from TF scientific_papers/pubmed.

    Either tensorflow_datasets or HF `datasets` can serve this corpus; pick one and pin it.
    """
    # import tensorflow_datasets as tfds
    # ds = tfds.load("scientific_papers/pubmed", split="train")
    # for i, ex in enumerate(ds.take(limit)):
    #     yield str(i), ex["abstract"].numpy().decode("utf-8")
    raise NotImplementedError(
        "iter_pubmed_abstracts: stream scientific_papers/pubmed abstracts (limit rows)."
    )


def build(
    kb_path: str | Path,
    meta_path: str | Path,
    limit: int = 100_000,
    embed_cfg: EmbedConfig | None = None,
    batch_flush: int = 1_000,
) -> None:
    """Embed `limit` abstracts and write a FAISS IndexFlatL2 + JSONL metadata sidecar."""
    import faiss  # local import so the package imports without faiss present

    kb_path, meta_path = Path(kb_path), Path(meta_path)
    kb_path.parent.mkdir(parents=True, exist_ok=True)

    embedder = BioMedLMEmbedder(embed_cfg)
    index = None
    vectors: list[np.ndarray] = []

    with open(meta_path, "w") as meta_fh:
        for doc_id, abstract in iter_pubmed_abstracts(limit):
            vec = embedder.embed_abstract(abstract)                 # (d,)
            if index is None:
                index = faiss.IndexFlatL2(vec.shape[0])             # exact L2 (paper)
            vectors.append(vec)
            meta_fh.write(json.dumps({"doc_id": doc_id, "text": abstract}) + "\n")

            if len(vectors) >= batch_flush:
                index.add(np.vstack(vectors).astype(np.float32))
                vectors.clear()

        if vectors and index is not None:
            index.add(np.vstack(vectors).astype(np.float32))

    if index is None:
        raise RuntimeError("no abstracts embedded; check iter_pubmed_abstracts().")
    faiss.write_index(index, str(kb_path))


def load_kb(kb_path: str | Path, meta_path: str | Path):
    """Load the FAISS index, the (n, d) vector matrix, and the metadata list.

    cosine gating (search.faiss_search) needs the raw vectors, so reconstruct them from the
    flat index.
    """
    import faiss

    index = faiss.read_index(str(kb_path))
    kb_matrix = index.reconstruct_n(0, index.ntotal).astype(np.float32)
    with open(meta_path) as fh:
        meta = [json.loads(line) for line in fh]
    return index, kb_matrix, meta
