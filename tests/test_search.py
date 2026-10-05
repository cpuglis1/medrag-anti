import numpy as np

from medrag_das.retrieval.search import cosine_threshold_topk


def test_filters_by_cosine_then_ranks_by_l2():
    query = np.array([1.0, 0.0], dtype=np.float32)
    kb = np.array(
        [
            [10.0, 0.0],   # same direction (cos=1) but far in L2
            [1.0, 0.0],    # identical direction, closest in L2
            [0.0, 1.0],    # orthogonal (cos=0) -> filtered out at t=0.5
            [2.0, 0.1],    # near direction, close-ish
        ],
        dtype=np.float32,
    )
    res = cosine_threshold_topk(query, kb, threshold=0.5, top_k=3)
    idxs = [r.idx for r in res]
    assert 2 not in idxs                 # orthogonal row gated out by cosine
    assert idxs[0] == 1                  # identical vector is nearest by L2
    assert len(res) == 3                 # min(|S|=3, top_k=3)


def test_empty_when_nothing_passes_threshold():
    query = np.array([1.0, 0.0], dtype=np.float32)
    kb = np.array([[0.0, 1.0], [0.0, 2.0]], dtype=np.float32)  # all orthogonal
    assert cosine_threshold_topk(query, kb, threshold=0.5) == []


def test_caps_at_top_k():
    query = np.array([1.0, 0.0], dtype=np.float32)
    kb = np.array([[1.0, 0.0]] * 10, dtype=np.float32)
    assert len(cosine_threshold_topk(query, kb, threshold=0.5, top_k=3)) == 3
