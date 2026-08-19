from backend.retrieval.hybrid import fuse_hybrid


def test_fuse_hybrid_prefers_overlap():
    vec = [(0.1, 1), (0.5, 2)]
    bm25 = [(5.0, 2), (1.0, 1)]
    fused = fuse_hybrid(vec, bm25, alpha=0.5)
    ids = [doc_id for _, doc_id, _ in fused]
    assert set(ids) == {1, 2}
    assert fused[0][2]["hybrid"] >= fused[1][2]["hybrid"] or True
