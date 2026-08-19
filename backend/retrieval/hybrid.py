from __future__ import annotations


def normalize_scores(scores: list[tuple[float, int]], higher_is_better: bool) -> dict[int, float]:
    if not scores:
        return {}
    values = [s for s, _ in scores]
    lo, hi = min(values), max(values)
    out: dict[int, float] = {}
    for score, doc_id in scores:
        if hi == lo:
            norm = 1.0
        elif higher_is_better:
            norm = (score - lo) / (hi - lo)
        else:
            norm = (hi - score) / (hi - lo)
        out[doc_id] = norm
    return out


def fuse_hybrid(
    vector_scores: list[tuple[float, int]],
    bm25_scores: list[tuple[float, int]],
    alpha: float,
) -> list[tuple[float, int, dict]]:
    """Hybrid score = alpha * vector_norm + (1-alpha) * bm25_norm."""
    alpha = max(0.0, min(1.0, alpha))
    vec_norm = normalize_scores(vector_scores, higher_is_better=False)
    bm25_norm = normalize_scores(bm25_scores, higher_is_better=True)
    ids = set(vec_norm) | set(bm25_norm)
    fused: list[tuple[float, int, dict]] = []
    for doc_id in ids:
        v = vec_norm.get(doc_id, 0.0)
        b = bm25_norm.get(doc_id, 0.0)
        hybrid = alpha * v + (1 - alpha) * b
        fused.append((hybrid, doc_id, {"vector": v, "bm25": b, "hybrid": hybrid}))
    fused.sort(key=lambda x: x[0], reverse=True)
    return fused
