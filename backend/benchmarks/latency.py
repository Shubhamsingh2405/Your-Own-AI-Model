from __future__ import annotations


def percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (p / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return int(ordered[low] * (1 - weight) + ordered[high] * weight)


def latency_summary(values: list[int]) -> dict:
    if not values:
        return {"count": 0, "minUs": 0, "maxUs": 0, "avgUs": 0, "p50Us": 0, "p95Us": 0, "p99Us": 0}
    return {
        "count": len(values),
        "minUs": min(values),
        "maxUs": max(values),
        "avgUs": int(sum(values) / len(values)),
        "p50Us": percentile(values, 50),
        "p95Us": percentile(values, 95),
        "p99Us": percentile(values, 99),
    }
