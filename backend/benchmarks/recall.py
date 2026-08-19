"""Recall@K evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RecallResult:
    algorithm: str
    latency_us: int
    recall: dict[str, float]
    top_k_ids: list[int]


def recall_at_k(ground_truth_ids: list[int], approximate_ids: list[int], k: int) -> float:
    """
    Recall@K = |approx_top_k ∩ ground_truth_top_k| / K

    Ground truth is brute-force top-K. K is capped by available results.
    """
    if k <= 0:
        return 1.0
    effective_k = min(k, len(ground_truth_ids))
    if effective_k == 0:
        return 1.0
    gt = set(ground_truth_ids[:effective_k])
    approx = set(approximate_ids[:effective_k])
    return len(gt & approx) / effective_k


def average_recall(recall_by_k: dict[str, list[float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, values in recall_by_k.items():
        out[k] = round(sum(values) / len(values), 4) if values else 0.0
    return out
