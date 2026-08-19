from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RerankInput:
    query: str
    doc_id: int
    title: str
    text: str
    vector_score: float
    bm25_score: float
    hybrid_score: float


@dataclass
class RerankResult:
    doc_id: int
    score: float
    vector_score: float
    bm25_score: float
    hybrid_score: float
    lexical_overlap: float


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: list[RerankInput], top_k: int) -> list[RerankResult]:
        raise NotImplementedError


class LexicalReranker(Reranker):
    """Lightweight reranker using lexical overlap + retrieval scores."""

    def __init__(self, vector_weight: float = 0.4, bm25_weight: float = 0.3, overlap_weight: float = 0.3) -> None:
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.overlap_weight = overlap_weight

    @staticmethod
    def _overlap(query: str, text: str) -> float:
        q = set(re.findall(r"[a-z0-9]+", query.lower()))
        t = set(re.findall(r"[a-z0-9]+", text.lower()))
        if not q:
            return 0.0
        return len(q & t) / len(q)

    def rerank(self, query: str, candidates: list[RerankInput], top_k: int) -> list[RerankResult]:
        results: list[RerankResult] = []
        for c in candidates:
            overlap = self._overlap(query, c.title + " " + c.text)
            score = (
                self.vector_weight * (1.0 - min(c.vector_score, 1.0))
                + self.bm25_weight * c.bm25_score
                + self.overlap_weight * overlap
            )
            results.append(
                RerankResult(
                    doc_id=c.doc_id,
                    score=score,
                    vector_score=c.vector_score,
                    bm25_score=c.bm25_score,
                    hybrid_score=c.hybrid_score,
                    lexical_overlap=overlap,
                )
            )
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
