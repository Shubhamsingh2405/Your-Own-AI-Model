from __future__ import annotations

import math
import re
from dataclasses import dataclass, field


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass
class BM25Index:
    k1: float = 1.5
    b: float = 0.75
    doc_lengths: dict[int, int] = field(default_factory=dict)
    doc_freq: dict[str, int] = field(default_factory=dict)
    postings: dict[str, dict[int, int]] = field(default_factory=dict)
    avg_dl: float = 0.0
    n_docs: int = 0

    def clear(self) -> None:
        self.doc_lengths.clear()
        self.doc_freq.clear()
        self.postings.clear()
        self.avg_dl = 0.0
        self.n_docs = 0

    def add_document(self, doc_id: int, text: str) -> None:
        tokens = _tokenize(text)
        self.doc_lengths[doc_id] = len(tokens)
        counts: dict[str, int] = {}
        for tok in tokens:
            counts[tok] = counts.get(tok, 0) + 1
        for term, tf in counts.items():
            self.postings.setdefault(term, {})[doc_id] = tf
        for term in counts:
            self.doc_freq[term] = len(self.postings[term])
        self.n_docs = len(self.doc_lengths)
        self.avg_dl = sum(self.doc_lengths.values()) / self.n_docs if self.n_docs else 0.0

    def rebuild(self, documents: dict[int, str]) -> None:
        self.clear()
        for doc_id, text in documents.items():
            self.add_document(doc_id, text)

    def _idf(self, term: str) -> float:
        df = self.doc_freq.get(term, 0)
        if df == 0 or self.n_docs == 0:
            return 0.0
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def score(self, query: str, doc_id: int) -> float:
        if doc_id not in self.doc_lengths:
            return 0.0
        dl = self.doc_lengths[doc_id]
        total = 0.0
        for term in _tokenize(query):
            if term not in self.postings or doc_id not in self.postings[term]:
                continue
            tf = self.postings[term][doc_id]
            idf = self._idf(term)
            denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avg_dl or 1))
            total += idf * (tf * (self.k1 + 1)) / denom
        return total

    def search(self, query: str, k: int) -> list[tuple[float, int]]:
        if not self.doc_lengths:
            return []
        scores = [(self.score(query, doc_id), doc_id) for doc_id in self.doc_lengths]
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:k]
