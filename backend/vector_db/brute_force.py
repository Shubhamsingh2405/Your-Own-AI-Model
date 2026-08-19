from __future__ import annotations

from backend.vector_db.distance import DistFn
from backend.vector_db.models import VectorItem


class BruteForce:
    def __init__(self) -> None:
        self.items: list[VectorItem] = []

    def insert(self, item: VectorItem) -> None:
        self.items.append(item)

    def knn(self, q: list[float], k: int, dist: DistFn) -> list[tuple[float, int]]:
        results = [(dist(q, v.emb), v.id) for v in self.items]
        results.sort()
        return results[:k]

    def remove(self, item_id: int) -> None:
        self.items = [v for v in self.items if v.id != item_id]
