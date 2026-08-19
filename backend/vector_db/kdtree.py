from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Optional

from backend.vector_db.distance import DistFn
from backend.vector_db.models import VectorItem


@dataclass
class KDNode:
    item: VectorItem
    left: Optional[KDNode] = None
    right: Optional[KDNode] = None


class KDTree:
    def __init__(self, dims: int) -> None:
        self.root: Optional[KDNode] = None
        self.dims = dims

    def _destroy(self, node: Optional[KDNode]) -> None:
        if not node:
            return
        self._destroy(node.left)
        self._destroy(node.right)

    def _insert(self, node: Optional[KDNode], item: VectorItem, depth: int) -> KDNode:
        if not node:
            return KDNode(item)
        axis = depth % self.dims
        if item.emb[axis] < node.item.emb[axis]:
            node.left = self._insert(node.left, item, depth + 1)
        else:
            node.right = self._insert(node.right, item, depth + 1)
        return node

    def _knn(
        self,
        node: Optional[KDNode],
        q: list[float],
        k: int,
        depth: int,
        dist: DistFn,
        heap: list[tuple[float, int]],
    ) -> None:
        if not node:
            return
        dn = dist(q, node.item.emb)
        if len(heap) < k or dn < -heap[0][0]:
            heapq.heappush(heap, (-dn, node.item.id))
            if len(heap) > k:
                heapq.heappop(heap)
        axis = depth % self.dims
        diff = q[axis] - node.item.emb[axis]
        closer = node.left if diff < 0 else node.right
        farther = node.right if diff < 0 else node.left
        self._knn(closer, q, k, depth + 1, dist, heap)
        if len(heap) < k or abs(diff) < -heap[0][0]:
            self._knn(farther, q, k, depth + 1, dist, heap)

    def insert(self, item: VectorItem) -> None:
        self.root = self._insert(self.root, item, 0)

    def knn(self, q: list[float], k: int, dist: DistFn) -> list[tuple[float, int]]:
        heap: list[tuple[float, int]] = []
        self._knn(self.root, q, k, 0, dist, heap)
        results = [(-d, item_id) for d, item_id in heap]
        results.sort()
        return results

    def rebuild(self, items: list[VectorItem]) -> None:
        self._destroy(self.root)
        self.root = None
        for item in items:
            self.insert(item)
