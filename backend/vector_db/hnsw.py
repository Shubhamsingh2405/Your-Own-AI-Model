from __future__ import annotations

import heapq
import math
import random
import time
from dataclasses import dataclass, field

from backend.vector_db.distance import DistFn
from backend.vector_db.models import VectorItem


@dataclass
class HNSWNode:
    item: VectorItem
    max_lyr: int
    nbrs: list[list[int]] = field(default_factory=list)


@dataclass
class GraphInfo:
    top_layer: int
    node_count: int
    nodes_per_layer: list[int]
    edges_per_layer: list[int]
    nodes: list[dict]
    edges: list[dict]


@dataclass
class HNSWStats:
    """Extended HNSW index statistics for monitoring and tuning."""

    node_count: int
    layer_count: int
    top_layer: int
    total_edges: int
    avg_degree: float
    max_degree: int
    avg_max_level: float
    entry_point: int
    m: int
    m0: int
    ef_construction: int
    ef_search: int
    seed: int
    build_time_us: int
    last_search_us: int
    search_count: int
    nodes_per_layer: list[int]
    edges_per_layer: list[int]


class HNSW:
    def __init__(
        self,
        m: int = 16,
        ef_build: int = 200,
        ef_search: int = 50,
        seed: int = 42,
    ) -> None:
        if m < 2:
            raise ValueError("HNSW M must be >= 2")
        self.M = m
        self.M0 = 2 * m
        self.ef_build = ef_build
        self.ef_search = ef_search
        self.seed = seed
        self.mL = 1.0 / math.log(m)
        self.rng = random.Random(seed)
        self.G: dict[int, HNSWNode] = {}
        self.top_layer = -1
        self.entry_pt = -1
        self.build_time_us = 0
        self.last_search_us = 0
        self.search_count = 0

    def _rand_level(self) -> int:
        return int(math.floor(-math.log(self.rng.random()) * self.mL))

    def _search_layer(
        self, q: list[float], ep: int, ef: int, layer: int, dist: DistFn
    ) -> list[tuple[float, int]]:
        vis: set[int] = set()
        cands: list[tuple[float, int]] = []
        found: list[tuple[float, int]] = []

        d0 = dist(q, self.G[ep].item.emb)
        vis.add(ep)
        heapq.heappush(cands, (d0, ep))
        heapq.heappush(found, (-d0, ep))

        while cands:
            cd, cid = heapq.heappop(cands)
            if len(found) >= ef and cd > -found[0][0]:
                break
            if layer >= len(self.G[cid].nbrs):
                continue
            for nid in self.G[cid].nbrs[layer]:
                if nid in vis or nid not in self.G:
                    continue
                vis.add(nid)
                nd = dist(q, self.G[nid].item.emb)
                if len(found) < ef or nd < -found[0][0]:
                    heapq.heappush(cands, (nd, nid))
                    heapq.heappush(found, (-nd, nid))
                    if len(found) > ef:
                        heapq.heappop(found)

        results = [(-d, nid) for d, nid in found]
        results.sort()
        return results

    @staticmethod
    def _select_nbrs(cands: list[tuple[float, int]], max_m: int) -> list[int]:
        return [nid for _, nid in cands[:max_m]]

    def insert(self, item: VectorItem, dist: DistFn) -> None:
        t0 = time.perf_counter()
        item_id = item.id
        lvl = self._rand_level()
        self.G[item_id] = HNSWNode(item=item, max_lyr=lvl, nbrs=[[] for _ in range(lvl + 1)])

        if self.entry_pt == -1:
            self.entry_pt = item_id
            self.top_layer = lvl
            self.build_time_us += int((time.perf_counter() - t0) * 1_000_000)
            return

        ep = self.entry_pt
        for lc in range(self.top_layer, lvl, -1):
            if lc < len(self.G[ep].nbrs):
                w = self._search_layer(item.emb, ep, 1, lc, dist)
                if w:
                    ep = w[0][1]

        for lc in range(min(self.top_layer, lvl), -1, -1):
            w = self._search_layer(item.emb, ep, self.ef_build, lc, dist)
            max_m = self.M0 if lc == 0 else self.M
            sel = self._select_nbrs(w, max_m)
            self.G[item_id].nbrs[lc] = sel

            for nid in sel:
                if nid not in self.G:
                    continue
                if len(self.G[nid].nbrs) <= lc:
                    self.G[nid].nbrs.extend([[] for _ in range(lc + 1 - len(self.G[nid].nbrs))])
                conn = self.G[nid].nbrs[lc]
                conn.append(item_id)
                if len(conn) > max_m:
                    ds = [
                        (dist(self.G[nid].item.emb, self.G[c].item.emb), c)
                        for c in conn
                        if c in self.G
                    ]
                    ds.sort()
                    self.G[nid].nbrs[lc] = [c for _, c in ds[:max_m]]

            if w:
                ep = w[0][1]

        if lvl > self.top_layer:
            self.top_layer = lvl
            self.entry_pt = item_id

        self.build_time_us += int((time.perf_counter() - t0) * 1_000_000)

    def knn(
        self,
        q: list[float],
        k: int,
        ef: int | None,
        dist: DistFn,
    ) -> list[tuple[float, int]]:
        t0 = time.perf_counter()
        ef_val = self.ef_search if ef is None else ef
        if self.entry_pt == -1:
            self.last_search_us = int((time.perf_counter() - t0) * 1_000_000)
            self.search_count += 1
            return []
        ep = self.entry_pt
        for lc in range(self.top_layer, 0, -1):
            if lc < len(self.G[ep].nbrs):
                w = self._search_layer(q, ep, 1, lc, dist)
                if w:
                    ep = w[0][1]
        w = self._search_layer(q, ep, max(ef_val, k), 0, dist)
        result = w[:k]
        self.last_search_us = int((time.perf_counter() - t0) * 1_000_000)
        self.search_count += 1
        return result

    def remove(self, item_id: int) -> None:
        if item_id not in self.G:
            return
        for nd in self.G.values():
            for layer in nd.nbrs:
                if item_id in layer:
                    layer.remove(item_id)
        if self.entry_pt == item_id:
            self.entry_pt = -1
            for nid in self.G:
                if nid != item_id:
                    self.entry_pt = nid
                    break
        del self.G[item_id]

    def _layer_counts(self) -> tuple[list[int], list[int]]:
        max_l = max(self.top_layer + 1, 1)
        nodes_per_layer = [0] * max_l
        edges_per_layer = [0] * max_l

        for item_id, nd in self.G.items():
            for lc in range(min(nd.max_lyr, max_l - 1) + 1):
                nodes_per_layer[lc] += 1
                if lc < len(nd.nbrs):
                    for nid in nd.nbrs[lc]:
                        if item_id < nid:
                            edges_per_layer[lc] += 1

        return nodes_per_layer, edges_per_layer

    def get_info(self) -> GraphInfo:
        nodes_per_layer, edges_per_layer = self._layer_counts()
        nodes: list[dict] = []
        edges: list[dict] = []

        for item_id, nd in self.G.items():
            nodes.append(
                {
                    "id": item_id,
                    "metadata": nd.item.metadata,
                    "category": nd.item.category,
                    "maxLyr": nd.max_lyr,
                }
            )
            for lc in range(len(nd.nbrs)):
                for nid in nd.nbrs[lc]:
                    if item_id < nid:
                        edges.append({"src": item_id, "dst": nid, "lyr": lc})

        return GraphInfo(
            top_layer=self.top_layer,
            node_count=len(self.G),
            nodes_per_layer=nodes_per_layer,
            edges_per_layer=edges_per_layer,
            nodes=nodes,
            edges=edges,
        )

    def get_stats(self) -> HNSWStats:
        nodes_per_layer, edges_per_layer = self._layer_counts()
        total_edges = sum(edges_per_layer)

        layer_count = self.top_layer + 1 if self.top_layer >= 0 else 0

        if not self.G:
            avg_degree = 0.0
            max_degree = 0
            avg_max_level = 0.0
        else:
            degrees = [len(nd.nbrs[0]) if nd.nbrs else 0 for nd in self.G.values()]
            avg_degree = sum(degrees) / len(degrees)
            max_degree = max(degrees) if degrees else 0
            avg_max_level = sum(nd.max_lyr for nd in self.G.values()) / len(self.G)

        layer_count = self.top_layer + 1 if self.top_layer >= 0 else 0

        return HNSWStats(
            node_count=len(self.G),
            layer_count=layer_count,
            top_layer=self.top_layer,
            total_edges=total_edges,
            avg_degree=round(avg_degree, 2),
            max_degree=max_degree,
            avg_max_level=round(avg_max_level, 2),
            entry_point=self.entry_pt,
            m=self.M,
            m0=self.M0,
            ef_construction=self.ef_build,
            ef_search=self.ef_search,
            seed=self.seed,
            build_time_us=self.build_time_us,
            last_search_us=self.last_search_us,
            search_count=self.search_count,
            nodes_per_layer=nodes_per_layer,
            edges_per_layer=edges_per_layer,
        )

    def size(self) -> int:
        return len(self.G)


def create_hnsw_from_settings(settings) -> HNSW:
    """Build an HNSW index using application settings."""
    return HNSW(
        m=settings.hnsw_m,
        ef_build=settings.hnsw_ef_construction,
        ef_search=settings.hnsw_ef_search,
        seed=settings.hnsw_seed,
    )
