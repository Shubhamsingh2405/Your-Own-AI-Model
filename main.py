"""
VectorDB Engine — HNSW, KD-Tree, Brute Force + RAG via Ollama.
"""

from __future__ import annotations

import heapq
import math
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import requests
from flask import Flask, jsonify, request, send_from_directory

DIMS = 16
BASE_DIR = Path(__file__).resolve().parent

DistFn = Callable[[list[float], list[float]], float]


# ---------------------------------------------------------------------------
# Distance metrics
# ---------------------------------------------------------------------------

def euclidean(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a)
    nb = sum(y * y for y in b)
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - dot / (math.sqrt(na) * math.sqrt(nb))


def manhattan(a: list[float], b: list[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))


def get_dist_fn(metric: str) -> DistFn:
    if metric == "cosine":
        return cosine
    if metric == "manhattan":
        return manhattan
    return euclidean


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class VectorItem:
    id: int
    metadata: str
    category: str
    emb: list[float]


@dataclass
class DocItem:
    id: int
    title: str
    text: str
    emb: list[float]


# ---------------------------------------------------------------------------
# Brute force
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# KD-Tree
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# HNSW
# ---------------------------------------------------------------------------

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


class HNSW:
    def __init__(self, m: int = 16, ef_build: int = 200) -> None:
        self.M = m
        self.M0 = 2 * m
        self.ef_build = ef_build
        self.mL = 1.0 / math.log(m)
        self.rng = random.Random(42)
        self.G: dict[int, HNSWNode] = {}
        self.top_layer = -1
        self.entry_pt = -1

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
        item_id = item.id
        lvl = self._rand_level()
        self.G[item_id] = HNSWNode(item=item, max_lyr=lvl, nbrs=[[] for _ in range(lvl + 1)])

        if self.entry_pt == -1:
            self.entry_pt = item_id
            self.top_layer = lvl
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

    def knn(self, q: list[float], k: int, ef: int, dist: DistFn) -> list[tuple[float, int]]:
        if self.entry_pt == -1:
            return []
        ep = self.entry_pt
        for lc in range(self.top_layer, 0, -1):
            if lc < len(self.G[ep].nbrs):
                w = self._search_layer(q, ep, 1, lc, dist)
                if w:
                    ep = w[0][1]
        w = self._search_layer(q, ep, max(ef, k), 0, dist)
        return w[:k]

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

    def get_info(self) -> GraphInfo:
        max_l = max(self.top_layer + 1, 1)
        nodes_per_layer = [0] * max_l
        edges_per_layer = [0] * max_l
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
            for lc in range(min(nd.max_lyr, max_l - 1) + 1):
                nodes_per_layer[lc] += 1
                if lc < len(nd.nbrs):
                    for nid in nd.nbrs[lc]:
                        if item_id < nid:
                            edges_per_layer[lc] += 1
                            edges.append({"src": item_id, "dst": nid, "lyr": lc})

        return GraphInfo(
            top_layer=self.top_layer,
            node_count=len(self.G),
            nodes_per_layer=nodes_per_layer,
            edges_per_layer=edges_per_layer,
            nodes=nodes,
            edges=edges,
        )

    def size(self) -> int:
        return len(self.G)


# ---------------------------------------------------------------------------
# Vector database (16D demo)
# ---------------------------------------------------------------------------

class VectorDB:
    def __init__(self, dims: int) -> None:
        self.dims = dims
        self.store: dict[int, VectorItem] = {}
        self.bf = BruteForce()
        self.kdt = KDTree(dims)
        self.hnsw = HNSW(16, 200)
        self.mu = threading.Lock()
        self.next_id = 1

    def insert(self, meta: str, cat: str, emb: list[float], dist: DistFn) -> int:
        with self.mu:
            item = VectorItem(self.next_id, meta, cat, emb)
            self.next_id += 1
            self.store[item.id] = item
            self.bf.insert(item)
            self.kdt.insert(item)
            self.hnsw.insert(item, dist)
            return item.id

    def remove(self, item_id: int) -> bool:
        with self.mu:
            if item_id not in self.store:
                return False
            del self.store[item_id]
            self.bf.remove(item_id)
            self.hnsw.remove(item_id)
            self.kdt.rebuild(list(self.store.values()))
            return True

    def search(
        self, q: list[float], k: int, metric: str, algo: str
    ) -> dict:
        with self.mu:
            dfn = get_dist_fn(metric)
            t0 = time.perf_counter()
            if algo == "bruteforce":
                raw = self.bf.knn(q, k, dfn)
            elif algo == "kdtree":
                raw = self.kdt.knn(q, k, dfn)
            else:
                raw = self.hnsw.knn(q, k, 50, dfn)
            us = int((time.perf_counter() - t0) * 1_000_000)

            hits = []
            for d, item_id in raw:
                if item_id in self.store:
                    v = self.store[item_id]
                    hits.append(
                        {
                            "id": item_id,
                            "meta": v.metadata,
                            "cat": v.category,
                            "emb": v.emb,
                            "dist": d,
                        }
                    )
            return {"hits": hits, "us": us, "algo": algo, "metric": metric}

    def benchmark(self, q: list[float], k: int, metric: str) -> dict:
        with self.mu:
            dfn = get_dist_fn(metric)

            def timed(fn) -> int:
                t0 = time.perf_counter()
                fn()
                return int((time.perf_counter() - t0) * 1_000_000)

            return {
                "bfUs": timed(lambda: self.bf.knn(q, k, dfn)),
                "kdUs": timed(lambda: self.kdt.knn(q, k, dfn)),
                "hnswUs": timed(lambda: self.hnsw.knn(q, k, 50, dfn)),
                "n": len(self.store),
            }

    def all(self) -> list[VectorItem]:
        with self.mu:
            return list(self.store.values())

    def hnsw_info(self) -> GraphInfo:
        with self.mu:
            return self.hnsw.get_info()

    def size(self) -> int:
        with self.mu:
            return len(self.store)


# ---------------------------------------------------------------------------
# Document database (Ollama embeddings)
# ---------------------------------------------------------------------------

class DocumentDB:
    def __init__(self) -> None:
        self.store: dict[int, DocItem] = {}
        self.hnsw = HNSW(16, 200)
        self.bf = BruteForce()
        self.mu = threading.Lock()
        self.next_id = 1
        self.dims = 0

    def insert(self, title: str, text: str, emb: list[float]) -> int:
        with self.mu:
            if self.dims == 0:
                self.dims = len(emb)
            item = DocItem(self.next_id, title, text, emb)
            self.next_id += 1
            self.store[item.id] = item
            vi = VectorItem(item.id, title, "doc", emb)
            self.hnsw.insert(vi, cosine)
            self.bf.insert(vi)
            return item.id

    def search(
        self, q: list[float], k: int, max_dist: float = 0.7
    ) -> list[tuple[float, DocItem]]:
        with self.mu:
            if not self.store:
                return []
            raw = (
                self.bf.knn(q, k, cosine)
                if len(self.store) < 10
                else self.hnsw.knn(q, k, 50, cosine)
            )
            out: list[tuple[float, DocItem]] = []
            for d, item_id in raw:
                if item_id in self.store and d <= max_dist:
                    out.append((d, self.store[item_id]))
            return out

    def remove(self, item_id: int) -> bool:
        with self.mu:
            if item_id not in self.store:
                return False
            del self.store[item_id]
            self.hnsw.remove(item_id)
            self.bf.remove(item_id)
            return True

    def all(self) -> list[DocItem]:
        with self.mu:
            return list(self.store.values())

    def size(self) -> int:
        with self.mu:
            return len(self.store)

    def get_dims(self) -> int:
        return self.dims


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_vec(s: str) -> list[float]:
    out: list[float] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except ValueError:
            pass
    return out


def chunk_text(text: str, chunk_words: int = 250, overlap_words: int = 30) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]
    chunks: list[str] = []
    step = chunk_words - overlap_words
    for i in range(0, len(words), step):
        end = min(i + chunk_words, len(words))
        chunks.append(" ".join(words[i:end]))
        if end == len(words):
            break
    return chunks


class OllamaClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 11434) -> None:
        self.base = f"http://{host}:{port}"
        self.embed_model = "nomic-embed-text"
        self.gen_model = "llama3.2"

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base}/api/tags", timeout=2)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def embed(self, text: str) -> list[float]:
        try:
            r = requests.post(
                f"{self.base}/api/embeddings",
                json={"model": self.embed_model, "prompt": text},
                timeout=30,
            )
            if r.status_code != 200:
                return []
            data = r.json()
            emb = data.get("embedding")
            return emb if isinstance(emb, list) else []
        except requests.RequestException:
            return []

    def generate(self, prompt: str) -> str:
        try:
            r = requests.post(
                f"{self.base}/api/generate",
                json={"model": self.gen_model, "prompt": prompt, "stream": False},
                timeout=180,
            )
            if r.status_code != 200:
                return "ERROR: Ollama unavailable. Run: ollama serve"
            return r.json().get("response", "")
        except requests.RequestException:
            return "ERROR: Ollama unavailable. Run: ollama serve"


def load_demo(db: VectorDB) -> None:
    dist = get_dist_fn("cosine")
    demo = [
        ("Linked List: nodes connected by pointers", "cs",
         [0.90, 0.85, 0.72, 0.68, 0.12, 0.08, 0.15, 0.10, 0.05, 0.08, 0.06, 0.09, 0.07, 0.11, 0.08, 0.06]),
        ("Binary Search Tree: O(log n) search and insert", "cs",
         [0.88, 0.82, 0.78, 0.74, 0.15, 0.10, 0.08, 0.12, 0.06, 0.07, 0.08, 0.05, 0.09, 0.06, 0.07, 0.10]),
        ("Dynamic Programming: memoization overlapping subproblems", "cs",
         [0.82, 0.76, 0.88, 0.80, 0.20, 0.18, 0.12, 0.09, 0.07, 0.06, 0.08, 0.07, 0.08, 0.09, 0.06, 0.07]),
        ("Graph BFS and DFS: breadth and depth first traversal", "cs",
         [0.85, 0.80, 0.75, 0.82, 0.18, 0.14, 0.10, 0.08, 0.06, 0.09, 0.07, 0.06, 0.10, 0.08, 0.09, 0.07]),
        ("Hash Table: O(1) lookup with collision chaining", "cs",
         [0.87, 0.78, 0.70, 0.76, 0.13, 0.11, 0.09, 0.14, 0.08, 0.07, 0.06, 0.08, 0.07, 0.10, 0.08, 0.09]),
        ("Calculus: derivatives integrals and limits", "math",
         [0.12, 0.15, 0.18, 0.10, 0.91, 0.86, 0.78, 0.72, 0.08, 0.06, 0.07, 0.09, 0.07, 0.08, 0.06, 0.10]),
        ("Linear Algebra: matrices eigenvalues eigenvectors", "math",
         [0.20, 0.18, 0.15, 0.12, 0.88, 0.90, 0.82, 0.76, 0.09, 0.07, 0.08, 0.06, 0.10, 0.07, 0.08, 0.09]),
        ("Probability: distributions random variables Bayes theorem", "math",
         [0.15, 0.12, 0.20, 0.18, 0.84, 0.80, 0.88, 0.82, 0.07, 0.08, 0.06, 0.10, 0.09, 0.06, 0.09, 0.08]),
        ("Number Theory: primes modular arithmetic RSA cryptography", "math",
         [0.22, 0.16, 0.14, 0.20, 0.80, 0.85, 0.76, 0.90, 0.08, 0.09, 0.07, 0.06, 0.08, 0.10, 0.07, 0.06]),
        ("Combinatorics: permutations combinations generating functions", "math",
         [0.18, 0.20, 0.16, 0.14, 0.86, 0.78, 0.84, 0.80, 0.06, 0.07, 0.09, 0.08, 0.06, 0.09, 0.10, 0.07]),
        ("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
         [0.08, 0.06, 0.09, 0.07, 0.07, 0.08, 0.06, 0.09, 0.90, 0.86, 0.78, 0.72, 0.08, 0.06, 0.09, 0.07]),
        ("Sushi: vinegared rice raw fish and nori rolls", "food",
         [0.06, 0.08, 0.07, 0.09, 0.09, 0.06, 0.08, 0.07, 0.86, 0.90, 0.82, 0.76, 0.07, 0.09, 0.06, 0.08]),
        ("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
         [0.09, 0.07, 0.06, 0.08, 0.08, 0.09, 0.07, 0.06, 0.82, 0.78, 0.90, 0.84, 0.09, 0.07, 0.08, 0.06]),
        ("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
         [0.07, 0.09, 0.08, 0.06, 0.06, 0.07, 0.09, 0.08, 0.78, 0.82, 0.86, 0.90, 0.06, 0.08, 0.07, 0.09]),
        ("Croissant: laminated pastry with buttery flaky layers", "food",
         [0.06, 0.07, 0.10, 0.09, 0.10, 0.06, 0.07, 0.10, 0.85, 0.80, 0.76, 0.82, 0.09, 0.07, 0.10, 0.06]),
        ("Basketball: fast-paced shooting dribbling slam dunks", "sports",
         [0.09, 0.07, 0.08, 0.10, 0.08, 0.09, 0.07, 0.06, 0.08, 0.07, 0.09, 0.06, 0.91, 0.85, 0.78, 0.72]),
        ("Football: tackles touchdowns field goals and strategy", "sports",
         [0.07, 0.09, 0.06, 0.08, 0.09, 0.07, 0.10, 0.08, 0.07, 0.09, 0.08, 0.07, 0.87, 0.89, 0.82, 0.76]),
        ("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
         [0.08, 0.06, 0.09, 0.07, 0.07, 0.08, 0.06, 0.09, 0.09, 0.06, 0.07, 0.08, 0.83, 0.80, 0.88, 0.82]),
        ("Chess: openings endgames tactics strategic board game", "sports",
         [0.25, 0.20, 0.22, 0.18, 0.22, 0.18, 0.20, 0.15, 0.06, 0.08, 0.07, 0.09, 0.80, 0.84, 0.78, 0.90]),
        ("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
         [0.06, 0.08, 0.07, 0.09, 0.08, 0.06, 0.09, 0.07, 0.10, 0.08, 0.06, 0.07, 0.85, 0.82, 0.86, 0.80]),
    ]
    for meta, cat, emb in demo:
        db.insert(meta, cat, emb, dist)


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)
db = VectorDB(DIMS)
doc_db = DocumentDB()
ollama = OllamaClient()
load_demo(db)


@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/search", methods=["GET"])
def search():
    q = parse_vec(request.args.get("v", ""))
    if len(q) != DIMS:
        return jsonify({"error": f"need {DIMS}D vector"}), 400
    k = int(request.args.get("k", 5))
    metric = request.args.get("metric") or "cosine"
    algo = request.args.get("algo") or "hnsw"
    out = db.search(q, k, metric, algo)
    return jsonify(
        {
            "results": [
                {
                    "id": h["id"],
                    "metadata": h["meta"],
                    "category": h["cat"],
                    "distance": h["dist"],
                    "embedding": h["emb"],
                }
                for h in out["hits"]
            ],
            "latencyUs": out["us"],
            "algo": out["algo"],
            "metric": out["metric"],
        }
    )


@app.route("/insert", methods=["POST"])
def insert():
    body = request.get_json(silent=True) or {}
    meta = body.get("metadata", "")
    cat = body.get("category", "")
    emb = body.get("embedding") or []
    if not meta or not emb or len(emb) != DIMS:
        return jsonify({"error": "invalid body"}), 400
    item_id = db.insert(meta, cat, emb, get_dist_fn("cosine"))
    return jsonify({"id": item_id})


@app.route("/delete/<int:item_id>", methods=["DELETE"])
def delete_item(item_id: int):
    ok = db.remove(item_id)
    return jsonify({"ok": ok})


@app.route("/items", methods=["GET"])
def items():
    return jsonify(
        [
            {
                "id": v.id,
                "metadata": v.metadata,
                "category": v.category,
                "embedding": v.emb,
            }
            for v in db.all()
        ]
    )


@app.route("/benchmark", methods=["GET"])
def benchmark():
    q = parse_vec(request.args.get("v", ""))
    if len(q) != DIMS:
        return jsonify({"error": f"need {DIMS}D vector"}), 400
    k = int(request.args.get("k", 5))
    metric = request.args.get("metric") or "cosine"
    b = db.benchmark(q, k, metric)
    return jsonify(
        {
            "bruteforceUs": b["bfUs"],
            "kdtreeUs": b["kdUs"],
            "hnswUs": b["hnswUs"],
            "itemCount": b["n"],
        }
    )


@app.route("/hnsw-info", methods=["GET"])
def hnsw_info():
    gi = db.hnsw_info()
    return jsonify(
        {
            "topLayer": gi.top_layer,
            "nodeCount": gi.node_count,
            "nodesPerLayer": gi.nodes_per_layer,
            "edgesPerLayer": gi.edges_per_layer,
            "nodes": gi.nodes,
            "edges": gi.edges,
        }
    )


@app.route("/doc/insert", methods=["POST"])
def doc_insert():
    body = request.get_json(silent=True) or {}
    title = body.get("title", "").strip()
    text = body.get("text", "").strip()
    if not title or not text:
        return jsonify({"error": "need title and text"}), 400

    chunks = chunk_text(text, 250, 30)
    ids: list[int] = []
    for i, chunk in enumerate(chunks):
        emb = ollama.embed(chunk)
        if not emb:
            return jsonify(
                {
                    "error": (
                        "Ollama unavailable. Install from https://ollama.com then run: "
                        "ollama pull nomic-embed-text && ollama pull llama3.2"
                    )
                }
            ), 503
        chunk_title = (
            f"{title} [{i + 1}/{len(chunks)}]" if len(chunks) > 1 else title
        )
        ids.append(doc_db.insert(chunk_title, chunk, emb))

    return jsonify({"ids": ids, "chunks": len(chunks), "dims": doc_db.get_dims()})


@app.route("/doc/delete/<int:item_id>", methods=["DELETE"])
def doc_delete(item_id: int):
    ok = doc_db.remove(item_id)
    return jsonify({"ok": ok})


@app.route("/doc/list", methods=["GET"])
def doc_list():
    docs = doc_db.all()
    out = []
    for d in docs:
        preview = d.text[:120] + ("…" if len(d.text) > 120 else "")
        out.append(
            {
                "id": d.id,
                "title": d.title,
                "preview": preview,
                "words": len(d.text.split()),
            }
        )
    return jsonify(out)


@app.route("/doc/search", methods=["POST"])
def doc_search():
    body = request.get_json(silent=True) or {}
    question = body.get("question", "").strip()
    k = int(body.get("k", 3))
    if not question:
        return jsonify({"error": "need question"}), 400
    q_emb = ollama.embed(question)
    if not q_emb:
        return jsonify({"error": "Ollama unavailable"}), 503
    hits = doc_db.search(q_emb, k)
    return jsonify(
        {
            "contexts": [
                {"id": item.id, "title": item.title, "distance": dist}
                for dist, item in hits
            ]
        }
    )


@app.route("/doc/ask", methods=["POST"])
def doc_ask():
    body = request.get_json(silent=True) or {}
    question = body.get("question", "").strip()
    k = int(body.get("k", 3))
    if not question:
        return jsonify({"error": "need question"}), 400

    q_emb = ollama.embed(question)
    if not q_emb:
        return jsonify({"error": "Ollama unavailable"}), 503

    hits = doc_db.search(q_emb, k)
    ctx_parts = []
    for i, (dist, item) in enumerate(hits):
        ctx_parts.append(f"[{i + 1}] {item.title}:\n{item.text}\n")
    ctx = "\n".join(ctx_parts)

    prompt = (
        "You are a helpful assistant. Answer the user's question directly. "
        "Use the provided context if it contains relevant information. "
        "If it doesn't, just use your own general knowledge. "
        "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like "
        "'the context doesn't mention'. Just answer the question naturally.\n\n"
        f"Context:\n{ctx}\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    answer = ollama.generate(prompt)

    return jsonify(
        {
            "answer": answer,
            "model": ollama.gen_model,
            "contexts": [
                {
                    "id": item.id,
                    "title": item.title,
                    "text": item.text,
                    "distance": dist,
                }
                for dist, item in hits
            ],
            "docCount": doc_db.size(),
        }
    )


@app.route("/status", methods=["GET"])
def status():
    up = ollama.is_available()
    return jsonify(
        {
            "ollamaAvailable": up,
            "embedModel": ollama.embed_model,
            "genModel": ollama.gen_model,
            "docCount": doc_db.size(),
            "docDims": doc_db.get_dims(),
            "demoDims": DIMS,
            "demoCount": db.size(),
        }
    )


@app.route("/stats", methods=["GET"])
def stats():
    return jsonify(
        {
            "count": db.size(),
            "dims": DIMS,
            "algorithms": ["bruteforce", "kdtree", "hnsw"],
            "metrics": ["euclidean", "cosine", "manhattan"],
        }
    )


@app.route("/", methods=["GET"])
def index():
    return send_from_directory(BASE_DIR, "index.html")


def main() -> None:
    ollama_up = ollama.is_available()
    print("=== VectorDB Engine ===")
    print("http://localhost:8080")
    print(f"{db.size()} demo vectors | {DIMS} dims | HNSW+KD-Tree+BruteForce")
    print(
        "Ollama: "
        + ("ONLINE" if ollama_up else "OFFLINE (install from ollama.com)")
    )
    if ollama_up:
        print(f"  embed model: {ollama.embed_model}  gen model: {ollama.gen_model}")
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)


if __name__ == "__main__":
    main()
