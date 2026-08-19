from __future__ import annotations

import threading
import time

from backend.benchmarks.latency import latency_summary
from backend.benchmarks.recall import RecallResult, average_recall, recall_at_k
from backend.benchmarks.synthetic import generate_vectors
from backend.config.settings import Settings, get_settings
from backend.vector_db.brute_force import BruteForce
from backend.vector_db.distance import DistFn, get_dist_fn
from backend.vector_db.hnsw import GraphInfo, HNSW, HNSWStats, create_hnsw_from_settings
from backend.vector_db.kdtree import KDTree
from backend.vector_db.models import VectorItem


class VectorDB:
    def __init__(self, dims: int, settings: Settings | None = None) -> None:
        cfg = settings or get_settings()
        self.dims = dims
        self.settings = cfg
        self.store: dict[int, VectorItem] = {}
        self.bf = BruteForce()
        self.kdt = KDTree(dims)
        self.hnsw = create_hnsw_from_settings(cfg)
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
        self,
        q: list[float],
        k: int,
        metric: str,
        algo: str,
        ef_search: int | None = None,
    ) -> dict:
        with self.mu:
            dfn = get_dist_fn(metric)
            t0 = time.perf_counter()
            if algo == "bruteforce":
                raw = self.bf.knn(q, k, dfn)
            elif algo == "kdtree":
                raw = self.kdt.knn(q, k, dfn)
            else:
                raw = self.hnsw.knn(q, k, ef_search, dfn)
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
            return {
                "hits": hits,
                "us": us,
                "algo": algo,
                "metric": metric,
                "efSearch": ef_search or self.hnsw.ef_search,
            }

    def benchmark(
        self,
        q: list[float],
        k: int,
        metric: str,
        ef_search: int | None = None,
    ) -> dict:
        with self.mu:
            dfn = get_dist_fn(metric)
            ef = ef_search or self.hnsw.ef_search

            def timed(fn) -> int:
                t0 = time.perf_counter()
                fn()
                return int((time.perf_counter() - t0) * 1_000_000)

            return {
                "bfUs": timed(lambda: self.bf.knn(q, k, dfn)),
                "kdUs": timed(lambda: self.kdt.knn(q, k, dfn)),
                "hnswUs": timed(lambda: self.hnsw.knn(q, k, ef, dfn)),
                "n": len(self.store),
                "efSearch": ef,
            }

    def evaluate_recall(
        self,
        q: list[float],
        k_values: list[int],
        metric: str,
        ef_search: int | None = None,
    ) -> dict:
        """Compare approximate search recall against brute-force ground truth."""
        with self.mu:
            if not self.store:
                return {
                    "datasetSize": 0,
                    "metric": metric,
                    "kValues": k_values,
                    "groundTruthAlgorithm": "bruteforce",
                    "results": [],
                }

            dfn = get_dist_fn(metric)
            ef = ef_search or self.hnsw.ef_search
            max_k = min(max(k_values), len(self.store))
            capped_k = [k for k in k_values if k > 0]

            t0 = time.perf_counter()
            bf_raw = self.bf.knn(q, max_k, dfn)
            bf_us = int((time.perf_counter() - t0) * 1_000_000)
            gt_ids = [item_id for _, item_id in bf_raw]

            def measure(name: str, raw: list[tuple[float, int]], latency_us: int) -> RecallResult:
                ids = [item_id for _, item_id in raw]
                recall = {
                    str(k): round(recall_at_k(gt_ids, ids, min(k, max_k)), 4)
                    for k in capped_k
                }
                return RecallResult(
                    algorithm=name,
                    latency_us=latency_us,
                    recall=recall,
                    top_k_ids=ids[:max_k],
                )

            t0 = time.perf_counter()
            hnsw_raw = self.hnsw.knn(q, max_k, ef, dfn)
            hnsw_us = int((time.perf_counter() - t0) * 1_000_000)

            t0 = time.perf_counter()
            kd_raw = self.kdt.knn(q, max_k, dfn)
            kd_us = int((time.perf_counter() - t0) * 1_000_000)

            results = [
                measure("bruteforce", bf_raw, bf_us),
                measure("hnsw", hnsw_raw, hnsw_us),
                measure("kdtree", kd_raw, kd_us),
            ]

            return {
                "datasetSize": len(self.store),
                "metric": metric,
                "efSearch": ef,
                "kValues": capped_k,
                "maxKUsed": max_k,
                "groundTruthAlgorithm": "bruteforce",
                "groundTruthIds": gt_ids,
                "results": [
                    {
                        "algorithm": r.algorithm,
                        "latencyUs": r.latency_us,
                        "recall": r.recall,
                        "topKIds": r.top_k_ids,
                    }
                    for r in results
                ],
            }

    def evaluate_recall_batch(
        self,
        queries: list[list[float]],
        k_values: list[int],
        metric: str,
        ef_search: int | None = None,
    ) -> dict:
        if not queries:
            return {
                "queryCount": 0,
                "datasetSize": self.size(),
                "metric": metric,
                "kValues": k_values,
                "groundTruthAlgorithm": "bruteforce",
                "results": [],
            }

        per_algo: dict[str, dict[str, list[float]]] = {}
        latencies: dict[str, list[int]] = {}

        for q in queries:
            one = self.evaluate_recall(q, k_values, metric, ef_search=ef_search)
            for row in one["results"]:
                algo = row["algorithm"]
                per_algo.setdefault(algo, {})
                latencies.setdefault(algo, [])
                latencies[algo].append(row["latencyUs"])
                for k, value in row["recall"].items():
                    per_algo[algo].setdefault(k, []).append(value)

        ef = ef_search or self.hnsw.ef_search
        results = []
        for algo, recall_lists in per_algo.items():
            avg = average_recall(recall_lists)
            lats = latencies[algo]
            results.append(
                {
                    "algorithm": algo,
                    "latencyUs": int(sum(lats) / len(lats)),
                    "latencyUsP50": sorted(lats)[len(lats) // 2],
                    "recall": avg,
                }
            )

        return {
            "queryCount": len(queries),
            "datasetSize": len(self.store),
            "metric": metric,
            "efSearch": ef,
            "kValues": k_values,
            "groundTruthAlgorithm": "bruteforce",
            "results": results,
        }

    def sample_query_embeddings(self, count: int) -> list[list[float]]:
        with self.mu:
            items = list(self.store.values())
            if not items:
                return []
            step = max(1, len(items) // count)
            picked = items[::step][:count]
            return [item.emb for item in picked]

    def all(self) -> list[VectorItem]:
        with self.mu:
            return list(self.store.values())

    def hnsw_info(self) -> GraphInfo:
        with self.mu:
            return self.hnsw.get_info()

    def hnsw_stats(self) -> HNSWStats:
        with self.mu:
            return self.hnsw.get_stats()

    def size(self) -> int:
        with self.mu:
            return len(self.store)

    def advanced_benchmark(
        self,
        sizes: list[int],
        k: int,
        metric: str,
        query_count: int = 10,
        ef_values: list[int] | None = None,
    ) -> dict:
        """Benchmark algorithms on synthetic datasets at multiple sizes."""
        ef_values = ef_values or [self.settings.hnsw_ef_search]
        dfn = get_dist_fn(metric)
        results = []

        for n in sizes:
            if n <= 0:
                continue
            vectors = generate_vectors(n, self.dims, seed=self.settings.hnsw_seed)
            queries = generate_vectors(min(query_count, n), self.dims, seed=self.settings.hnsw_seed + 1)

            bf = BruteForce()
            kdt = KDTree(self.dims)
            hnsw = create_hnsw_from_settings(self.settings)

            t0 = time.perf_counter()
            for i, emb in enumerate(vectors):
                item = VectorItem(i + 1, f"synthetic-{i+1}", "synthetic", emb)
                bf.insert(item)
                kdt.insert(item)
                hnsw.insert(item, dfn)
            build_us = int((time.perf_counter() - t0) * 1_000_000)

            size_result = {
                "datasetSize": n,
                "dimensions": self.dims,
                "buildTimeUs": build_us,
                "algorithms": {},
            }

            for algo_name, search_fn in [
                ("bruteforce", lambda q, ef: bf.knn(q, k, dfn)),
                ("kdtree", lambda q, ef: kdt.knn(q, k, dfn)),
            ]:
                lats: list[int] = []
                for q in queries:
                    t0 = time.perf_counter()
                    search_fn(q, self.settings.hnsw_ef_search)
                    lats.append(int((time.perf_counter() - t0) * 1_000_000))
                size_result["algorithms"][algo_name] = {
                    "latency": latency_summary(lats),
                    "recall": {str(k): 1.0},
                }

            for ef in ef_values:
                lats = []
                recalls: list[float] = []
                for q in queries:
                    t0 = time.perf_counter()
                    approx = hnsw.knn(q, k, ef, dfn)
                    lats.append(int((time.perf_counter() - t0) * 1_000_000))
                    gt = bf.knn(q, k, dfn)
                    gt_ids = [i for _, i in gt]
                    approx_ids = [i for _, i in approx]
                    recalls.append(recall_at_k(gt_ids, approx_ids, k))
                size_result["algorithms"][f"hnsw_ef{ef}"] = {
                    "efSearch": ef,
                    "latency": latency_summary(lats),
                    "recall": {str(k): round(sum(recalls) / len(recalls), 4) if recalls else 0.0},
                }

            results.append(size_result)

        return {"metric": metric, "k": k, "queryCount": query_count, "sizes": results}
