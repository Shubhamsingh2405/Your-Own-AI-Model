#!/usr/bin/env python3
"""Advanced algorithm benchmark with latency percentiles and recall."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.config.settings import get_settings  # noqa: E402
from backend.vector_db.vector_db import VectorDB  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="100,500,1000")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--queries", type=int, default=10)
    parser.add_argument("--ef", default="50,100")
    args = parser.parse_args()

    sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]
    ef_values = [int(x.strip()) for x in args.ef.split(",") if x.strip()]
    settings = get_settings()
    db = VectorDB(settings.demo_dims, settings)
    result = db.advanced_benchmark(sizes, args.k, args.metric, args.queries, ef_values)

    print("=== Advanced Benchmark ===")
    for row in result["sizes"]:
        print(f"\nDataset size: {row['datasetSize']} | build: {row['buildTimeUs']} us")
        for algo, stats in row["algorithms"].items():
            lat = stats["latency"]
            recall = stats.get("recall", {})
            r_str = " ".join(f"R@{k}={v:.2%}" for k, v in recall.items())
            print(f"  {algo:14} p50={lat['p50Us']}us p95={lat['p95Us']}us p99={lat['p99Us']}us  {r_str}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
