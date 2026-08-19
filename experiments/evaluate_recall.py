#!/usr/bin/env python3
"""Run Recall@K evaluation on the demo VectorDB."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services.engine import create_engine  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Recall@K for HNSW vs brute force")
    parser.add_argument("--queries", type=int, default=10)
    parser.add_argument("--metric", default="cosine", choices=["cosine", "euclidean", "manhattan"])
    parser.add_argument("--ef", type=int, default=None)
    parser.add_argument("--k", default="1,5,10")
    args = parser.parse_args()

    k_values = sorted({int(x.strip()) for x in args.k.split(",") if x.strip()})
    engine = create_engine()
    queries = engine.vector_db.sample_query_embeddings(args.queries)
    result = engine.vector_db.evaluate_recall_batch(
        queries, k_values, args.metric, ef_search=args.ef
    )

    print("=== Recall@K Evaluation ===")
    print(f"Queries: {result['queryCount']} | Dataset: {result['datasetSize']} vectors")
    for row in result["results"]:
        recall_str = "  ".join(f"R@{k}={row['recall'][k]:.2%}" for k in sorted(row["recall"], key=int))
        print(f"{row['algorithm']:10}  latency={row['latencyUs']} us  {recall_str}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
