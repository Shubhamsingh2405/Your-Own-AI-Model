#!/usr/bin/env python3
"""Evaluate RAG retrieval on the curated eval dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services.engine import create_engine  # noqa: E402
from backend.rag.evaluator import evaluate_retrieval, load_eval_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--mode", default="hybrid", choices=["vector", "hybrid", "hybrid_rerank"])
    args = parser.parse_args()

    engine = create_engine()
    dataset = load_eval_dataset(engine.settings.base_dir / "data" / "eval" / "rag_eval.json")

    if not dataset:
        print("No eval dataset found.")
        return

    recalls = []
    for case in dataset:
        rerank = args.mode == "hybrid_rerank"
        mode = "hybrid" if rerank else args.mode
        hits = engine.retriever.retrieve(
            case.question,
            engine.embedding_service,
            k=args.k,
            mode=mode,
            rerank=rerank,
        )
        recalls.append(evaluate_retrieval(hits, case, args.k)["recallAtK"])

    avg = sum(recalls) / len(recalls) if recalls else 0
    print(f"=== RAG Evaluation ===\nCases: {len(dataset)} | mode: {args.mode} | Recall@{args.k}: {avg:.2%}")
    print(json.dumps({"retrievalRecallAtK": avg}, indent=2))


if __name__ == "__main__":
    main()
