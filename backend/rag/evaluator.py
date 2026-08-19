from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalCase:
    question: str
    expected_answer: str
    relevant_document: str
    relevant_chunk: str


def load_eval_dataset(path: Path) -> list[EvalCase]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        EvalCase(
            question=row["question"],
            expected_answer=row.get("expected_answer", ""),
            relevant_document=row.get("relevant_document", ""),
            relevant_chunk=row.get("relevant_chunk", ""),
        )
        for row in raw
    ]


def token_overlap(a: str, b: str) -> float:
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def evaluate_retrieval(hits: list, case: EvalCase, k: int) -> dict:
    top = hits[:k]
    matched = any(
        case.relevant_document.lower() in hit.item.title.lower()
        or case.relevant_chunk.lower() in hit.item.text.lower()
        for hit in top
    )
    return {"recallAtK": 1.0 if matched else 0.0}


def evaluate_answer(answer: str, case: EvalCase) -> dict:
    overlap = token_overlap(answer, case.expected_answer) if case.expected_answer else 0.0
    return {"answerOverlap": round(overlap, 4)}
