from __future__ import annotations

from backend.rag.retriever import RetrievalHit


ABSTAIN_MESSAGE = (
    "I couldn't find sufficient information in the indexed documents to answer this confidently."
)


def build_rag_prompt(question: str, hits: list[RetrievalHit]) -> str:
    ctx_parts = []
    for i, hit in enumerate(hits):
        ctx_parts.append(f"[{i + 1}] {hit.item.title}:\n{hit.item.text}\n")
    ctx = "\n".join(ctx_parts)
    return (
        "You are a helpful assistant. Answer ONLY using the provided document excerpts. "
        "If the excerpts do not contain enough information, reply exactly: "
        f"\"{ABSTAIN_MESSAGE}\"\n"
        "Do not invent facts or cite sources not present in the excerpts.\n\n"
        f"Document excerpts:\n{ctx}\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def format_citations(hits: list[RetrievalHit]) -> list[dict]:
    out = []
    for i, hit in enumerate(hits):
        similarity = round(1.0 - min(hit.vector_distance, 1.0), 4)
        out.append(
            {
                "index": i + 1,
                "chunkId": hit.item.id,
                "documentId": hit.item.document_id,
                "title": hit.item.title,
                "chunkIndex": hit.item.chunk_index,
                "similarity": similarity,
                "vectorDistance": round(hit.vector_distance, 5),
                "bm25Score": round(hit.bm25_score, 4),
                "hybridScore": round(hit.hybrid_score, 4),
                "rerankScore": round(hit.rerank_score, 4) if hit.rerank_score is not None else None,
            }
        )
    return out


def should_abstain(hits: list[RetrievalHit], min_similarity: float) -> bool:
    if not hits:
        return True
    best = min(h.vector_distance for h in hits)
    return (1.0 - min(best, 1.0)) < min_similarity
