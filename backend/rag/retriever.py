from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from backend.retrieval.bm25 import BM25Index
from backend.retrieval.hybrid import fuse_hybrid
from backend.retrieval.reranker import LexicalReranker, RerankInput, RerankResult

if TYPE_CHECKING:
    from backend.embeddings.service import EmbeddingService
    from backend.vector_db.document_db import DocumentDB
    from backend.vector_db.models import DocItem


@dataclass
class RetrievalHit:
    item: DocItem
    vector_distance: float
    vector_score: float
    bm25_score: float
    hybrid_score: float
    rerank_score: float | None = None
    lexical_overlap: float | None = None


class DocumentRetriever:
    def __init__(self, doc_db: DocumentDB, alpha: float = 0.7) -> None:
        self.doc_db = doc_db
        self.alpha = alpha
        self.reranker = LexicalReranker()

    def retrieve(
        self,
        question: str,
        embedding_service: EmbeddingService,
        k: int = 5,
        mode: str = "vector",
        alpha: float | None = None,
        rerank: bool = False,
        rerank_pool: int = 20,
    ) -> list[RetrievalHit]:
        alpha_val = self.alpha if alpha is None else alpha
        q_emb = self.doc_db.embed_query(question, embedding_service)

        if mode == "vector":
            raw = self.doc_db.search(q_emb, k)
            return [
                RetrievalHit(
                    item=item,
                    vector_distance=dist,
                    vector_score=dist,
                    bm25_score=0.0,
                    hybrid_score=1.0 - min(dist, 1.0),
                )
                for dist, item in raw
            ]

        pool_k = max(rerank_pool, k) if rerank else k
        vec_hits = self.doc_db.search(q_emb, pool_k, max_dist=1.0)
        bm25_hits = self.doc_db.bm25_search(question, pool_k)

        vec_scores = [(dist, item.id) for dist, item in vec_hits]
        fused = fuse_hybrid(vec_scores, bm25_hits, alpha_val)

        hits: list[RetrievalHit] = []
        for _, doc_id, breakdown in fused[:pool_k]:
            item = self.doc_db.get_item(doc_id)
            if not item:
                continue
            dist = next((d for d, i in vec_hits if i.id == doc_id), 1.0)
            hits.append(
                RetrievalHit(
                    item=item,
                    vector_distance=dist,
                    vector_score=dist,
                    bm25_score=breakdown["bm25"],
                    hybrid_score=breakdown["hybrid"],
                )
            )

        if rerank and hits:
            candidates = [
                RerankInput(
                    query=question,
                    doc_id=h.item.id,
                    title=h.item.title,
                    text=h.item.text,
                    vector_score=h.vector_score,
                    bm25_score=h.bm25_score,
                    hybrid_score=h.hybrid_score,
                )
                for h in hits
            ]
            reranked: list[RerankResult] = self.reranker.rerank(question, candidates, k)
            id_to_hit = {h.item.id: h for h in hits}
            out: list[RetrievalHit] = []
            for rr in reranked:
                base = id_to_hit[rr.doc_id]
                out.append(
                    RetrievalHit(
                        item=base.item,
                        vector_distance=base.vector_distance,
                        vector_score=base.vector_score,
                        bm25_score=base.bm25_score,
                        hybrid_score=base.hybrid_score,
                        rerank_score=rr.score,
                        lexical_overlap=rr.lexical_overlap,
                    )
                )
            return out

        return hits[:k]
