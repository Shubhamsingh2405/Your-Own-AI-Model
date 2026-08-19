from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from backend.config.settings import Settings, get_settings
from backend.embeddings.errors import EmbeddingError
from backend.rag.chunker import chunk_text_with_positions
from backend.retrieval.bm25 import BM25Index
from backend.retrieval.hybrid import fuse_hybrid
from backend.storage.sqlite_store import SQLiteStore, StoredChunk
from backend.vector_db.brute_force import BruteForce
from backend.vector_db.distance import cosine
from backend.vector_db.hnsw import create_hnsw_from_settings
from backend.vector_db.models import DocItem, VectorItem

if TYPE_CHECKING:
    from backend.embeddings.service import EmbeddingService


class DocumentDB:
    """
    Document retrieval index backed by SQLite persistence.

    SQLite stores documents/chunks/embeddings.
    In-memory HNSW + BruteForce + BM25 provide fast search.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        cfg = settings or get_settings()
        self.settings = cfg
        self.storage = SQLiteStore(cfg.sqlite_path)
        self.store: dict[int, DocItem] = {}
        self.hnsw = create_hnsw_from_settings(cfg)
        self.bf = BruteForce()
        self.bm25 = BM25Index()
        self.mu = threading.Lock()
        self.dims = 0
        self._max_dist = cfg.doc_max_dist
        self._load_from_storage()

    def _stored_to_item(self, row: StoredChunk) -> DocItem:
        return DocItem(
            id=row.chunk_id,
            title=row.title,
            text=row.text,
            emb=row.embedding,
            document_id=row.document_id,
            chunk_index=row.chunk_index,
            start_position=row.start_position,
            end_position=row.end_position,
            embed_model=row.embed_model,
        )

    def _index_item(self, item: DocItem) -> None:
        vi = VectorItem(item.id, item.title, "doc", item.emb)
        self.hnsw.insert(vi, cosine)
        self.bf.insert(vi)

    def _rebuild_indexes(self) -> None:
        self.hnsw = create_hnsw_from_settings(self.settings)
        self.bf = BruteForce()
        self.dims = 0
        for item in sorted(self.store.values(), key=lambda x: x.id):
            if self.dims == 0:
                self.dims = len(item.emb)
            self._index_item(item)
        self.bm25.rebuild({item.id: item.title + " " + item.text for item in self.store.values()})

    def _load_from_storage(self) -> None:
        rows = self.storage.load_all_chunks()
        self.store = {row.chunk_id: self._stored_to_item(row) for row in rows}
        self._rebuild_indexes()

    def get_item(self, chunk_id: int) -> DocItem | None:
        with self.mu:
            return self.store.get(chunk_id)

    def insert_document(
        self,
        title: str,
        text: str,
        embedding_service: EmbeddingService,
        source: str = "",
        metadata: dict | None = None,
    ) -> tuple[int, list[int]]:
        """Chunk, embed, persist, and index a document."""
        segments = chunk_text_with_positions(text, settings=self.settings)
        if not segments:
            raise ValueError("document text is empty")

        embedding_service.sync_dimension_from_index(self.dims)
        prepared: list[dict] = []
        embed_model = embedding_service.model_name

        for segment in segments:
            result = embedding_service.embed(
                segment.text,
                expected_dim=self.dims or None,
            )
            chunk_title = (
                f"{title} [{segment.chunk_index + 1}/{len(segments)}]"
                if len(segments) > 1
                else title
            )
            prepared.append(
                {
                    "title": chunk_title,
                    "text": segment.text,
                    "start_position": segment.start_position,
                    "end_position": segment.end_position,
                    "chunk_index": segment.chunk_index,
                    "embedding": result.vector,
                }
            )

        document_id, chunk_ids = self.storage.insert_document_with_chunks(
            document_title=title,
            chunks=prepared,
            embed_model=embed_model,
            source=source,
            metadata=metadata,
        )

        with self.mu:
            for chunk_id, payload in zip(chunk_ids, prepared):
                item = DocItem(
                    id=chunk_id,
                    title=payload["title"],
                    text=payload["text"],
                    emb=payload["embedding"],
                    document_id=document_id,
                    chunk_index=payload["chunk_index"],
                    start_position=payload["start_position"],
                    end_position=payload["end_position"],
                    embed_model=embed_model,
                )
                self.store[item.id] = item
                if self.dims == 0:
                    self.dims = len(item.emb)
                self._index_item(item)
                self.bm25.add_document(item.id, item.title + " " + item.text)

        embedding_service.sync_dimension_from_index(self.dims)
        return document_id, chunk_ids

    def embed_query(self, text: str, embedding_service: EmbeddingService) -> list[float]:
        embedding_service.sync_dimension_from_index(self.dims)
        result = embedding_service.embed(text, expected_dim=self.dims or None)
        return result.vector

    def search(
        self,
        q: list[float],
        k: int,
        max_dist: float | None = None,
        ef_search: int | None = None,
    ) -> list[tuple[float, DocItem]]:
        threshold = self._max_dist if max_dist is None else max_dist
        with self.mu:
            if not self.store:
                return []
            if self.dims and len(q) != self.dims:
                raise EmbeddingError(
                    f"query embedding dimension mismatch: expected {self.dims}, got {len(q)}"
                )

            raw = (
                self.bf.knn(q, k, cosine)
                if len(self.store) < 10
                else self.hnsw.knn(q, k, ef_search, cosine)
            )
            out: list[tuple[float, DocItem]] = []
            for d, item_id in raw:
                if item_id in self.store and d <= threshold:
                    out.append((d, self.store[item_id]))
            return out

    def bm25_search(self, query: str, k: int) -> list[tuple[float, int]]:
        with self.mu:
            return self.bm25.search(query, k)

    def hybrid_search(
        self,
        q: list[float],
        query_text: str,
        k: int,
        alpha: float,
        max_dist: float | None = None,
        ef_search: int | None = None,
    ) -> list[tuple[float, DocItem, dict]]:
        threshold = self._max_dist if max_dist is None else max_dist
        with self.mu:
            if not self.store:
                return []
            if self.dims and len(q) != self.dims:
                raise EmbeddingError(
                    f"query embedding dimension mismatch: expected {self.dims}, got {len(q)}"
                )

            pool = max(k * 3, 10)
            vec_raw = (
                self.bf.knn(q, pool, cosine)
                if len(self.store) < 10
                else self.hnsw.knn(q, pool, ef_search, cosine)
            )
            vec_scores = [(d, item_id) for d, item_id in vec_raw if item_id in self.store]
            bm25_scores = self.bm25.search(query_text, pool)
            fused = fuse_hybrid(vec_scores, bm25_scores, alpha)

            out: list[tuple[float, DocItem, dict]] = []
            for hybrid_score, item_id, breakdown in fused:
                if item_id not in self.store:
                    continue
                dist = next((d for d, i in vec_scores if i == item_id), 1.0)
                if dist <= threshold:
                    out.append((dist, self.store[item_id], breakdown))
                if len(out) >= k:
                    break
            return out

    def explain_search(
        self,
        q: list[float],
        query_text: str,
        k: int,
        alpha: float,
        ef_search: int | None = None,
    ) -> list[dict]:
        hits = self.hybrid_search(q, query_text, k, alpha, max_dist=1.0, ef_search=ef_search)
        explained = []
        for dist, item, breakdown in hits:
            explained.append(
                {
                    "id": item.id,
                    "title": item.title,
                    "text": item.text,
                    "documentId": item.document_id,
                    "chunkIndex": item.chunk_index,
                    "vectorDistance": round(dist, 5),
                    "vectorScore": round(1.0 - min(dist, 1.0), 4),
                    "bm25Score": round(breakdown["bm25"], 4),
                    "hybridScore": round(breakdown["hybrid"], 4),
                }
            )
        return explained

    def remove(self, item_id: int) -> bool:
        with self.mu:
            if item_id not in self.store:
                return False
            if not self.storage.delete_chunk(item_id):
                return False
            del self.store[item_id]
            self._rebuild_indexes()
            return True

    def all(self) -> list[DocItem]:
        with self.mu:
            return list(self.store.values())

    def size(self) -> int:
        with self.mu:
            return len(self.store)

    def document_count(self) -> int:
        return self.storage.count_documents()

    def get_dims(self) -> int:
        return self.dims

    def hnsw_stats(self):
        with self.mu:
            return self.hnsw.get_stats()

    def get_embed_model(self) -> str:
        for item in self.store.values():
            if item.embed_model:
                return item.embed_model
        return self.settings.embedding_model
