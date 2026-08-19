from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class StoredChunk:
    chunk_id: int
    document_id: int
    chunk_index: int
    title: str
    text: str
    start_position: int
    end_position: int
    embedding: list[float]
    embed_model: str
    embed_dim: int


class SQLiteStore:
    """Persistent document, chunk, and embedding storage."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        title TEXT NOT NULL,
                        source TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL,
                        metadata TEXT NOT NULL DEFAULT '{}'
                    );

                    CREATE TABLE IF NOT EXISTS chunks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        document_id INTEGER NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        title TEXT NOT NULL,
                        text TEXT NOT NULL,
                        start_position INTEGER NOT NULL,
                        end_position INTEGER NOT NULL,
                        FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                    );

                    CREATE TABLE IF NOT EXISTS embeddings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        chunk_id INTEGER NOT NULL UNIQUE,
                        model TEXT NOT NULL,
                        dimension INTEGER NOT NULL,
                        vector_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
                    );

                    CREATE INDEX IF NOT EXISTS idx_chunks_document_id
                        ON chunks(document_id);

                    CREATE TABLE IF NOT EXISTS embedding_cache (
                        cache_key TEXT PRIMARY KEY,
                        model TEXT NOT NULL,
                        dimension INTEGER NOT NULL,
                        vector_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        last_used_at TEXT NOT NULL
                    );
                    """
                )
                conn.commit()
            finally:
                conn.close()

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _encode_vector(vector: list[float]) -> str:
        return json.dumps(vector)

    @staticmethod
    def _decode_vector(raw: str) -> list[float]:
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("invalid embedding payload")
        return [float(x) for x in data]

    def insert_document(
        self,
        title: str,
        source: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """
                    INSERT INTO documents (title, source, created_at, metadata)
                    VALUES (?, ?, ?, ?)
                    """,
                    (title, source, self._utc_now(), json.dumps(metadata or {})),
                )
                conn.commit()
                return int(cur.lastrowid)
            finally:
                conn.close()

    def insert_document_with_chunks(
        self,
        document_title: str,
        chunks: list[dict[str, Any]],
        embed_model: str,
        source: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> tuple[int, list[int]]:
        """
        Insert one document and all chunks atomically.

        Each chunk dict must include:
        title, text, start_position, end_position, chunk_index, embedding
        """
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN")
                cur = conn.execute(
                    """
                    INSERT INTO documents (title, source, created_at, metadata)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        document_title,
                        source,
                        self._utc_now(),
                        json.dumps(metadata or {}),
                    ),
                )
                document_id = int(cur.lastrowid)
                chunk_ids: list[int] = []
                created_at = self._utc_now()

                for chunk in chunks:
                    emb = chunk["embedding"]
                    cur = conn.execute(
                        """
                        INSERT INTO chunks (
                            document_id, chunk_index, title, text,
                            start_position, end_position
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            document_id,
                            chunk["chunk_index"],
                            chunk["title"],
                            chunk["text"],
                            chunk["start_position"],
                            chunk["end_position"],
                        ),
                    )
                    chunk_id = int(cur.lastrowid)
                    conn.execute(
                        """
                        INSERT INTO embeddings (
                            chunk_id, model, dimension, vector_json, created_at
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            chunk_id,
                            embed_model,
                            len(emb),
                            self._encode_vector(emb),
                            created_at,
                        ),
                    )
                    chunk_ids.append(chunk_id)

                conn.commit()
                return document_id, chunk_ids
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def delete_chunk(self, chunk_id: int) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    def load_all_chunks(self) -> list[StoredChunk]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT
                        c.id AS chunk_id,
                        c.document_id,
                        c.chunk_index,
                        c.title,
                        c.text,
                        c.start_position,
                        c.end_position,
                        e.model AS embed_model,
                        e.dimension AS embed_dim,
                        e.vector_json
                    FROM chunks c
                    INNER JOIN embeddings e ON e.chunk_id = c.id
                    ORDER BY c.id ASC
                    """
                ).fetchall()
                return [
                    StoredChunk(
                        chunk_id=int(row["chunk_id"]),
                        document_id=int(row["document_id"]),
                        chunk_index=int(row["chunk_index"]),
                        title=row["title"],
                        text=row["text"],
                        start_position=int(row["start_position"]),
                        end_position=int(row["end_position"]),
                        embed_model=row["embed_model"],
                        embed_dim=int(row["embed_dim"]),
                        embedding=self._decode_vector(row["vector_json"]),
                    )
                    for row in rows
                ]
            finally:
                conn.close()

    def count_documents(self) -> int:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()
                return int(row["n"]) if row else 0
            finally:
                conn.close()

    def count_chunks(self) -> int:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
                return int(row["n"]) if row else 0
            finally:
                conn.close()

    def get_embedding_cache(self, cache_key: str) -> list[float] | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT vector_json FROM embedding_cache WHERE cache_key = ?
                    """,
                    (cache_key,),
                ).fetchone()
                if not row:
                    return None
                conn.execute(
                    """
                    UPDATE embedding_cache
                    SET last_used_at = ?
                    WHERE cache_key = ?
                    """,
                    (self._utc_now(), cache_key),
                )
                conn.commit()
                return self._decode_vector(row["vector_json"])
            finally:
                conn.close()

    def put_embedding_cache(
        self,
        cache_key: str,
        model: str,
        dimension: int,
        vector: list[float],
    ) -> None:
        now = self._utc_now()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO embedding_cache (
                        cache_key, model, dimension, vector_json, created_at, last_used_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(cache_key) DO UPDATE SET
                        model = excluded.model,
                        dimension = excluded.dimension,
                        vector_json = excluded.vector_json,
                        last_used_at = excluded.last_used_at
                    """,
                    (
                        cache_key,
                        model,
                        dimension,
                        self._encode_vector(vector),
                        now,
                        now,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def count_embedding_cache(self) -> int:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM embedding_cache"
                ).fetchone()
                return int(row["n"]) if row else 0
            finally:
                conn.close()
