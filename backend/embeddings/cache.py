from __future__ import annotations

import hashlib
import threading

from backend.config.settings import Settings, get_settings
from backend.embeddings.base import EmbeddingProvider
from backend.storage.sqlite_store import SQLiteStore


class EmbeddingCache:
    """SQLite-backed cache for text embeddings keyed by model + content hash."""

    def __init__(self, storage: SQLiteStore, enabled: bool = True) -> None:
        self.storage = storage
        self.enabled = enabled
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(model: str, text: str) -> str:
        payload = f"{model}\0{text}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def get(self, model: str, text: str) -> list[float] | None:
        if not self.enabled:
            return None
        key = self.make_key(model, text)
        with self._lock:
            row = self.storage.get_embedding_cache(key)
            if row is None:
                self.misses += 1
                return None
            self.hits += 1
            return row

    def put(self, model: str, text: str, vector: list[float]) -> None:
        if not self.enabled:
            return
        key = self.make_key(model, text)
        with self._lock:
            self.storage.put_embedding_cache(key, model, len(vector), vector)

    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total) if total else 0.0
            return {
                "enabled": self.enabled,
                "hits": self.hits,
                "misses": self.misses,
                "hitRate": round(hit_rate, 4),
                "entries": self.storage.count_embedding_cache(),
            }
