from __future__ import annotations

from backend.config.settings import Settings, get_settings
from backend.embeddings.base import EmbeddingProvider, EmbeddingResult
from backend.embeddings.cache import EmbeddingCache
from backend.embeddings.errors import EmbeddingError
from backend.embeddings.ollama_provider import OllamaEmbeddingProvider
from backend.embeddings.service import EmbeddingService
from backend.embeddings.ollama_client import OllamaClient
from backend.storage.sqlite_store import SQLiteStore


def create_embedding_service(
    storage: SQLiteStore | None = None,
    settings: Settings | None = None,
) -> EmbeddingService:
    cfg = settings or get_settings()
    store = storage or SQLiteStore(cfg.sqlite_path)
    provider: EmbeddingProvider = OllamaEmbeddingProvider(cfg)
    cache = EmbeddingCache(store, enabled=cfg.embedding_cache_enabled)
    return EmbeddingService(provider, cache, cfg)


__all__ = [
    "EmbeddingProvider",
    "EmbeddingResult",
    "EmbeddingCache",
    "EmbeddingError",
    "EmbeddingService",
    "OllamaClient",
    "OllamaEmbeddingProvider",
    "create_embedding_service",
]
