"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    base_dir: Path
    demo_dims: int
    ollama_host: str
    ollama_port: int
    ollama_url: str
    embedding_model: str
    embedding_dimension: int
    embedding_cache_enabled: bool
    gen_model: str
    flask_host: str
    flask_port: int
    hnsw_m: int
    hnsw_ef_construction: int
    hnsw_ef_search: int
    hnsw_seed: int
    chunk_size: int
    chunk_overlap: int
    doc_max_dist: float
    hybrid_alpha: float
    rag_min_similarity: float
    sqlite_path: Path


def get_settings() -> Settings:
    base_dir = Path(__file__).resolve().parent.parent.parent
    host = _env_str("OLLAMA_HOST", "127.0.0.1")
    port = _env_int("OLLAMA_PORT", 11434)
    url = _env_str("OLLAMA_URL", f"http://{host}:{port}")

    return Settings(
        base_dir=base_dir,
        demo_dims=_env_int("DEMO_DIMS", 16),
        ollama_host=host,
        ollama_port=port,
        ollama_url=url,
        embedding_model=_env_str("EMBEDDING_MODEL", "nomic-embed-text"),
        embedding_dimension=_env_int("EMBEDDING_DIMENSION", 0),
        embedding_cache_enabled=_env_str("EMBEDDING_CACHE_ENABLED", "true").lower()
        in ("1", "true", "yes", "on"),
        gen_model=_env_str("LLM_MODEL", _env_str("GEN_MODEL", "llama3.2")),
        flask_host=_env_str("FLASK_HOST", "0.0.0.0"),
        flask_port=_env_int("FLASK_PORT", 8080),
        hnsw_m=_env_int("HNSW_M", 16),
        hnsw_ef_construction=_env_int("HNSW_EF_CONSTRUCTION", 200),
        hnsw_ef_search=_env_int("HNSW_EF_SEARCH", 50),
        hnsw_seed=_env_int("HNSW_SEED", 42),
        chunk_size=_env_int("CHUNK_SIZE", 250),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 30),
        doc_max_dist=float(_env_str("DOC_MAX_DIST", "0.7")),
        hybrid_alpha=float(_env_str("HYBRID_ALPHA", "0.7")),
        rag_min_similarity=float(_env_str("RAG_MIN_SIMILARITY", "0.25")),
        sqlite_path=Path(_env_str("SQLITE_PATH", str(base_dir / "data" / "vectordb.sqlite"))),
    )
