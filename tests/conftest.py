import tempfile
from pathlib import Path

import pytest

from backend.config.settings import Settings
from backend.embeddings.base import EmbeddingProvider, EmbeddingResult
from backend.vector_db.vector_db import VectorDB


class FakeEmbedProvider(EmbeddingProvider):
    @property
    def model_name(self) -> str:
        return "fake-embed"

    def is_available(self) -> bool:
        return True

    def embed_text(self, text: str) -> list[float]:
        n = len(text)
        return [0.01 * ((n + i) % 17) for i in range(8)]


@pytest.fixture
def temp_settings():
    td = tempfile.mkdtemp()
    return Settings(
        base_dir=Path("."),
        demo_dims=8,
        ollama_host="127.0.0.1",
        ollama_port=11434,
        ollama_url="http://127.0.0.1:11434",
        embedding_model="fake-embed",
        embedding_dimension=0,
        embedding_cache_enabled=True,
        gen_model="fake-gen",
        flask_host="127.0.0.1",
        flask_port=8080,
        hnsw_m=8,
        hnsw_ef_construction=50,
        hnsw_ef_search=25,
        hnsw_seed=42,
        chunk_size=100,
        chunk_overlap=10,
        doc_max_dist=0.9,
        hybrid_alpha=0.7,
        rag_min_similarity=0.25,
        sqlite_path=Path(td) / "test.sqlite",
    )


@pytest.fixture
def vector_db(temp_settings):
    db = VectorDB(temp_settings.demo_dims, temp_settings)
    from backend.vector_db.distance import get_dist_fn

    dist = get_dist_fn("cosine")
    for i in range(30):
        emb = [0.01 * ((i + j) % 11) for j in range(temp_settings.demo_dims)]
        db.insert(f"item-{i}", "test", emb, dist)
    return db
