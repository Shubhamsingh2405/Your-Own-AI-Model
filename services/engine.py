from __future__ import annotations

from dataclasses import dataclass

from backend.config.settings import Settings, get_settings
from backend.data.demo import load_demo
from backend.embeddings import create_embedding_service
from backend.embeddings.ollama_client import OllamaClient
from backend.embeddings.service import EmbeddingService
from backend.rag.retriever import DocumentRetriever
from backend.vector_db.document_db import DocumentDB
from backend.vector_db.vector_db import VectorDB


@dataclass
class AIEngine:
    """Single entry point for Streamlit and scripts."""

    settings: Settings
    vector_db: VectorDB
    document_db: DocumentDB
    embedding_service: EmbeddingService
    retriever: DocumentRetriever
    ollama: OllamaClient


def create_engine(settings: Settings | None = None) -> AIEngine:
    cfg = settings or get_settings()
    vector_db = VectorDB(cfg.demo_dims, cfg)
    load_demo(vector_db)
    document_db = DocumentDB(cfg)
    embedding_service = create_embedding_service(document_db.storage, cfg)
    embedding_service.sync_dimension_from_index(document_db.get_dims())
    ollama = OllamaClient(cfg)
    retriever = DocumentRetriever(document_db, alpha=cfg.hybrid_alpha)
    return AIEngine(
        settings=cfg,
        vector_db=vector_db,
        document_db=document_db,
        embedding_service=embedding_service,
        retriever=retriever,
        ollama=ollama,
    )
