from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VectorItem:
    id: int
    metadata: str
    category: str
    emb: list[float]


@dataclass
class DocItem:
    id: int
    title: str
    text: str
    emb: list[float]
    document_id: int = 0
    chunk_index: int = 0
    start_position: int = 0
    end_position: int = 0
    embed_model: str = ""
