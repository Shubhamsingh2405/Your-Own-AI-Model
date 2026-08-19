from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingResult:
    vector: list[float]
    model: str
    dimension: int
    cached: bool = False


class EmbeddingProvider(ABC):
    """Abstract embedding backend (Ollama, etc.)."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Return embedding vector or empty list on failure."""
        raise NotImplementedError
