from __future__ import annotations

from backend.config.settings import Settings, get_settings
from backend.embeddings.base import EmbeddingProvider, EmbeddingResult
from backend.embeddings.cache import EmbeddingCache
from backend.embeddings.errors import EmbeddingError


class EmbeddingService:
    """
    High-level embedding API with caching and dimension validation.
    """

    UNAVAILABLE_MSG = (
        "Ollama unavailable. Install from https://ollama.com then run: "
        "ollama pull nomic-embed-text && ollama pull llama3.2"
    )

    def __init__(
        self,
        provider: EmbeddingProvider,
        cache: EmbeddingCache,
        settings: Settings | None = None,
    ) -> None:
        self.provider = provider
        self.cache = cache
        self.settings = settings or get_settings()
        self._locked_dimension: int | None = (
            self.settings.embedding_dimension
            if self.settings.embedding_dimension > 0
            else None
        )

    @property
    def model_name(self) -> str:
        return self.provider.model_name

    def is_available(self) -> bool:
        return self.provider.is_available()

    def get_dimension(self) -> int:
        return self._locked_dimension or 0

    def lock_dimension(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if self._locked_dimension is None:
            self._locked_dimension = dimension
        elif self._locked_dimension != dimension:
            raise EmbeddingError(
                f"embedding dimension mismatch: expected {self._locked_dimension}, got {dimension}"
            )

    def embed(self, text: str, expected_dim: int | None = None) -> EmbeddingResult:
        if not text.strip():
            raise EmbeddingError("cannot embed empty text")

        target_dim = expected_dim or self._locked_dimension
        cached = self.cache.get(self.provider.model_name, text)
        if cached is not None:
            if target_dim and len(cached) != target_dim:
                raise EmbeddingError(
                    f"cached embedding dimension mismatch: expected {target_dim}, got {len(cached)}"
                )
            if self._locked_dimension is None:
                self._locked_dimension = len(cached)
            return EmbeddingResult(
                vector=cached,
                model=self.provider.model_name,
                dimension=len(cached),
                cached=True,
            )

        vector = self.provider.embed_text(text)
        if not vector:
            raise EmbeddingError(self.UNAVAILABLE_MSG)

        if target_dim and len(vector) != target_dim:
            raise EmbeddingError(
                f"embedding dimension mismatch: expected {target_dim}, got {len(vector)}"
            )

        if self._locked_dimension is None:
            self._locked_dimension = len(vector)
        elif len(vector) != self._locked_dimension:
            raise EmbeddingError(
                f"embedding dimension mismatch: expected {self._locked_dimension}, got {len(vector)}"
            )

        self.cache.put(self.provider.model_name, text, vector)
        return EmbeddingResult(
            vector=vector,
            model=self.provider.model_name,
            dimension=len(vector),
            cached=False,
        )

    def sync_dimension_from_index(self, index_dim: int) -> None:
        if index_dim > 0:
            self.lock_dimension(index_dim)

    def stats(self) -> dict:
        return {
            "model": self.provider.model_name,
            "dimension": self.get_dimension(),
            "configuredDimension": self.settings.embedding_dimension,
            "cache": self.cache.stats(),
        }
