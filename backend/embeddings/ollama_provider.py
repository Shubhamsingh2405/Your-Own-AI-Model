from __future__ import annotations

import requests

from backend.config.settings import Settings, get_settings
from backend.embeddings.base import EmbeddingProvider


class OllamaEmbeddingProvider(EmbeddingProvider):
    def __init__(self, settings: Settings | None = None) -> None:
        cfg = settings or get_settings()
        self._base = cfg.ollama_url.rstrip("/")
        self._model = cfg.embedding_model

    @property
    def model_name(self) -> str:
        return self._model

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self._base}/api/tags", timeout=2)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def embed_text(self, text: str) -> list[float]:
        try:
            r = requests.post(
                f"{self._base}/api/embeddings",
                json={"model": self._model, "prompt": text},
                timeout=30,
            )
            if r.status_code != 200:
                return []
            data = r.json()
            emb = data.get("embedding")
            if not isinstance(emb, list) or not emb:
                return []
            return [float(x) for x in emb]
        except requests.RequestException:
            return []
