from __future__ import annotations

import requests

from backend.config.settings import Settings, get_settings


class OllamaClient:
    """LLM generation client (embeddings go through EmbeddingService)."""

    def __init__(self, settings: Settings | None = None) -> None:
        cfg = settings or get_settings()
        self.base = cfg.ollama_url.rstrip("/")
        self.embed_model = cfg.embedding_model
        self.gen_model = cfg.gen_model

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base}/api/tags", timeout=2)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def generate(self, prompt: str) -> str:
        try:
            r = requests.post(
                f"{self.base}/api/generate",
                json={"model": self.gen_model, "prompt": prompt, "stream": False},
                timeout=180,
            )
            if r.status_code != 200:
                return "ERROR: Ollama unavailable. Run: ollama serve"
            return r.json().get("response", "")
        except requests.RequestException:
            return "ERROR: Ollama unavailable. Run: ollama serve"
