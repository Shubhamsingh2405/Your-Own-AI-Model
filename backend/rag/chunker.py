from __future__ import annotations

import re
from dataclasses import dataclass

from backend.config.settings import Settings, get_settings


@dataclass
class TextChunk:
    text: str
    start_position: int
    end_position: int
    chunk_index: int


def _word_spans(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def chunk_text_with_positions(
    text: str,
    chunk_words: int | None = None,
    overlap_words: int | None = None,
    settings: Settings | None = None,
) -> list[TextChunk]:
    cfg = settings or get_settings()
    size = chunk_words if chunk_words is not None else cfg.chunk_size
    overlap = overlap_words if overlap_words is not None else cfg.chunk_overlap

    spans = _word_spans(text)
    if not spans:
        return []
    if len(spans) <= size:
        return [
            TextChunk(
                text=text.strip(),
                start_position=spans[0][1],
                end_position=spans[-1][2],
                chunk_index=0,
            )
        ]

    chunks: list[TextChunk] = []
    step = max(size - overlap, 1)
    chunk_index = 0
    for i in range(0, len(spans), step):
        end = min(i + size, len(spans))
        part = spans[i:end]
        chunk_text_value = " ".join(word for word, _, _ in part)
        chunks.append(
            TextChunk(
                text=chunk_text_value,
                start_position=part[0][1],
                end_position=part[-1][2],
                chunk_index=chunk_index,
            )
        )
        chunk_index += 1
        if end == len(spans):
            break
    return chunks


def chunk_text(
    text: str,
    chunk_words: int | None = None,
    overlap_words: int | None = None,
    settings: Settings | None = None,
) -> list[str]:
    return [
        chunk.text
        for chunk in chunk_text_with_positions(
            text,
            chunk_words=chunk_words,
            overlap_words=overlap_words,
            settings=settings,
        )
    ]
