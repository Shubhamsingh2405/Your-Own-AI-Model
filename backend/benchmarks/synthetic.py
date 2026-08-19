from __future__ import annotations

import random


def generate_vectors(count: int, dims: int, seed: int = 42) -> list[list[float]]:
    rng = random.Random(seed)
    vectors: list[list[float]] = []
    for _ in range(count):
        vec = [rng.random() for _ in range(dims)]
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        vectors.append([x / norm for x in vec])
    return vectors
