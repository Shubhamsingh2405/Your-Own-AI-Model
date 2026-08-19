from backend.benchmarks.recall import recall_at_k
from backend.vector_db.distance import cosine, euclidean, manhattan


def test_cosine_identical():
    assert cosine([1, 0], [1, 0]) == 0.0


def test_euclidean_identical():
    assert euclidean([1, 0], [1, 0]) == 0.0


def test_manhattan_identical():
    assert manhattan([1, 0], [1, 0]) == 0.0


def test_recall_at_k_perfect():
    assert recall_at_k([1, 2, 3], [1, 2, 3], 3) == 1.0


def test_recall_at_k_partial():
    assert recall_at_k([1, 2, 3], [1, 9, 8], 3) == 1 / 3
