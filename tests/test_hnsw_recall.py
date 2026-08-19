def test_hnsw_recall_on_demo_db(vector_db):
    queries = vector_db.sample_query_embeddings(10)
    result = vector_db.evaluate_recall_batch(queries, [1, 5, 10], "cosine")
    hnsw = next(r for r in result["results"] if r["algorithm"] == "hnsw")
    assert hnsw["recall"]["10"] >= 0.8


def test_advanced_benchmark_runs(vector_db):
    result = vector_db.advanced_benchmark([50, 100], k=5, metric="cosine", query_count=5)
    assert len(result["sizes"]) == 2
    assert "hnsw_ef25" in result["sizes"][0]["algorithms"] or any(
        k.startswith("hnsw_ef") for k in result["sizes"][0]["algorithms"]
    )
