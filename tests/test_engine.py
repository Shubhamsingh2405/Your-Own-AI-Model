def test_engine_loads_demo_vectors():
    from services.engine import create_engine

    engine = create_engine()
    assert engine.vector_db.size() == 20
    assert engine.settings.demo_dims == 16


def test_engine_search():
    from services.engine import create_engine

    engine = create_engine()
    q = engine.vector_db.all()[0].emb
    out = engine.vector_db.search(q, 3, "cosine", "hnsw")
    assert len(out["hits"]) == 3
    assert out["us"] >= 0
