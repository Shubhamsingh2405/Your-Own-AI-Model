from __future__ import annotations

import streamlit as st

from backend.embeddings.errors import EmbeddingError
from backend.rag.generator import (
    ABSTAIN_MESSAGE,
    build_rag_prompt,
    format_citations,
    should_abstain,
)
from backend.utils.vectors import parse_vec
from services.engine import AIEngine


def _fmt_us(us: int) -> str:
    return f"{us / 1000:.2f} ms" if us >= 1000 else f"{us} µs"


def page_vector_search(engine: AIEngine) -> None:
    st.header("Vector Search")
    st.caption("Search the 16D demo vector index with Brute Force, KD-Tree, or HNSW.")

    items = engine.vector_db.all()
    dims = engine.settings.demo_dims

    demo_labels = {f"#{v.id} {v.metadata[:40]}": v.emb for v in items}
    source = st.radio("Query source", ["Demo item", "Custom vector"], horizontal=True)

    if source == "Demo item":
        if not demo_labels:
            st.warning("Demo index is empty.")
            return
        choice = st.selectbox("Pick a demo vector", list(demo_labels.keys()))
        q = demo_labels[choice]
    else:
        raw = st.text_input(
            f"Comma-separated {dims}D vector",
            "0.88,0.82,0.78,0.74,0.15,0.10,0.08,0.12,0.06,0.07,0.08,0.05,0.09,0.06,0.07,0.10",
        )
        q = parse_vec(raw)

    c1, c2, c3 = st.columns(3)
    with c1:
        algo = st.selectbox("Algorithm", ["hnsw", "kdtree", "bruteforce"])
    with c2:
        metric = st.selectbox("Metric", ["cosine", "euclidean", "manhattan"])
    with c3:
        k = st.number_input("K", min_value=1, max_value=20, value=5)

    ef = None
    if algo == "hnsw":
        ef = st.slider("efSearch", 10, 200, engine.settings.hnsw_ef_search)

    if st.button("Search", type="primary"):
        if len(q) != dims:
            st.error(f"Query must be {dims} dimensions (got {len(q)}).")
            return
        try:
            out = engine.vector_db.search(q, int(k), metric, algo, ef_search=ef)
        except Exception as exc:
            st.error(f"Search failed: {exc}")
            return

        st.metric("Latency", _fmt_us(out["us"]))
        rows = [
            {
                "ID": h["id"],
                "Distance": round(h["dist"], 5),
                "Category": h["cat"],
                "Metadata": h["meta"],
            }
            for h in out["hits"]
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)


def page_hnsw_playground(engine: AIEngine) -> None:
    st.header("HNSW Playground")
    st.caption("Inspect the custom HNSW graph and measure search latency.")

    stats = engine.vector_db.hnsw_stats()
    cfg = engine.settings

    c1, c2, c3 = st.columns(3)
    c1.metric("Nodes", stats.node_count)
    c2.metric("Layers", stats.layer_count)
    c3.metric("Total edges", stats.total_edges)

    c4, c5, c6 = st.columns(3)
    c4.metric("Entry point", stats.entry_point)
    c5.metric("Avg degree (L0)", stats.avg_degree)
    c6.metric("Max degree (L0)", stats.max_degree)

    st.subheader("Configuration")
    st.json(
        {
            "M": cfg.hnsw_m,
            "M0": stats.m0,
            "efConstruction": cfg.hnsw_ef_construction,
            "efSearch (default)": cfg.hnsw_ef_search,
            "seed": cfg.hnsw_seed,
            "buildTimeUs": stats.build_time_us,
        }
    )

    if stats.nodes_per_layer:
        st.subheader("Nodes per layer")
        import pandas as pd

        st.bar_chart(
            pd.DataFrame(
                {"nodes": stats.nodes_per_layer},
                index=[f"L{i}" for i in range(len(stats.nodes_per_layer))],
            )
        )

    st.subheader("Test search")
    ef = st.slider("efSearch", 10, 200, cfg.hnsw_ef_search, key="hnsw_ef")
    k = st.number_input("K", 1, 10, 5, key="hnsw_k")
    if items := engine.vector_db.all():
        q = items[0].emb
        if st.button("Run HNSW search on first demo vector"):
            from backend.vector_db.distance import get_dist_fn

            import time

            dfn = get_dist_fn("cosine")
            t0 = time.perf_counter()
            raw = engine.vector_db.hnsw.knn(q, int(k), ef, dfn)
            us = int((time.perf_counter() - t0) * 1_000_000)
            st.metric("Search latency", _fmt_us(us))
            st.write("Top IDs:", [i for _, i in raw])


def page_benchmark(engine: AIEngine) -> None:
    st.header("Benchmark")
    st.caption("Compare Brute Force, KD-Tree, and HNSW on the same query.")

    items = engine.vector_db.all()
    if not items:
        st.warning("No vectors in demo index.")
        return

    q = items[0].emb
    metric = st.selectbox("Metric", ["cosine", "euclidean", "manhattan"])
    k = st.number_input("K", 1, 10, 5)
    ef = st.slider("HNSW efSearch", 10, 200, engine.settings.hnsw_ef_search)

    if st.button("Run benchmark", type="primary"):
        b = engine.vector_db.benchmark(q, int(k), metric, ef_search=ef)
        rows = [
            {"Algorithm": "Brute Force", "Latency (µs)": b["bfUs"]},
            {"Algorithm": "KD-Tree", "Latency (µs)": b["kdUs"]},
            {"Algorithm": "HNSW", "Latency (µs)": b["hnswUs"]},
        ]
        st.dataframe(rows, hide_index=True, use_container_width=True)
        st.bar_chart({r["Algorithm"]: r["Latency (µs)"] for r in rows})


def page_documents(engine: AIEngine) -> None:
    st.header("Documents")
    st.caption("Chunk, embed via Ollama, and persist to SQLite.")

    if not engine.ollama.is_available():
        st.warning("Ollama is offline. Start `ollama serve` and pull `nomic-embed-text` to insert documents.")

    title = st.text_input("Title")
    text = st.text_area("Document text", height=200)
    uploaded = st.file_uploader("Or upload a text file", type=["txt", "md"])

    if uploaded is not None:
        text = uploaded.read().decode("utf-8", errors="replace")
        st.info(f"Loaded {len(text)} characters from {uploaded.name}")

    if st.button("Insert document", type="primary"):
        if not title.strip() or not text.strip():
            st.error("Title and text are required.")
        elif not engine.ollama.is_available():
            st.error("Ollama unavailable — cannot embed documents.")
        else:
            try:
                doc_id, chunk_ids = engine.document_db.insert_document(
                    title.strip(), text.strip(), engine.embedding_service
                )
                st.success(
                    f"Inserted document #{doc_id} — {len(chunk_ids)} chunk(s), "
                    f"{engine.document_db.get_dims()}D embeddings."
                )
            except (EmbeddingError, ValueError) as exc:
                st.error(str(exc))

    docs = engine.document_db.all()
    st.subheader(f"Stored chunks ({len(docs)})")
    if not docs:
        st.info("No documents indexed yet.")
        return

    rows = [
        {
            "Chunk ID": d.id,
            "Title": d.title[:60],
            "Words": len(d.text.split()),
            "Dim": len(d.emb),
        }
        for d in docs
    ]
    st.dataframe(rows, hide_index=True, use_container_width=True)

    del_id = st.number_input("Chunk ID to delete", min_value=1, step=1)
    if st.button("Delete chunk"):
        if engine.document_db.remove(int(del_id)):
            st.success(f"Deleted chunk {del_id} and rebuilt indexes.")
            st.rerun()
        else:
            st.error("Chunk not found or delete failed.")


def page_semantic_search(engine: AIEngine) -> None:
    st.header("Semantic Search")
    st.caption("Embed a natural-language question and retrieve top-K chunks via HNSW.")

    if engine.document_db.size() == 0:
        st.info("Insert documents first.")
        return
    if not engine.ollama.is_available():
        st.error("Ollama offline — embeddings require nomic-embed-text.")
        return

    question = st.text_input("Question", "What is dynamic programming?")
    k = st.number_input("Top-K", 1, 10, 3)

    if st.button("Search", type="primary"):
        try:
            hits = engine.retriever.retrieve(
                question, engine.embedding_service, k=int(k), mode="vector"
            )
        except EmbeddingError as exc:
            st.error(str(exc))
            return

        if not hits:
            st.warning("No results within distance threshold.")
            return

        for i, hit in enumerate(hits, 1):
            sim = 1.0 - min(hit.vector_distance, 1.0)
            with st.expander(f"Result {i} — {hit.item.title} (distance {hit.vector_distance:.4f})"):
                st.markdown(f"**Similarity:** {sim:.4f}")
                st.text(hit.item.text[:500] + ("…" if len(hit.item.text) > 500 else ""))


def page_ask_ai(engine: AIEngine) -> None:
    st.header("Ask AI (RAG)")
    st.caption("Retrieve chunks → build prompt → generate answer with Ollama.")

    if engine.document_db.size() == 0:
        st.info("Insert documents before asking questions.")
        return

    question = st.text_area("Question", "Explain the main concept in my notes.")
    k = st.number_input("Top-K chunks", 1, 10, 3)
    mode = st.selectbox("Retrieval", ["vector", "hybrid", "hybrid_rerank"])

    if st.button("Ask", type="primary"):
        if not question.strip():
            st.error("Enter a question.")
            return
        if not engine.ollama.is_available():
            st.error("Ollama offline.")
            return

        try:
            hits = engine.retriever.retrieve(
                question,
                engine.embedding_service,
                k=int(k),
                mode="hybrid" if mode.startswith("hybrid") else "vector",
                rerank=mode == "hybrid_rerank",
            )
        except EmbeddingError as exc:
            st.error(str(exc))
            return

        citations = format_citations(hits)

        if should_abstain(hits, engine.settings.rag_min_similarity):
            st.warning(ABSTAIN_MESSAGE)
            st.subheader("Sources")
            for c in citations:
                st.markdown(f"**[{c['index']}]** {c['title']} — chunk {c['chunkIndex']}")
            return

        prompt = build_rag_prompt(question, hits)
        with st.spinner("Generating answer…"):
            answer = engine.ollama.generate(prompt)

        st.subheader("Answer")
        st.markdown(answer)

        st.subheader("Sources")
        for c in citations:
            st.markdown(
                f"**[{c['index']}]** {c['title']} — chunk {c['chunkIndex']} "
                f"(similarity {c['similarity']})"
            )

        st.subheader("Retrieved context")
        for hit in hits:
            st.markdown(f"- **{hit.item.title}** — distance {hit.vector_distance:.4f}")

        st.caption(
            f"LLM: {engine.ollama.gen_model} · "
            f"Embeddings: {engine.embedding_service.model_name}"
        )


def page_evaluation(engine: AIEngine) -> None:
    st.header("Evaluation")
    st.caption("Recall@K using Brute Force as ground truth.")

    k_values = st.multiselect("K values", [1, 5, 10], default=[1, 5, 10])
    query_count = st.slider("Number of queries", 5, 20, 10)
    ef = st.slider("HNSW efSearch", 10, 200, engine.settings.hnsw_ef_search)

    if st.button("Run Recall@K evaluation", type="primary"):
        if not k_values:
            st.error("Select at least one K value.")
            return
        queries = engine.vector_db.sample_query_embeddings(query_count)
        result = engine.vector_db.evaluate_recall_batch(
            queries, sorted(k_values), "cosine", ef_search=ef
        )

        st.write(
            f"**Queries:** {result['queryCount']} · "
            f"**Dataset:** {result['datasetSize']} vectors · "
            f"**Ground truth:** brute force"
        )

        rows = []
        for row in result["results"]:
            for k, recall in row["recall"].items():
                rows.append(
                    {
                        "Algorithm": row["algorithm"],
                        "K": int(k),
                        "Recall": f"{recall:.1%}",
                        "Recall (numeric)": recall,
                        "Latency (avg µs)": row["latencyUs"],
                    }
                )
        st.dataframe(rows, hide_index=True, use_container_width=True)

        hnsw_rows = [r for r in rows if r["Algorithm"] == "hnsw"]
        if hnsw_rows:
            import pandas as pd

            st.bar_chart(
                pd.DataFrame(
                    {"recall": [r["Recall (numeric)"] for r in hnsw_rows]},
                    index=[f"K={r['K']}" for r in hnsw_rows],
                )
            )


def page_system_status(engine: AIEngine) -> None:
    st.header("System Status")

    ollama_up = engine.ollama.is_available()
    emb = engine.embedding_service.stats()
    demo_stats = engine.vector_db.hnsw_stats()
    doc_stats = engine.document_db.hnsw_stats()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Application")
        st.success("Streamlit UI running")
        st.write(f"Demo vectors: **{engine.vector_db.size()}** ({engine.settings.demo_dims}D)")
        st.write(f"Document chunks: **{engine.document_db.size()}**")
        st.write(f"Documents: **{engine.document_db.document_count()}**")
    with c2:
        st.subheader("Ollama")
        if ollama_up:
            st.success("Online")
        else:
            st.error("Offline")
        st.write(f"Embedding model: `{engine.embedding_service.model_name}`")
        st.write(f"LLM model: `{engine.ollama.gen_model}`")
        st.write(f"Embed dimension: {emb['dimension'] or 'not set'}")

    st.subheader("SQLite")
    st.write(f"Path: `{engine.settings.sqlite_path}`")
    st.write(f"Persisted chunks: {engine.document_db.storage.count_chunks()}")
    st.write(f"Embedding cache entries: {engine.document_db.storage.count_embedding_cache()}")

    st.subheader("HNSW (demo index)")
    st.json(
        {
            "nodes": demo_stats.node_count,
            "layers": demo_stats.layer_count,
            "edges": demo_stats.total_edges,
            "entryPoint": demo_stats.entry_point,
            "buildTimeUs": demo_stats.build_time_us,
            "searchCount": demo_stats.search_count,
        }
    )

    if engine.document_db.size() > 0:
        st.subheader("HNSW (document index)")
        st.json(
            {
                "nodes": doc_stats.node_count,
                "layers": doc_stats.layer_count,
                "edges": doc_stats.total_edges,
            }
        )
