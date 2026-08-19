# Your-OWN-AI — AI/ML Vector Search & RAG Platform

A **from-scratch vector database** and **RAG experimentation platform** in Python with a **Streamlit** UI.

Implements custom **HNSW**, **KD-Tree**, and **Brute Force** search, **Ollama embeddings**, **SQLite persistence**, **Recall@K evaluation**, and a transparent **RAG pipeline** — without FAISS, Pinecone, LangChain, or a separate frontend framework.

---

## Architecture

```
Streamlit UI (app.py)
        │
   services/engine.py   ← thin facade (@st.cache_resource)
        │
   ┌────┴────────────────────────────┐
   │         backend/ (AI/ML core)    │
   ├─ vector_db/  HNSW · KD-Tree · BF │
   ├─ embeddings/ Ollama + cache       │
   ├─ rag/        chunk · retrieve · generate │
   ├─ retrieval/  BM25 · hybrid · rerank      │
   ├─ storage/    SQLite                     │
   └─ benchmarks/ recall · latency            │
        │
     Ollama (nomic-embed-text + llama3.2)
```

**One frontend:** Streamlit only. No Flask, no `index.html`, no REST layer required for the UI.

---

## Quick Start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

streamlit run app.py
```

Open **http://localhost:8501**

### Ollama (documents + RAG)

```powershell
ollama pull nomic-embed-text
ollama pull llama3.2
```

Copy `.env.example` → `.env` to customize settings.

---

## Streamlit Pages

| Page | Purpose |
|------|---------|
| **Vector Search** | Query 16D demo index — algorithm, metric, K, latency |
| **HNSW Playground** | Graph stats, layers, degree, search latency |
| **Benchmark** | Brute Force vs KD-Tree vs HNSW bar chart |
| **Documents** | Paste/upload text → chunk → embed → SQLite |
| **Semantic Search** | Natural-language → embedding → HNSW top-K |
| **Ask AI** | Full RAG with sources and citations |
| **Evaluation** | Recall@1/5/10 vs brute-force ground truth |
| **System Status** | Ollama, SQLite, HNSW, vector counts |

---

## Project Structure

```
Your-OWN-AI/
├── app.py                 # Streamlit entry point
├── services/engine.py     # AIEngine facade
├── ui/pages.py            # Page render functions
├── backend/
│   ├── vector_db/         # Custom HNSW, KD-Tree, Brute Force
│   ├── embeddings/        # Ollama provider + cache
│   ├── rag/               # Chunker, retriever, generator
│   ├── retrieval/         # BM25, hybrid, reranker
│   ├── storage/           # SQLite
│   └── benchmarks/        # Recall, latency
├── experiments/           # CLI benchmark scripts
├── tests/
├── data/eval/             # RAG eval dataset
├── requirements.txt
└── .env.example
```

---

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama server |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `LLM_MODEL` | `llama3.2` | Generation model |
| `HNSW_M` | 16 | Graph connectivity |
| `HNSW_EF_SEARCH` | 50 | Search accuracy/speed |
| `HYBRID_ALPHA` | 0.7 | Vector weight in hybrid search |
| `SQLITE_PATH` | `data/vectordb.sqlite` | Persistence |

---

## Experiments (CLI)

```powershell
python experiments/evaluate_recall.py --queries 20
python experiments/benchmark_algorithms.py --sizes 100,500,1000
python experiments/evaluate_rag.py --mode hybrid
```

All metrics are computed at runtime — nothing is hardcoded.

---

## Testing

```powershell
python -m pytest -q
```

Tests cover distance functions, BM25, hybrid fusion, HNSW recall, and engine loading. **Ollama is not required** for tests.

---

## Docker

```powershell
docker compose up --build
```

App: **http://localhost:8501**

---

## Interview Talking Points

1. **Custom HNSW** — layer traversal, efSearch trade-off, graph stats
2. **Brute force baseline** — ground truth for Recall@K
3. **Embeddings** — Ollama + dimension lock + SQLite cache
4. **RAG** — retrieve → cite → abstain when context is weak
5. **SQLite + in-memory HNSW** — persist documents, fast search
6. **Streamlit** — thin UI over importable Python core

---

## Limitations

- HNSW is in-memory; rebuilds on document delete
- Demo vectors use fixed 16D data (not Ollama)
- Not designed for billion-scale production loads

---

## License

MIT
