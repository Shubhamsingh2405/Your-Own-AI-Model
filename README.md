# Your-OWN-AI — Vector Database in Python

A **vector database** built from scratch in **Python**, with a web UI.  
Implements **HNSW**, **KD-Tree**, and **Brute Force** search side-by-side, plus a **RAG pipeline** using local Ollama models.

No C++ compiler or build step — just Python, Flask, and Ollama.

---

## Quick Start

```powershell
git clone https://github.com/YOUR_USERNAME/Your-OWN-AI.git
cd Your-OWN-AI

python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python main.py
```

Open **http://localhost:8080** in your browser.

For RAG features, install [Ollama](https://ollama.com) and run:

```powershell
ollama pull nomic-embed-text
ollama pull llama3.2
```

---

## What This Project Does

| Feature | Description |
|---|---|
| **3 Search Algorithms** | HNSW, KD-Tree, Brute Force — compare speed in the UI |
| **3 Distance Metrics** | Cosine, Euclidean, Manhattan |
| **16D Demo Vectors** | 20 pre-loaded vectors (CS, Math, Food, Sports) |
| **2D PCA Scatter Plot** | Live semantic-space visualization |
| **Document Embedding** | Paste text → Ollama `nomic-embed-text` (768D) |
| **RAG Pipeline** | Ask questions → HNSW retrieval → local LLM answer |
| **REST API** | insert, delete, search, benchmark, hnsw-info |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.10+, Flask |
| HTTP client (Ollama) | `requests` |
| Vector indexes | Custom HNSW, KD-Tree, Brute Force (pure Python) |
| Frontend | `index.html` (vanilla JS, canvas PCA plot) |
| LLM / embeddings | Ollama (`nomic-embed-text`, `llama3.2`) |

---

## How It Works

```
Your Text
    │
    ▼
Ollama (nomic-embed-text)     → 768-dimensional embedding
    │
    ▼
HNSW Index (Python)             → multilayer graph index
    │
    ▼
Semantic Search               → nearest-neighbor retrieval
    │
    ▼
Ollama (llama3.2)             → answer from retrieved chunks
    │
    ▼
Answer
```

---

## Prerequisites

1. **Python 3.10+** — [python.org/downloads](https://www.python.org/downloads/) (check “Add to PATH”)
2. **Git** — [git-scm.com/download/win](https://git-scm.com/download/win)
3. **Ollama** (optional, for documents & RAG) — [ollama.com](https://ollama.com)

Verify Python:

```powershell
python --version
pip --version
```

---

## Full Setup (Windows)

### 1. Install Ollama models (for RAG)

```powershell
ollama pull nomic-embed-text
ollama pull llama3.2
ollama list
```

> 8GB RAM recommended. Models use ~3GB total.

### 2. Clone and install dependencies

```powershell
git clone https://github.com/YOUR_USERNAME/Your-OWN-AI.git
cd Your-OWN-AI

python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Run the server

**Terminal 1** — Ollama (if not already in system tray):

```powershell
ollama serve
```

**Terminal 2** — VectorDB:

```powershell
cd Your-OWN-AI
.venv\Scripts\Activate.ps1
python main.py
```

Expected output:

```
=== VectorDB Engine ===
http://localhost:8080
20 demo vectors | 16 dims | HNSW+KD-Tree+BruteForce
Ollama: ONLINE
  embed model: nomic-embed-text  gen model: llama3.2
```

Open **http://localhost:8080**.

---

## Using the Application

### Tab 1: Search (Demo Vectors)

- Search: `binary tree`, `sushi`, `basketball`, `calculus`
- Pick algorithm: **HNSW**, **KD-Tree**, or **Brute Force**
- Pick metric: **Cosine**, **Euclidean**, or **Manhattan**
- **⚡ SEARCH** — results + scatter plot highlights
- **▶ COMPARE ALL ALGOS** — latency benchmark

### Tab 2: Documents

1. Enter a title and paste text
2. **⚡ EMBED & INSERT** — chunks text (250 words, 30 overlap), embeds via Ollama
3. Each chunk is indexed in HNSW

### Tab 3: Ask AI (RAG)

1. Insert documents in Tab 2 first
2. Ask a question → **🤖 ASK AI**
3. Pipeline: embed question → HNSW retrieve → llama3.2 generate

---

## REST API

Base URL: `http://localhost:8080`

### Demo vectors

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/search?v=...&k=5&metric=cosine&algo=hnsw` | K-NN search |
| `POST` | `/insert` | Insert vector (`metadata`, `category`, `embedding`) |
| `DELETE` | `/delete/:id` | Delete by ID |
| `GET` | `/items` | List all vectors |
| `GET` | `/benchmark?v=...&k=5&metric=cosine` | Compare algorithms |
| `GET` | `/hnsw-info` | HNSW graph stats |
| `GET` | `/stats` | DB stats |

### Documents & RAG

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `POST` | `/doc/insert` | `{"title":"...","text":"..."}` | Embed and store |
| `GET` | `/doc/list` | — | List documents |
| `DELETE` | `/doc/delete/:id` | — | Delete chunk |
| `POST` | `/doc/ask` | `{"question":"...","k":3}` | RAG answer |
| `GET` | `/status` | — | Ollama status |

### Examples

```powershell
curl "http://localhost:8080/search?v=0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1&k=3&metric=cosine&algo=hnsw"

curl -X POST http://localhost:8080/doc/ask `
  -H "Content-Type: application/json" `
  -d '{"question":"What is dynamic programming?","k":3}'
```

---

## Project Structure

```
Your-OWN-AI/
├── main.py           ← Backend: indexes, REST API, RAG
├── requirements.txt  ← flask, requests
├── index.html        ← Frontend UI
└── README.md
```

### Architecture (`main.py`)

```
BruteForce     O(N·d)     Exact baseline
KDTree         O(log N)   Exact, axis-aligned splits
HNSW           O(log N)   Approximate graph search

VectorDB       16D demo vectors (all 3 algorithms)
DocumentDB     768D Ollama embeddings (HNSW + brute fallback)
OllamaClient   /api/embeddings + /api/generate
```

---

## Common Issues

| Problem | Fix |
|---|---|
| `Ollama: OFFLINE` | Run `ollama serve` |
| Slow first embed | Model downloading — wait ~2 min |
| `python: command not found` | Reinstall Python with “Add to PATH” |
| Port 8080 in use | `netstat -ano \| findstr 8080` then `taskkill /PID <pid> /F` |
| Slow LLM answers | Use `llama3.2:1b` instead (see below) |

### Faster LLM

```powershell
ollama pull llama3.2:1b
```

In `main.py`, set on `OllamaClient`:

```python
self.gen_model = "llama3.2:1b"
```

Restart with `python main.py`.

---

## License

MIT — use this however you want.
