"""Your-OWN-AI — Streamlit entry point."""

from __future__ import annotations

import streamlit as st

from services.engine import create_engine
from ui.pages import (
    page_ask_ai,
    page_benchmark,
    page_documents,
    page_evaluation,
    page_hnsw_playground,
    page_semantic_search,
    page_system_status,
    page_vector_search,
)

PAGES = {
    "Vector Search": page_vector_search,
    "HNSW Playground": page_hnsw_playground,
    "Benchmark": page_benchmark,
    "Documents": page_documents,
    "Semantic Search": page_semantic_search,
    "Ask AI": page_ask_ai,
    "Evaluation": page_evaluation,
    "System Status": page_system_status,
}


@st.cache_resource
def get_engine():
    return create_engine()


def main() -> None:
    st.set_page_config(
        page_title="Your-OWN-AI",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("Your-OWN-AI")
    st.sidebar.caption("Vector Search & RAG Experimentation")
    page_name = st.sidebar.radio("Navigation", list(PAGES.keys()))
    st.sidebar.divider()
    st.sidebar.markdown(
        "Custom **HNSW** · KD-Tree · Brute Force\n\n"
        "Embeddings & LLM via **Ollama**"
    )

    engine = get_engine()
    PAGES[page_name](engine)


if __name__ == "__main__":
    main()
