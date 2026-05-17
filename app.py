"""
Chat with Your Documents — a RAG app built with Streamlit + Gemini + sentence-transformers.
Free to use: no API key required from visitors.
"""

import os
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from rag.ingest import ingest_pdf
from rag.retriever import retrieve
from rag.chat import answer

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Chat with Your Documents",
    page_icon="📄",
    layout="wide",
)

# ── Load API key (from Streamlit secrets or .env — not exposed to users) ──────

def get_api_key() -> str | None:
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.getenv("GEMINI_API_KEY")

GEMINI_API_KEY = get_api_key()

# ── Cache the embedding model so it loads once per session ────────────────────

@st.cache_resource(show_spinner="Loading embedding model…")
def load_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")

# ── Session state defaults ────────────────────────────────────────────────────

if "doc_store" not in st.session_state:
    st.session_state.doc_store = None
if "history" not in st.session_state:
    st.session_state.history = []

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📄 Doc RAG")
    st.markdown("Upload a PDF and ask questions about it.")
    st.markdown("**Free to use — no account needed.**")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file:
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        if (
            st.session_state.doc_store is None
            or st.session_state.doc_store.get("file_key") != file_key
        ):
            with st.spinner(f"Ingesting {uploaded_file.name}…"):
                try:
                    model = load_model()
                    doc = ingest_pdf(model, uploaded_file.read(), uploaded_file.name)
                    doc["file_key"] = file_key
                    st.session_state.doc_store = doc
                    st.session_state.history = []
                    st.success(
                        f"Ingested **{uploaded_file.name}** — "
                        f"{len(doc['chunks'])} chunks ready."
                    )
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    if st.session_state.doc_store:
        st.divider()
        st.caption(f"Active doc: **{st.session_state.doc_store['filename']}**")
        if st.button("Clear document & chat"):
            st.session_state.doc_store = None
            st.session_state.history = []
            st.rerun()

    st.divider()
    st.markdown(
        "Built with [Streamlit](https://streamlit.io) · "
        "[Gemini](https://aistudio.google.com) · "
        "[sentence-transformers](https://www.sbert.net)"
    )

# ── Main area ─────────────────────────────────────────────────────────────────

st.title("Chat with Your Documents")

if not GEMINI_API_KEY:
    st.error("Gemini API key not configured. Add it to `.streamlit/secrets.toml`.")
    st.stop()

if st.session_state.doc_store is None:
    st.info("Upload a PDF in the sidebar to begin.")
    st.stop()

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask a question about your document…")

if query:
    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("Searching document…"):
        model = load_model()
        chunks = retrieve(model, query, st.session_state.doc_store)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                response_text = answer(
                    GEMINI_API_KEY,
                    query,
                    chunks,
                    st.session_state.history,
                )
                st.markdown(response_text)
            except Exception as e:
                response_text = f"Error: {e}"
                st.error(response_text)

    st.session_state.history.append({"role": "user", "content": query})
    st.session_state.history.append({"role": "assistant", "content": response_text})
    if len(st.session_state.history) > 20:
        st.session_state.history = st.session_state.history[-20:]
