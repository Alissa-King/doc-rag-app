"""
Chat with Your Documents — a RAG app built with Streamlit + Anthropic.
"""

import os
import anthropic
import streamlit as st
from dotenv import load_dotenv

from rag.ingest import ingest_pdf
from rag.retriever import retrieve
from rag.chat import answer

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Chat with Your Documents",
    page_icon="📄",
    layout="wide",
)

# ── Session state defaults ────────────────────────────────────────────────────

if "doc_store" not in st.session_state:
    st.session_state.doc_store = None
if "history" not in st.session_state:
    st.session_state.history = []  # list of {"role": ..., "content": ...}
if "client" not in st.session_state:
    st.session_state.client = None

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📄 Doc RAG")
    st.markdown("Upload a PDF and ask questions about it.")

    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        value=os.getenv("ANTHROPIC_API_KEY", ""),
        help="Get one at console.anthropic.com",
    )

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file and api_key:
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        if (
            st.session_state.doc_store is None
            or st.session_state.doc_store.get("file_key") != file_key
        ):
            with st.spinner(f"Ingesting {uploaded_file.name}…"):
                try:
                    client = anthropic.Anthropic(api_key=api_key)
                    st.session_state.client = client
                    doc = ingest_pdf(client, uploaded_file.read(), uploaded_file.name)
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
        "[Anthropic](https://anthropic.com) · "
        "[Voyage embeddings](https://docs.anthropic.com/en/docs/build-with-claude/embeddings)"
    )

# ── Main area ─────────────────────────────────────────────────────────────────

st.title("Chat with Your Documents")

if not api_key:
    st.info("Enter your Anthropic API key in the sidebar to get started.")
    st.stop()

if st.session_state.doc_store is None:
    st.info("Upload a PDF in the sidebar to begin.")
    st.stop()

# Render conversation history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
query = st.chat_input("Ask a question about your document…")

if query:
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(query)

    # Retrieve relevant chunks
    with st.spinner("Searching document…"):
        chunks = retrieve(st.session_state.client, query, st.session_state.doc_store)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                response_text = answer(
                    st.session_state.client,
                    query,
                    chunks,
                    st.session_state.history,
                )
                st.markdown(response_text)
            except Exception as e:
                response_text = f"Error: {e}"
                st.error(response_text)

    # Append to history (keep context manageable — last 20 turns)
    st.session_state.history.append({"role": "user", "content": query})
    st.session_state.history.append({"role": "assistant", "content": response_text})
    if len(st.session_state.history) > 20:
        st.session_state.history = st.session_state.history[-20:]
