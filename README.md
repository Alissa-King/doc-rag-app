# Chat with Your Documents

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://doc-rag-app.streamlit.app/)

A Retrieval-Augmented Generation (RAG) app that lets you upload a PDF and have a natural conversation about its contents. Built with Streamlit and the Anthropic API.

**[Try it live →](https://doc-rag-app.streamlit.app/)**

## How it works

```
PDF upload → text extraction → chunking → Voyage embeddings → cosine similarity search
                                                                         ↓
                                                 User question → retrieve top-5 chunks
                                                                         ↓
                                                          Claude answers from context
```

1. **Ingest** — the PDF is parsed, split into overlapping 500-word chunks, and embedded using Anthropic's Voyage 3 embedding model.
2. **Retrieve** — when you ask a question, it's embedded with the same model and the top-5 most similar chunks are retrieved via cosine similarity.
3. **Generate** — Claude receives only the retrieved chunks as context and answers strictly from them. Prompt caching keeps repeated queries fast and cheap.

## Tech stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| Embeddings | Anthropic Voyage 3 |
| Vector search | NumPy (cosine similarity) |
| LLM | Claude Sonnet (claude-sonnet-4-6) |
| PDF parsing | pypdf |

## Getting started

### 1. Clone and install

```bash
git clone https://github.com/Alissa-King/doc-rag-app.git
cd doc-rag-app
pip install -r requirements.txt
```

### 2. Set your API key

```bash
cp .env.example .env
# edit .env and add your Anthropic API key
```

Or just paste it directly into the sidebar when the app opens.

### 3. Run

```bash
streamlit run app.py
```

## Features

- Multi-turn conversation with memory (last 20 turns)
- Prompt caching — repeated questions over the same document reuse cached context, cutting latency and cost
- Chunking with overlap so context boundaries don't cut off answers
- "Clear document" button to swap PDFs mid-session
- Works entirely in-memory — no database required

## Project structure

```
doc-rag-app/
├── app.py              # Streamlit UI and session management
├── rag/
│   ├── ingest.py       # PDF parsing, chunking, embedding
│   ├── retriever.py    # Cosine similarity search
│   └── chat.py         # Claude API call with prompt caching
├── requirements.txt
└── .env.example
```
