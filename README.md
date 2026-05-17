# Chat with Your Documents

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://doc-rag-app.streamlit.app/)

A Retrieval-Augmented Generation (RAG) app that lets you upload a PDF and have a natural conversation about its contents. **Free to use — no account or API key needed for visitors.**

**[Try it live →](https://doc-rag-app.streamlit.app/)**

## How it works

```
PDF upload → text extraction → chunking → sentence-transformers embeddings → cosine similarity search
                                                                                       ↓
                                                           User question → retrieve top-5 chunks
                                                                                       ↓
                                                                    Gemini answers from context
```

1. **Ingest** — the PDF is parsed, split into overlapping 500-word chunks, and embedded locally using `all-MiniLM-L6-v2` (no API call needed).
2. **Retrieve** — when you ask a question, it's embedded with the same model and the top-5 most similar chunks are retrieved via cosine similarity.
3. **Generate** — Gemini 1.5 Flash receives only the retrieved chunks as context and answers strictly from them.

## Tech stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector search | NumPy (cosine similarity) |
| LLM | Gemini 1.5 Flash |
| PDF parsing | pypdf |

## Running locally

### 1. Clone and install

```bash
git clone https://github.com/Alissa-King/doc-rag-app.git
cd doc-rag-app
pip install -r requirements.txt
```

### 2. Set your Gemini API key

Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) — no credit card required.

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit secrets.toml and paste your key
```

### 3. Run

```bash
streamlit run app.py
```

## Deploying to Streamlit Community Cloud

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Under **Advanced settings → Secrets**, add:
   ```toml
   GEMINI_API_KEY = "your_key_here"
   ```
4. Deploy — visitors use the app for free with no key of their own

## Features

- Free for everyone — embeddings run locally, Gemini free tier handles the LLM
- Multi-turn conversation with memory (last 20 turns)
- Chunking with overlap so answers aren't split at chunk boundaries
- Embedding model cached on first load — instant on subsequent queries
- "Clear document" button to swap PDFs mid-session
- No database required — everything runs in memory

## Project structure

```
doc-rag-app/
├── app.py                        # Streamlit UI and session management
├── rag/
│   ├── ingest.py                 # PDF parsing, chunking, embedding
│   ├── retriever.py              # Cosine similarity search
│   └── chat.py                   # Gemini API call
├── .streamlit/
│   └── secrets.toml.example      # Template for API key config
├── requirements.txt
└── .env.example
```
