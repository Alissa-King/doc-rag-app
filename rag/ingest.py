"""
PDF ingestion: parse → chunk → embed → store in a simple in-memory vector store.
Uses the Anthropic embeddings API (voyage-3) via the anthropic client.
"""

import io
import math
import anthropic
import numpy as np
from pypdf import PdfReader


CHUNK_SIZE = 500      # words per chunk
CHUNK_OVERLAP = 50    # words of overlap between consecutive chunks
EMBED_MODEL = "voyage-3"
EMBED_BATCH = 128     # max texts per embedding request


def extract_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    return "\n\n".join(pages)


def _chunk_words(text: str) -> list[str]:
    words = text.split()
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + CHUNK_SIZE])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def _embed_batch(client: anthropic.Anthropic, texts: list[str]) -> list[list[float]]:
    response = client.beta.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        betas=["embeddings-2025-04-14"],
    )
    return [item.embedding for item in response.data]


def embed_chunks(client: anthropic.Anthropic, chunks: list[str]) -> np.ndarray:
    all_embeddings = []
    for i in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[i : i + EMBED_BATCH]
        all_embeddings.extend(_embed_batch(client, batch))
    return np.array(all_embeddings, dtype=np.float32)


def ingest_pdf(client: anthropic.Anthropic, file_bytes: bytes, filename: str) -> dict:
    """Return a document store dict ready for retrieval."""
    text = extract_text(file_bytes)
    chunks = _chunk_words(text)
    embeddings = embed_chunks(client, chunks)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    embeddings_normalized = embeddings / norms
    return {
        "filename": filename,
        "chunks": chunks,
        "embeddings": embeddings_normalized,
    }
