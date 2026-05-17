"""
PDF ingestion: parse → chunk → embed with sentence-transformers → in-memory store.
"""

import io
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


CHUNK_SIZE = 500      # words per chunk
CHUNK_OVERLAP = 50    # words of overlap between consecutive chunks


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


def ingest_pdf(model: SentenceTransformer, file_bytes: bytes, filename: str) -> dict:
    """Return a document store dict ready for retrieval."""
    text = extract_text(file_bytes)
    chunks = _chunk_words(text)
    embeddings = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
    return {
        "filename": filename,
        "chunks": chunks,
        "embeddings": np.array(embeddings, dtype=np.float32),
    }
