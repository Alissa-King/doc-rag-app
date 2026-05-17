"""
Cosine-similarity retrieval over in-memory normalized embeddings.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


TOP_K = 5


def retrieve(
    model: SentenceTransformer,
    query: str,
    doc_store: dict,
    top_k: int = TOP_K,
) -> list[str]:
    q_emb = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    scores = doc_store["embeddings"] @ q_emb
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [doc_store["chunks"][i] for i in top_indices]
