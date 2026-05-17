"""
Cosine-similarity retrieval over in-memory normalized embeddings.
"""

import anthropic
import numpy as np
from rag.ingest import EMBED_MODEL


TOP_K = 5


def retrieve(
    client: anthropic.Anthropic,
    query: str,
    doc_store: dict,
    top_k: int = TOP_K,
) -> list[str]:
    response = client.beta.embeddings.create(
        model=EMBED_MODEL,
        input=[query],
        betas=["embeddings-2025-04-14"],
    )
    q_emb = np.array(response.data[0].embedding, dtype=np.float32)
    norm = np.linalg.norm(q_emb)
    if norm:
        q_emb /= norm

    scores = doc_store["embeddings"] @ q_emb
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [doc_store["chunks"][i] for i in top_indices]
