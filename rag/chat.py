"""
Generate an answer from Claude using retrieved context chunks.
Uses prompt caching on the context block to save tokens on repeated questions.
"""

import anthropic


CHAT_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1024

SYSTEM_PROMPT = """You are a precise document assistant. Answer the user's question \
using ONLY the provided context excerpts. If the answer is not in the context, say \
"I couldn't find that in the uploaded document." Do not fabricate information."""


def build_context_block(chunks: list[str]) -> str:
    formatted = "\n\n---\n\n".join(
        f"[Excerpt {i+1}]\n{chunk}" for i, chunk in enumerate(chunks)
    )
    return f"<context>\n{formatted}\n</context>"


def answer(
    client: anthropic.Anthropic,
    query: str,
    chunks: list[str],
    history: list[dict],
) -> str:
    context_text = build_context_block(chunks)

    # Inject context as the first user turn with cache_control so repeated
    # questions over the same document hit the prompt cache.
    context_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": context_text,
                "cache_control": {"type": "ephemeral"},
            }
        ],
    }
    query_message = {"role": "user", "content": query}

    # history already contains assistant turns; prepend context before each query
    messages = [context_message] + history + [query_message]

    response = client.messages.create(
        model=CHAT_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=messages,
        betas=["prompt-caching-2024-07-31"],
    )
    return response.content[0].text
