"""
Generate an answer from Gemini using retrieved context chunks.
"""

import google.generativeai as genai


CHAT_MODEL = "gemini-1.5-flash"
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
    api_key: str,
    query: str,
    chunks: list[str],
    history: list[dict],
) -> str:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=CHAT_MODEL,
        system_instruction=SYSTEM_PROMPT,
    )

    context_text = build_context_block(chunks)

    # Build Gemini-format history (role: user/model)
    gemini_history = []
    for msg in history:
        role = "model" if msg["role"] == "assistant" else "user"
        gemini_history.append({"role": role, "parts": [msg["content"]]})

    chat = model.start_chat(history=gemini_history)
    full_message = f"{context_text}\n\nQuestion: {query}"
    response = chat.send_message(full_message)
    return response.text
