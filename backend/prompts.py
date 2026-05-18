SYSTEM_PROMPT = """You are a careful RAG assistant.

Answer the user's question using only the provided context.
If the context does not contain the answer, say you do not know from the available documents.
Be concise, direct, and include source citations when possible.
"""


def build_context_prompt(question: str, chunks: list[dict]) -> str:
    context_blocks = []

    for index, chunk in enumerate(chunks, start=1):
        source = chunk["metadata"].get("source", "unknown source")
        text = chunk["text"]
        context_blocks.append(f"[{index}] Source: {source}\n{text}")

    context = "\n\n".join(context_blocks)

    return f"""Context:
{context}

Question:
{question}

Answer with citations like [1], [2] when the context supports the answer.
"""

