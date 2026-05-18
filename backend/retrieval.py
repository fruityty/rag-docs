from openai import OpenAI
import chromadb

from backend.config import get_settings
from backend.prompts import SYSTEM_PROMPT, build_context_prompt


settings = get_settings()
client = OpenAI(api_key=settings.openai_api_key)
chroma_client = chromadb.PersistentClient(path=str(settings.chroma_path))


def get_collection():
    return chroma_client.get_or_create_collection(name=settings.collection_name)


def embed_text(text: str) -> list[float]:
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=text,
    )
    return response.data[0].embedding


def retrieve(question: str, top_k: int | None = None) -> list[dict]:
    collection = get_collection()
    query_embedding = embed_text(question)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k or settings.top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for text, metadata, distance in zip(documents, metadatas, distances):
        chunks.append(
            {
                "text": text,
                "metadata": metadata or {},
                "distance": distance,
            }
        )

    return chunks


def answer_question(question: str, top_k: int | None = None) -> dict:
    chunks = retrieve(question, top_k=top_k)
    user_prompt = build_context_prompt(question, chunks)

    response = client.responses.create(
        model=settings.chat_model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    return {
        "answer": response.output_text,
        "sources": [
            {
                "source": chunk["metadata"].get("source", "unknown source"),
                "chunk_id": chunk["metadata"].get("chunk_id"),
                "distance": chunk["distance"],
            }
            for chunk in chunks
        ],
    }

