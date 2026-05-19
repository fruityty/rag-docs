from pathlib import Path
from uuid import uuid4

import chromadb
from openai import OpenAI

from backend.config import get_settings


SUPPORTED_EXTENSIONS = {".txt", ".md", ".mdx"}
CHUNK_SIZE = 1_000
CHUNK_OVERLAP = 200

settings = get_settings()
client = OpenAI(api_key=settings.openai_api_key)
chroma_client = chromadb.PersistentClient(path=str(settings.chroma_path))


def load_documents(docs_path: Path) -> list[dict]:
    docs = []

    for path in docs_path.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        text = path.read_text(encoding="utf-8")
        if text.strip():
            docs.append({"source": str(path), "text": text})

    return docs


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    return [item.embedding for item in response.data]


def reset_collection():
    try:
        chroma_client.delete_collection(settings.collection_name)
    except ValueError:
        pass
    return chroma_client.get_or_create_collection(name=settings.collection_name)


def ingest() -> None:
    settings.docs_path.mkdir(parents=True, exist_ok=True)
    settings.chroma_path.mkdir(parents=True, exist_ok=True)

    documents = load_documents(settings.docs_path)
    if not documents:
        print(f"No .txt or .md documents found in {settings.docs_path}")
        return

    collection = reset_collection()

    ids = []
    chunks = []
    metadatas = []

    for doc in documents:
        for index, chunk in enumerate(chunk_text(doc["text"])):
            chunk_id = f"{Path(doc['source']).name}:{index}"
            ids.append(str(uuid4()))
            chunks.append(chunk)
            metadatas.append(
                {
                    "source": doc["source"],
                    "chunk_id": chunk_id,
                }
            )

    embeddings = embed_texts(chunks)
    collection.add(
        ids=ids,
        documents=chunks,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print(f"Indexed {len(chunks)} chunks from {len(documents)} documents.")


if __name__ == "__main__":
    ingest()
