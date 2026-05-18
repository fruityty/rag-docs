# Example Knowledge Base

This is a starter document for your RAG system.

The current project uses FastAPI as the API layer, ChromaDB as the local vector database, and OpenAI for embeddings and answer generation.

To add your own knowledge, place `.txt` or `.md` files in the `data/docs` folder, then run:

```powershell
python -m backend.ingest
```

The ingestion script reads documents, chunks them, creates embeddings, and stores the chunks in the local Chroma database.

