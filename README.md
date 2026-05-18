# Custom RAG Starter

This is a customizable Retrieval-Augmented Generation starter using:

- FastAPI for the backend API
- OpenAI for embeddings and answer generation
- ChromaDB for local vector storage
- A plain HTML chat page for quick testing

## 1. Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and add your OpenAI API key.

## 2. Add documents

Put `.txt` or `.md` files into:

```text
data/docs/
```

Example:

```text
data/docs/company_policy.md
data/docs/product_faq.txt
```

## 3. Build the index

```powershell
python -m backend.ingest
```

## 4. Start the API

```powershell
uvicorn backend.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

## Project Structure

```text
backend/
  config.py       # environment settings
  ingest.py       # load, chunk, embed, and store documents
  main.py         # FastAPI app
  prompts.py      # RAG prompt template
  retrieval.py    # vector search and answer generation
data/
  docs/           # source documents
  chroma/         # local vector database
frontend/
  index.html      # simple chat UI
```

## How The RAG Flow Works

1. `backend.ingest` reads files from `data/docs`.
2. Documents are split into overlapping chunks.
3. Each chunk is embedded with OpenAI embeddings.
4. Chunks are stored in Chroma with metadata.
5. A user asks a question through the chat API.
6. The app retrieves the most relevant chunks.
7. The model answers using only the retrieved context.

