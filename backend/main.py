from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.retrieval import answer_question


app = FastAPI(title="Custom RAG API")


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)


@app.get("/")
def index():
    return FileResponse("frontend/index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest):
    return answer_question(request.question, top_k=request.top_k)

