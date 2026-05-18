from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel
import os


load_dotenv()


class Settings(BaseModel):
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
    chroma_path: Path = Path(os.getenv("CHROMA_PATH", "data/chroma"))
    docs_path: Path = Path(os.getenv("DOCS_PATH", "data/docs"))
    collection_name: str = os.getenv("COLLECTION_NAME", "custom_rag_docs")
    top_k: int = int(os.getenv("TOP_K", "5"))


@lru_cache
def get_settings() -> Settings:
    return Settings()

