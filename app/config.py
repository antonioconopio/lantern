"""Settings loaded from .env"""

from functools import lru_cache

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_model: str = "anthropic:claude-haiku-5-5"
    embedding_model: str = "huggingface:sentence-transformers/all-MiniLM-L6-v2"

    chroma_persist_dir: str = "data/chroma"
    chroma_collection: str = "lantern"
    upload_dir: str = "data/uploads"

    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_k: int = 4

@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_llm() -> BaseChatModel:
    """Create the chat model once and reuse it across requests."""
    return init_chat_model(get_settings().llm_model)


@lru_cache
def get_embeddings() -> Embeddings:
    """Create the embedding model once (local models are slow to load)."""
    return init_embeddings(get_settings().embedding_model)


@lru_cache
def get_vector_store() -> VectorStore:
    """Chroma collection persisted to disk, shared across requests."""
    settings = get_settings()
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_persist_dir,
    )