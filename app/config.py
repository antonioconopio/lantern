"""Settings loaded from .env"""

from functools import lru_cache

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_model: str = "anthropic:claude-haiku-5-5"

    chroma_persist_dir: str = "data/chroma"
    upload_dir: str = "data/uploads"


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_llm() -> BaseChatModel:
    """Create the chat model once and reuse it across requests."""
    settings = get_settings()
    return init_chat_model(settings.llm_model)