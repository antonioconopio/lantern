"""Shared fixtures: an in-memory vector store with fake embeddings, and an API
client wired to it. No API keys, network calls or model downloads needed."""

import uuid

import pytest
from fastapi.testclient import TestClient
from langchain_chroma import Chroma
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.config import Settings, get_llm, get_settings, get_vector_store
from app.main import app


@pytest.fixture
def store():
    """Fresh in-memory Chroma collection per test.

    DeterministicFakeEmbedding gives the same vector for the same text, so
    tests are repeatable, but similarity between different texts is not
    meaningful: tests check plumbing, not retrieval quality (that's the eval set).
    """
    s = Chroma(
        collection_name=f"test-{uuid.uuid4().hex}",
        embedding_function=DeterministicFakeEmbedding(size=64),
    )
    yield s
    s.delete_collection()


@pytest.fixture
def fake_llm():
    return FakeListChatModel(responses=["Lanterns use wicks [1]."])


@pytest.fixture
def client(store, fake_llm, tmp_path):
    settings = Settings(upload_dir=str(tmp_path / "uploads"), retrieval_k=4)
    app.dependency_overrides[get_vector_store] = lambda: store
    app.dependency_overrides[get_llm] = lambda: fake_llm
    app.dependency_overrides[get_settings] = lambda: settings
    yield TestClient(app)
    app.dependency_overrides.clear()