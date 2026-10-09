"""Shared fixtures: an in-memory vector store with fake embeddings, a fake
structured-output model, and an API client wired to both. No API keys,
network calls or model downloads needed."""

import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda

from app.config import (
    Settings,
    get_history_store,
    get_llm,
    get_settings,
    get_vector_store,
)
from app.main import app
from lantern.answer import GroundedAnswer
from lantern.memory import ChatHistoryStore


class FakeStructuredChatModel(FakeListChatModel):
    """Fake chat model whose with_structured_output() returns a fixed object.

    LangChain's built-in fakes don't support structured output, so this
    stands in for it. Every prompt it receives is recorded in `calls`, so
    tests can check what context reached the model.
    """

    structured: Any = None
    calls: list = []
    structured_kwargs: dict = {}

    def with_structured_output(self, schema, **kwargs):
        self.structured_kwargs = kwargs

        def respond(prompt_value):
            self.calls.append(prompt_value)
            assert isinstance(self.structured, schema)
            return self.structured

        return RunnableLambda(respond)


def make_fake_llm(rewrites=("REWRITTEN QUESTION",), **answer_fields) -> FakeStructuredChatModel:
    """`rewrites` are the plain-text replies (used by the query-rewrite step);
    `answer_fields` override the structured GroundedAnswer it returns."""
    fields = {
        "answer": "Lanterns burn oil [1].",
        "citations": [1],
        "answerable": True,
        "confidence": "high",
        **answer_fields,
    }
    return FakeStructuredChatModel(
        responses=list(rewrites), structured=GroundedAnswer(**fields), calls=[]
    )


class RecordingRetriever(BaseRetriever):
    """Returns fixed docs and records every query it's asked."""

    docs: list[Document] = []
    queries: list[str] = []

    def _get_relevant_documents(self, query, *, run_manager):
        self.queries.append(query)
        return self.docs


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
    return make_fake_llm()


@pytest.fixture
def client(store, fake_llm, tmp_path):
    settings = Settings(upload_dir=str(tmp_path / "uploads"), retrieval_k=4)
    app.dependency_overrides[get_vector_store] = lambda: store
    app.dependency_overrides[get_llm] = lambda: fake_llm
    app.dependency_overrides[get_settings] = lambda: settings
    histories = ChatHistoryStore(max_messages=10)
    app.dependency_overrides[get_history_store] = lambda: histories
    yield TestClient(app)
    app.dependency_overrides.clear()