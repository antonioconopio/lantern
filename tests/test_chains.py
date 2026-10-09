from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.config import get_llm
from app.main import app
from lantern.chains import build_qa_chain, qa_prompt


def test_prompt_includes_question():
    messages = qa_prompt.format_messages(question="What is RAG?")
    assert messages[0].type == "system"
    assert messages[-1].content == "What is RAG?"


def test_chain_returns_model_text():
    fake = FakeListChatModel(responses=["RAG means retrieval-augmented generation."])
    chain = build_qa_chain(fake)
    assert chain.invoke({"question": "What is RAG?"}) == (
        "RAG means retrieval-augmented generation."
    )


def test_ask_endpoint_uses_injected_model():
    fake = FakeListChatModel(responses=["Hello from Lantern."])
    app.dependency_overrides[get_llm] = lambda: fake
    try:
        client = TestClient(app)
        resp = client.post("/ask", json={"question": "Hi?"})
        assert resp.status_code == 200
        assert resp.json() == {"answer": "Hello from Lantern."}
    finally:
        app.dependency_overrides.clear()


def test_ask_rejects_empty_question():
    client = TestClient(app)
    resp = client.post("/ask", json={"question": ""})
    assert resp.status_code == 422