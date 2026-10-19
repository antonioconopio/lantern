"""Agent tests: tools on their own, then the graph driven by a scripted fake model."""

import asyncio

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.config import get_llm
from app.main import app
from lantern.agents.graph import build_agent_chain, number_new_docs, render_research
from lantern.agents.tools import calculator, make_tools
from lantern.ingest import ingest_file
from tests.conftest import make_fake_agent, search_call


def doc(text, page=1, doc_id="d1"):
    return Document(
        page_content=text, metadata={"document_id": doc_id, "filename": "f.txt", "page": page}
    )


def ask(llm, store, question="Q?", history=None, **kw):
    chain = build_agent_chain(llm, store, **kw)
    return asyncio.run(chain.ainvoke({"question": question, "history": history or []}))


@pytest.fixture
def filled_store(store, tmp_path):
    path = tmp_path / "lanterns.txt"
    path.write_text("Lanterns traditionally burn oil through a wick.")
    ingest_file(path, "lanterns.txt", store, 1000, 150)
    return store


# ---- Tools ------------------------------------------------------------------


def calc(expr):
    return calculator.invoke({"expression": expr})


def test_calculator_results():
    assert calc("60 + 85") == "60 + 85 = 145"
    assert calc("10 / 4") == "10 / 4 = 2.5"
    assert calc("10 / 5") == "10 / 5 = 2"
    assert calc("-(2 + 3) * 2") == "-(2 + 3) * 2 = -10"


@pytest.mark.parametrize(
    "expr", ["__import__('os')", "x + 1", "1 / 0", "2 ** 1000", "'a' * 3", "1 +"]
)
def test_calculator_rejects_bad_input_with_error_string(expr):
    assert calc(expr).startswith("Error:")


def test_list_documents_empty_then_listed(store, filled_store):
    empty = {t.name: t for t in make_tools(_empty_store())}["list_documents"]
    assert empty.invoke({}) == "No documents have been uploaded yet."
    listed = {t.name: t for t in make_tools(filled_store)}["list_documents"]
    assert "lanterns.txt (1 chunks)" in listed.invoke({})


def _empty_store():
    from langchain_chroma import Chroma
    from langchain_core.embeddings import DeterministicFakeEmbedding
    import uuid

    return Chroma(
        collection_name=f"empty-{uuid.uuid4().hex}",
        embedding_function=DeterministicFakeEmbedding(size=64),
    )


def test_search_returns_documents_as_artifact(filled_store):
    search = {t.name: t for t in make_tools(filled_store)}["search_documents"]
    msg = search.invoke(
        {"name": "search_documents", "args": {"query": "oil"}, "id": "1", "type": "tool_call"}
    )
    assert isinstance(msg.artifact[0], Document)


def test_all_three_tools_are_bound(filled_store):
    llm = make_fake_agent()
    ask(llm, filled_store)
    assert llm.bound_tools == ["search_documents", "calculator", "list_documents"]


# ---- Helpers ----------------------------------------------------------------


def test_numbering_continues_and_duplicates_keep_first_number():
    docs = []
    first = number_new_docs([doc("alpha"), doc("beta")], docs)
    second = number_new_docs([doc("beta"), doc("gamma")], docs)
    assert first.startswith("[1]") and "[2]" in first
    assert second.startswith("[2]") and "[3] f.txt" in second
    assert [d.page_content for d in docs] == ["alpha", "beta", "gamma"]


def test_numbering_empty_result():
    assert "no relevant documents" in number_new_docs([], [])


def test_render_research():
    msgs = [
        AIMessage("", tool_calls=[search_call("oil")]),
        ToolMessage("[1] f.txt, page 1\nx", tool_call_id="s1"),
        AIMessage("done"),
    ]
    text = render_research(msgs)
    assert "Called search_documents(query='oil')" in text
    assert "Result:\n[1] f.txt" in text and "Research notes: done" in text
    assert render_research([]) == "(No research was done.)"


# ---- Graph ------------------------------------------------------------------


def test_greeting_uses_no_tools(filled_store):
    llm = make_fake_agent(citations=[], answer="Hi!")
    out = ask(llm, filled_store, "hi")
    assert out["steps"] == [] and out["docs"] == []
    assert len(llm.agent_calls) == 1
    assert out["standalone_question"] == "hi"


def test_single_search_reaches_respond_prompt_numbered(filled_store):
    llm = make_fake_agent([search_call("what burns")])
    out = ask(llm, filled_store)
    assert out["steps"] == [{"tool": "search_documents", "input": {"query": "what burns"}}]
    assert out["standalone_question"] == "what burns"
    assert len(out["docs"]) == 1
    assert "[1] lanterns.txt" in llm.respond_calls[0].to_messages()[0].content


def test_repeated_search_does_not_duplicate_sources(filled_store):
    llm = make_fake_agent([search_call("a", "s1")], [search_call("a", "s2")])
    out = ask(llm, filled_store)
    assert len(out["steps"]) == 2 and len(out["docs"]) == 1


def test_two_tool_calls_in_one_turn(filled_store):
    llm = make_fake_agent(
        [search_call("fee"), {"name": "calculator", "args": {"expression": "60 + 85"}, "id": "c1"}]
    )
    out = ask(llm, filled_store)
    assert [s["tool"] for s in out["steps"]] == ["search_documents", "calculator"]
    assert "60 + 85 = 145" in llm.respond_calls[0].to_messages()[0].content


def test_tool_results_are_fed_back_to_next_agent_call(filled_store):
    llm = make_fake_agent([search_call("a")])
    ask(llm, filled_store)
    second_prompt = llm.agent_calls[1]
    assert any(isinstance(m, ToolMessage) and "lanterns.txt" in m.content for m in second_prompt)


def test_tool_call_cap_stops_loop_and_still_answers(filled_store):
    llm = make_fake_agent(*[[search_call("a", f"s{i}")] for i in range(5)])
    out = ask(llm, filled_store, max_tool_calls=2)
    assert len(out["steps"]) == 2
    assert out["answer"].answer  # respond still ran


def test_parallel_calls_count_toward_cap(filled_store):
    llm = make_fake_agent([search_call("a", "s1"), search_call("b", "s2")], [search_call("c", "s3")])
    out = ask(llm, filled_store, max_tool_calls=2)
    assert len(out["steps"]) == 2


def test_unknown_tool_becomes_error_message(filled_store):
    llm = make_fake_agent([{"name": "nope", "args": {}, "id": "x"}])
    out = ask(llm, filled_store)
    assert out["answer"] is not None
    assert "Error: unknown tool nope" in llm.agent_calls[1][-1].content


def test_history_reaches_agent_and_respond_prompts(filled_store):
    llm = make_fake_agent([search_call("a")])
    ask(llm, filled_store, history=[HumanMessage("earlier question")])
    assert any(m.content == "earlier question" for m in llm.agent_calls[0])
    assert "earlier question" in str(llm.respond_calls[0].to_messages())


# ---- API --------------------------------------------------------------------


def test_api_agent_mode_returns_steps_and_sources(client):
    client.post("/documents", files={"file": ("l.txt", b"Lanterns burn oil.", "text/plain")})
    app.dependency_overrides[get_llm] = lambda: make_fake_agent([search_call("oil")])

    body = client.post("/ask", json={"question": "Q?", "mode": "agent"}).json()

    assert body["mode"] == "agent"
    assert body["steps"] == [{"tool": "search_documents", "input": {"query": "oil"}}]
    assert [s["index"] for s in body["sources"]] == [1]


def test_api_agent_greeting_has_no_steps(client):
    app.dependency_overrides[get_llm] = lambda: make_fake_agent(citations=[], answer="Hi!")
    body = client.post("/ask", json={"question": "hi", "mode": "agent"}).json()
    assert body["steps"] == [] and body["sources"] == []


def test_api_unknown_mode_is_422(client):
    assert client.post("/ask", json={"question": "Q?", "mode": "magic"}).status_code == 422
