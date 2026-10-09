"""Chat history store tests."""

from langchain_core.messages import AIMessage, HumanMessage

from lantern.memory import ChatHistoryStore


def test_new_session_is_empty():
    assert ChatHistoryStore().get("new") == []


def test_add_turn_stores_question_then_answer():
    store = ChatHistoryStore()
    store.add_turn("s1", "What is RAG?", "Retrieval-augmented generation.")
    assert store.get("s1") == [
        HumanMessage("What is RAG?"),
        AIMessage("Retrieval-augmented generation."),
    ]


def test_sessions_are_isolated():
    store = ChatHistoryStore()
    store.add_turn("a", "q-a", "ans-a")
    assert store.get("b") == []


def test_history_is_trimmed_to_most_recent_messages():
    store = ChatHistoryStore(max_messages=4)
    for i in range(5):
        store.add_turn("s1", f"q{i}", f"a{i}")
    assert [m.content for m in store.get("s1")] == ["q3", "a3", "q4", "a4"]


def test_get_returns_a_copy():
    store = ChatHistoryStore()
    store.add_turn("s1", "q", "a")
    store.get("s1").clear()
    assert len(store.get("s1")) == 2


def test_clear():
    store = ChatHistoryStore()
    store.add_turn("s1", "q", "a")
    assert store.clear("s1") is True
    assert store.get("s1") == []
    assert store.clear("s1") is False