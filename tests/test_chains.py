"""Chain tests with fake models: no real LLM calls."""

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

from lantern.answer import GroundedAnswer
from lantern.chains import (
    build_qa_chain,
    build_rag_chain,
    build_rewrite_chain,
    qa_prompt,
    rag_prompt,
    rewrite_prompt,
)
from lantern.retrieval import build_retriever, format_docs
from tests.conftest import RecordingRetriever, make_fake_llm


def test_prompt_includes_question():
    messages = qa_prompt.format_messages(question="What is RAG?")
    assert messages[0].type == "system"
    assert messages[-1].content == "What is RAG?"


def test_qa_chain_returns_model_text():
    fake = FakeListChatModel(responses=["RAG means retrieval-augmented generation."])
    chain = build_qa_chain(fake)
    assert chain.invoke({"question": "What is RAG?"}) == (
        "RAG means retrieval-augmented generation."
    )


def test_format_docs_numbers_sources_with_file_and_page():
    docs = [
        Document(page_content="alpha", metadata={"filename": "a.pdf", "page": 2}),
        Document(page_content="beta", metadata={"filename": "b.md", "page": 1}),
    ]
    text = format_docs(docs)
    assert "[1] a.pdf, page 2\nalpha" in text
    assert "[2] b.md, page 1\nbeta" in text


def test_format_docs_handles_no_results():
    assert "no relevant documents" in format_docs([])


def test_rag_prompt_puts_context_in_system_message():
    messages = rag_prompt.format_messages(context="[1] x.pdf, page 1\nfacts", question="Q?")
    assert "[1] x.pdf, page 1" in messages[0].content
    assert messages[-1].content == "Q?"


def test_rag_prompt_places_history_between_system_and_question():
    history = [HumanMessage("first q"), AIMessage("first a")]
    messages = rag_prompt.format_messages(context="ctx", question="Q?", history=history)
    assert [m.content for m in messages[1:]] == ["first q", "first a", "Q?"]


# ---- Stage 4: memory and query rewriting -------------------------------------

HISTORY = [
    HumanMessage("What does chapter 2 cover?"),
    AIMessage("Chapter 2 covers wicks [1]."),
]


def test_first_question_is_not_rewritten():
    llm = make_fake_llm(rewrites=["SHOULD NOT BE USED"])
    retriever = RecordingRetriever(docs=[], queries=[])

    result = build_rag_chain(llm, retriever).invoke({"question": "What is a wick?"})

    assert result["standalone_question"] == "What is a wick?"
    assert retriever.queries == ["What is a wick?"]


def test_follow_up_is_rewritten_and_retrieval_uses_rewrite():
    llm = make_fake_llm(rewrites=["What does chapter 3 cover?"])
    retriever = RecordingRetriever(docs=[], queries=[])

    result = build_rag_chain(llm, retriever).invoke(
        {"question": "What about chapter 3?", "history": HISTORY}
    )

    assert result["standalone_question"] == "What does chapter 3 cover?"
    assert retriever.queries == ["What does chapter 3 cover?"]


def test_answer_prompt_gets_history_and_original_question():
    llm = make_fake_llm(rewrites=["What does chapter 3 cover?"])
    retriever = RecordingRetriever(docs=[], queries=[])

    build_rag_chain(llm, retriever).invoke(
        {"question": "What about chapter 3?", "history": HISTORY}
    )

    messages = llm.calls[0].to_messages()
    assert [m.content for m in messages[1:]] == [
        "What does chapter 2 cover?",
        "Chapter 2 covers wicks [1].",
        "What about chapter 3?",
    ]


def test_rewrite_chain_sends_history_and_strips_output():
    llm = FakeListChatModel(responses=["  What does chapter 3 cover?\n"])
    assert (
        build_rewrite_chain(llm).invoke({"question": "And chapter 3?", "history": HISTORY})
        == "What does chapter 3 cover?"
    )
    messages = rewrite_prompt.format_messages(question="And chapter 3?", history=HISTORY)
    assert messages[1].content == "What does chapter 2 cover?"
    assert messages[-1].content == "Follow-up question: And chapter 3?"


def test_rag_chain_returns_structured_answer_and_retrieved_docs(store, fake_llm):
    store.add_documents(
        [Document(page_content="Lanterns burn oil.", metadata={"filename": "l.txt", "page": 1})]
    )
    chain = build_rag_chain(fake_llm, build_retriever(store, k=4))

    result = chain.invoke({"question": "What do lanterns burn?"})

    assert isinstance(result["answer"], GroundedAnswer)
    assert result["answer"].citations == [1]
    assert result["question"] == "What do lanterns burn?"
    assert [d.page_content for d in result["docs"]] == ["Lanterns burn oil."]


def test_rag_chain_sends_numbered_sources_to_model(store, fake_llm):
    store.add_documents(
        [Document(page_content="Lanterns burn oil.", metadata={"filename": "l.txt", "page": 3})]
    )
    build_rag_chain(fake_llm, build_retriever(store, k=4)).invoke({"question": "Q?"})

    system, human = fake_llm.calls[0].to_messages()
    assert "[1] l.txt, page 3\nLanterns burn oil." in system.content
    assert human.content == "Q?"


def test_grounded_answer_schema_validates_confidence():
    with pytest.raises(ValidationError):
        GroundedAnswer(answer="x", answerable=True, confidence="certain")


def test_schema_binds_to_real_anthropic_model():
    """Builds (but never calls) a real model, catching schema problems that
    the fake model can't, such as types the provider's tool format rejects."""
    anthropic = pytest.importorskip("langchain_anthropic")
    model = anthropic.ChatAnthropic(model="claude-haiku-5-5", api_key="test-key")
    assert model.with_structured_output(GroundedAnswer) is not None