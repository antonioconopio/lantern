"""Chain tests with fake models: no real LLM calls."""

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from lantern.chains import build_qa_chain, build_rag_chain, qa_prompt, rag_prompt
from lantern.retrieval import build_retriever, format_docs


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


def test_rag_chain_returns_answer_and_retrieved_docs(store):
    store.add_documents(
        [Document(page_content="Lanterns burn oil.", metadata={"filename": "l.txt", "page": 1})]
    )
    fake = FakeListChatModel(responses=["They burn oil [1]."])
    chain = build_rag_chain(fake, build_retriever(store, k=4))

    result = chain.invoke({"question": "What do lanterns burn?"})

    assert result["answer"] == "They burn oil [1]."
    assert result["question"] == "What do lanterns burn?"
    assert [d.page_content for d in result["docs"]] == ["Lanterns burn oil."]