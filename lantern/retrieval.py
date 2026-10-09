"""Retriever setup and helpers for turning retrieved chunks into prompt context."""

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore


def build_retriever(store: VectorStore, k: int = 4) -> BaseRetriever:
    """Similarity-search retriever returning the top-k chunks for a query."""
    return store.as_retriever(search_kwargs={"k": k})


def format_docs(docs: list[Document]) -> str:
    """Number each chunk so the model can cite it as [1], [2], ..."""
    if not docs:
        return "(no relevant documents found)"
    return "\n\n".join(
        f"[{i}] {d.metadata.get('filename', 'unknown')}, page {d.metadata.get('page', '?')}\n"
        f"{d.page_content}"
        for i, d in enumerate(docs, start=1)
    )