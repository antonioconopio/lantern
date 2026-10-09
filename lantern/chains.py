"""LCEL chains"""

from operator import itemgetter

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough

from lantern.retrieval import format_docs

SYSTEM_PROMPT = (
    "You are Lantern, a helpful assistant that answers questions clearly "
    "and concisely. If you don't know the answer, say so instead of guessing."
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


def build_qa_chain(llm: BaseChatModel) -> Runnable:
    """Input: {"question": str}  Output: str"""
    return qa_prompt | llm | StrOutputParser()


RAG_SYSTEM_PROMPT = (
    "You are Lantern, an assistant that answers questions using ONLY the "
    "numbered sources below.\n"
    "- Cite the sources you use inline, like [1] or [2][3].\n"
    "- If the sources don't contain the answer, say you couldn't find it in "
    "the documents. Do not use outside knowledge.\n"
    "- Be concise.\n\n"
    "Sources:\n{context}"
)

rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


def build_rag_chain(llm: BaseChatModel, retriever: BaseRetriever) -> Runnable:
    """Retrieve chunks, answer from them, and return both.

    Input:  {"question": str}
    Output: {"question": str, "docs": list[Document], "answer": str}

    Returning the docs alongside the answer is what lets the API build
    citations: the [n] markers in the answer map to docs[n-1].
    """
    # Run retrieval, keeping the question alongside the results.
    retrieve = RunnableParallel(
        question=itemgetter("question"),
        docs=itemgetter("question") | retriever,
    )

    # Format the docs into the prompt and generate the answer.
    generate = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"]))
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    # .assign adds "answer" while keeping "question" and "docs" in the output.
    return retrieve | RunnablePassthrough.assign(answer=generate)