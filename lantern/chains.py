"""LCEL chains"""

from operator import itemgetter

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableBranch, RunnablePassthrough

from lantern.answer import GroundedAnswer
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


REWRITE_SYSTEM_PROMPT = (
    "Given the conversation so far and a follow-up question, rewrite the "
    "follow-up as a standalone question that can be understood without the "
    "conversation. Replace pronouns and vague references (it, that, the second "
    "one) with what they refer to.\n"
    "Do NOT answer the question. If it is already standalone, return it "
    "unchanged. Reply with the question only."
)

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", REWRITE_SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "Follow-up question: {question}"),
    ]
)


def build_rewrite_chain(llm: BaseChatModel) -> Runnable:
    """Input: {"question": str, "history": list[BaseMessage]}  Output: str"""
    return rewrite_prompt | llm | StrOutputParser() | (lambda s: s.strip())


RAG_SYSTEM_PROMPT = (
    "You are Lantern, an assistant that answers questions using ONLY the "
    "numbered sources below.\n"
    "- Cite the sources you use inline, like [1] or [2][3], and list their "
    "numbers in `citations`.\n"
    "- If the sources don't contain the answer, set `answerable` to false and "
    "say you couldn't find it in the documents. Do not use outside knowledge.\n"
    "- Earlier messages are for understanding the question only; citation "
    "numbers in them refer to old sources, not the ones below.\n"
    "- Be concise.\n\n"
    "Sources:\n{context}"
)

rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_SYSTEM_PROMPT),
        MessagesPlaceholder("history", optional=True),
        ("human", "{question}"),
    ]
)


def build_rag_chain(llm: BaseChatModel, retriever: BaseRetriever) -> Runnable:
    """Rewrite the question if needed, retrieve, and answer with structured output.

    Input:  {"question": str, "history": list[BaseMessage] (optional)}
    Output: the input plus
            "standalone_question": str            what was sent to the retriever
            "docs": list[Document]                retrieved chunks; citation n = docs[n-1]
            "answer": GroundedAnswer
    """
    # Step 0: default history to empty so callers can leave it out.
    with_history = RunnablePassthrough.assign(history=lambda x: x.get("history") or [])

    # Step 1: decide what to search for. RunnableBranch runs the first branch
    # whose condition is true, else the default (the last argument). A first
    # question needs no rewrite, which saves a model call.
    standalone = RunnableBranch(
        (lambda x: not x["history"], itemgetter("question")),
        build_rewrite_chain(llm),
    )

    # Step 3: answer from the retrieved sources. The prompt also gets the
    # history so the answer can refer back naturally.
    generate = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"]))
        | rag_prompt
        | llm.with_structured_output(GroundedAnswer)
    )

    # Each .assign adds one key and keeps everything before it.
    return (
        with_history
        | RunnablePassthrough.assign(standalone_question=standalone)
        # Step 2: retrieve with the standalone question, not the raw follow-up.
        | RunnablePassthrough.assign(docs=itemgetter("standalone_question") | retriever)
        | RunnablePassthrough.assign(answer=generate)
    )