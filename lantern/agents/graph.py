"""LangGraph agent: the model decides whether to search, how often, and when to calculate."""

from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.vectorstores import VectorStore
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from lantern.agents.tools import make_tools
from lantern.answer import GroundedAnswer
from lantern.retrieval import format_docs

AGENT_SYSTEM_PROMPT = (
    "You are Lantern, a research assistant for the user's uploaded documents.\n"
    "- For any factual question, ALWAYS search with search_documents. Never "
    "answer from memory.\n"
    "- Write specific, standalone search queries, using the conversation to "
    "resolve follow-ups like 'what about that one?'.\n"
    "- Search again for questions with several parts, or when results look weak.\n"
    "- Use the calculator for any arithmetic.\n"
    "- For greetings or small talk, answer directly without tools.\n"
    "When you have what you need, stop and reply with a one-line summary; a "
    "later step writes the final cited answer."
)

RESPOND_SYSTEM_PROMPT = (
    "You are Lantern, an assistant that answers questions using ONLY the "
    "research below.\n"
    "- Cite the numbered sources you use inline, like [1] or [2][3], and list "
    "their numbers in `citations`.\n"
    "- Calculator and document-list results need no citation.\n"
    "- If the research doesn't contain the answer, set `answerable` to false and "
    "say you couldn't find it in the documents. Do not use outside knowledge.\n"
    "- For greetings or small talk, reply briefly with `answerable` true and no "
    "citations.\n"
    "- Earlier messages are for understanding the question only; citation "
    "numbers in them refer to old sources.\n"
    "- Be concise.\n\n"
    "Research:\n{research}"
)

respond_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", RESPOND_SYSTEM_PROMPT),
        MessagesPlaceholder("history", optional=True),
        ("human", "{question}"),
    ]
)


class AgentState(TypedDict):
    question: str
    history: list[BaseMessage]
    messages: Annotated[list[BaseMessage], add_messages]  # this turn's tool calls + results
    docs: list[Document]  # every unique chunk found; source n = docs[n-1]
    steps: list[dict]  # tool calls made, for the API response
    tool_calls_made: int
    answer: GroundedAnswer | None


def _doc_key(doc: Document) -> tuple:
    m = doc.metadata
    return (m.get("document_id"), m.get("page"), m.get("start_index"), doc.page_content[:50])


def number_new_docs(found: list[Document], docs: list[Document]) -> str:
    """Add unseen chunks to `docs` (in place) and format `found` with global numbers.

    A chunk found twice keeps its first number, so [n] always means docs[n-1].
    """
    index = {_doc_key(d): i for i, d in enumerate(docs, start=1)}
    lines = []
    for d in found:
        key = _doc_key(d)
        if key not in index:
            docs.append(d)
            index[key] = len(docs)
        n = index[key]
        lines.append(
            f"[{n}] {d.metadata.get('filename', 'unknown')}, page {d.metadata.get('page', '?')}\n"
            f"{d.page_content}"
        )
    return "\n\n".join(lines) if lines else format_docs([])


def render_research(messages: list[BaseMessage]) -> str:
    """Turn this turn's tool calls and results into plain text for the respond step."""
    parts = []
    for m in messages:
        if isinstance(m, AIMessage):
            for call in m.tool_calls:
                args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
                parts.append(f"Called {call['name']}({args})")
            if m.text:
                parts.append(f"Research notes: {m.text}")
        elif isinstance(m, ToolMessage):
            parts.append(f"Result:\n{m.content}")
    return "\n\n".join(parts) or "(No research was done.)"


def build_agent_graph(llm: BaseChatModel, store: VectorStore, k: int, max_tool_calls: int):
    tools = make_tools(store, k=k)
    tools_by_name = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)
    answerer = llm.with_structured_output(GroundedAnswer, method="json_schema")

    async def agent(state: AgentState) -> dict:
        prompt = [
            SystemMessage(AGENT_SYSTEM_PROMPT),
            *state["history"],
            HumanMessage(state["question"]),
            *state["messages"],
        ]
        return {"messages": [await llm_with_tools.ainvoke(prompt)]}

    async def run_tools(state: AgentState) -> dict:
        calls = state["messages"][-1].tool_calls
        docs = list(state["docs"])  # copy: only `messages` has an appending reducer
        steps = list(state["steps"])
        results = []
        for call in calls:
            steps.append({"tool": call["name"], "input": call["args"]})
            tool = tools_by_name.get(call["name"])
            try:
                if tool is None:
                    raise ValueError(f"unknown tool {call['name']}")
                msg = await tool.ainvoke({**call, "type": "tool_call"})
                if call["name"] == "search_documents":
                    msg.content = number_new_docs(msg.artifact, docs)
            except Exception as e:  # every tool call needs a result or the API rejects the next request
                msg = ToolMessage(f"Error: {e}", tool_call_id=call["id"])
            results.append(msg)
        return {
            "messages": results,
            "docs": docs,
            "steps": steps,
            "tool_calls_made": state["tool_calls_made"] + len(calls),
        }

    def route(state: AgentState) -> Literal["tools", "respond"]:
        wants_tools = bool(state["messages"][-1].tool_calls)
        return "tools" if wants_tools and state["tool_calls_made"] < max_tool_calls else "respond"

    async def respond(state: AgentState) -> dict:
        prompt = await respond_prompt.ainvoke(
            {
                "research": render_research(state["messages"]),
                "history": state["history"],
                "question": state["question"],
            }
        )
        return {"answer": await answerer.ainvoke(prompt)}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", run_tools)
    graph.add_node("respond", respond)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, ["tools", "respond"])
    graph.add_edge("tools", "agent")  # the loop
    graph.add_edge("respond", END)
    return graph.compile()


def build_agent_chain(
    llm: BaseChatModel, store: VectorStore, k: int = 4, max_tool_calls: int = 6
) -> Runnable:
    """Same input and output shape as build_rag_chain, plus "steps".

    Input:  {"question": str, "history": list[BaseMessage] (optional)}
    Output: question, history, docs, answer, steps, and
            standalone_question (first search query, else the question)
    """
    graph = build_agent_graph(llm, store, k, max_tool_calls)

    async def run(inputs: dict) -> dict:
        question, history = inputs["question"], inputs.get("history") or []
        state = await graph.ainvoke(
            {
                "question": question,
                "history": history,
                "messages": [],
                "docs": [],
                "steps": [],
                "tool_calls_made": 0,
                "answer": None,
            }
        )
        searches = [s for s in state["steps"] if s["tool"] == "search_documents"]
        return {
            "question": question,
            "history": history,
            "docs": state["docs"],
            "answer": state["answer"],
            "steps": state["steps"],
            "standalone_question": searches[0]["input"]["query"] if searches else question,
        }

    return RunnableLambda(run)
