"""Tools the agent can call."""

import ast
import operator

from langchain_core.documents import Document
from langchain_core.tools import BaseTool, tool
from langchain_core.vectorstores import VectorStore

from lantern.ingest import list_documents
from lantern.retrieval import build_retriever, format_docs

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}
MAX_EXPONENT = 100


def _eval_node(node: ast.AST) -> int | float:
    """Walk the parsed expression, allowing only numbers and arithmetic."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError(f"exponent larger than {MAX_EXPONENT}")
        return _BINARY_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("only numbers and + - * / // % ** are allowed")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. "60 + 85" or "(12 * 3) / 4".

    Use this for any arithmetic instead of computing in your head. Supports
    + - * / // % ** and parentheses.
    """
    try:
        result = _eval_node(ast.parse(expression, mode="eval").body)
    except (SyntaxError, ValueError, ZeroDivisionError, TypeError) as e:
        return f"Error: {e}"
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return f"{expression} = {result}"


def make_tools(store: VectorStore, k: int = 4) -> list[BaseTool]:
    """Build the tools that need the vector store (the model can't pass one in)."""
    retriever = build_retriever(store, k=k)

    @tool(response_format="content_and_artifact")
    def search_documents(query: str) -> tuple[str, list[Document]]:
        """Search the user's uploaded documents for passages relevant to a query.

        Use this for any question about the documents' content. Write a
        specific, standalone query. For questions with several parts,
        search once per part.
        """
        docs = retriever.invoke(query)
        return format_docs(docs), docs

    @tool("list_documents")
    def list_documents_tool() -> str:
        """List the documents the user has uploaded."""
        docs = list_documents(store)
        if not docs:
            return "No documents have been uploaded yet."
        return "\n".join(f"- {d['filename']} ({d['chunks']} chunks)" for d in docs)

    return [search_documents, calculator, list_documents_tool]
