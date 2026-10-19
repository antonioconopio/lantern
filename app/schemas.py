"""Request and response models"""

from typing import Literal
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = Field(
        default=None,
        max_length=100,
        description="Omit to start a new conversation; send back the returned id for follow-ups.",
    )
    mode: Literal["rag", "agent"] = Field(
        default="rag",
        description="rag: fixed rewrite-retrieve-answer pipeline. agent: the model chooses its own tool calls.",
    )


class Source(BaseModel):
    """One retrieved chunk. Its position in the list matches the [n] in the answer."""

    index: int
    document_id: str
    document: str
    page: int
    snippet: str


class ToolStep(BaseModel):
    tool: str
    input: dict


class AskResponse(BaseModel):
    answer: str
    answerable: bool
    confidence: Literal["high", "medium", "low"]
    sources: list[Source]
    session_id: str
    standalone_question: str = Field(
        description="The question as sent to the retriever, after rewriting follow-ups."
    )
    mode: Literal["rag", "agent"]
    steps: list[ToolStep]


class DocumentInfo(BaseModel):
    id: str
    filename: str
    chunks: int