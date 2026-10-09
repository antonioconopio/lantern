"""Request and response models"""

from typing import Literal
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)

class Source(BaseModel):
    """One retrieved chunk. Its position in the list matches the [n] in the answer."""

    index: int
    document_id: str
    document: str
    page: int
    snippet: str


class AskResponse(BaseModel):
    answer: str
    answerable: bool
    confidence: Literal["high", "medium", "low"]
    sources: list[Source]


class DocumentInfo(BaseModel):
    id: str
    filename: str
    chunks: int