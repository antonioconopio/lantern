"""The structured answer the model must return.

with_structured_output() turns this class into a JSON schema the model is
forced to fill in. The Field descriptions are sent to the model as part of
that schema, so they double as instructions.
"""

from typing import Literal

from pydantic import BaseModel, Field


class GroundedAnswer(BaseModel):
    """An answer to the user's question, grounded in the numbered sources."""

    answer: str = Field(
        description=(
            "The answer, citing sources inline like [1] or [2][3]. If the "
            "sources don't contain the answer, say so briefly."
        )
    )
    citations: list[int] = Field(
        default_factory=list,
        description="Numbers of the sources actually used, e.g. [1, 3]. Empty if none.",
    )
    answerable: bool = Field(
        description="True only if the sources contain enough to answer the question."
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description=(
            "high: sources state the answer directly. medium: answer needs "
            "combining or interpreting sources. low: sources are only loosely related."
        )
    )