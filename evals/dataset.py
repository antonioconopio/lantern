"""Eval case format and loader.

Each line of dataset.jsonl is one case:
    id                unique name, used to compare runs
    question          what the user asks
    history           optional earlier turns, for testing follow-up questions
    expected_answer   reference answer the judge compares against
    expected_sources  the document + page(s) the answer comes from
    answerable        false when the documents don't contain the answer
    tags              for slicing results (single, multi, unanswerable, follow-up, ...)
"""

import json
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field, model_validator

DATASET_PATH = Path(__file__).parent / "dataset.jsonl"
CORPUS_DIR = Path(__file__).parent / "corpus"


class SourceRef(BaseModel):
    document: str
    page: int

    def key(self) -> tuple[str, int]:
        return (self.document, self.page)


class Turn(BaseModel):
    question: str
    answer: str


class EvalCase(BaseModel):
    id: str
    question: str
    expected_answer: str
    expected_sources: list[SourceRef] = Field(default_factory=list)
    answerable: bool = True
    history: list[Turn] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _sources_match_answerable(self):
        if self.answerable and not self.expected_sources:
            raise ValueError(f"{self.id}: answerable cases need expected_sources")
        if not self.answerable and self.expected_sources:
            raise ValueError(f"{self.id}: unanswerable cases can't have expected_sources")
        return self

    def history_messages(self) -> list[BaseMessage]:
        messages: list[BaseMessage] = []
        for turn in self.history:
            messages += [HumanMessage(turn.question), AIMessage(turn.answer)]
        return messages


def load_dataset(path: Path = DATASET_PATH) -> list[EvalCase]:
    cases = [
        EvalCase.model_validate(json.loads(line))
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    ids = [c.id for c in cases]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"Duplicate case ids: {sorted(duplicates)}")
    return cases