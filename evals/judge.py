"""LLM-as-judge: decides whether an answer matches the reference answer.

Exact string matching can't grade free-text answers ("$40/year" vs "forty
dollars annually"), so a model compares meaning instead. Judges make mistakes
too: read the reasons for failed cases rather than trusting the score blindly.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field


class JudgeVerdict(BaseModel):
    # reason comes first so the model explains before it decides.
    reason: str = Field(description="One or two sentences explaining the verdict.")
    correct: bool = Field(description="True if the candidate answer is correct.")


JUDGE_SYSTEM_PROMPT = (
    "You grade answers from a document question-answering system.\n"
    "Compare the candidate answer to the reference answer.\n"
    "- Mark it correct if it contains the key facts of the reference answer. "
    "Extra details are fine as long as they don't contradict the reference.\n"
    "- Mark it incorrect if it misses a key fact, contradicts the reference, "
    "or says the information isn't available when the reference has it.\n"
    "- Ignore citation markers like [1] and differences in wording or format."
)

judge_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", JUDGE_SYSTEM_PROMPT),
        (
            "human",
            "Question: {question}\n\nReference answer: {expected_answer}\n\n"
            "Candidate answer: {answer}",
        ),
    ]
)


def build_judge(llm: BaseChatModel) -> Runnable:
    """Input: {"question", "expected_answer", "answer"}  Output: JudgeVerdict"""
    return judge_prompt | llm.with_structured_output(JudgeVerdict, method="json_schema")