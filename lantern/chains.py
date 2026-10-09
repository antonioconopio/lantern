"""LCEL chains"""

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

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
    """Build the basic QA chain.

    The model is passed in rather than created here, so tests can swap in a
    fake model and later stages can reuse the same chain with any provider.

    Input:  {"question": str}
    Output: str
    """
    return qa_prompt | llm | StrOutputParser()