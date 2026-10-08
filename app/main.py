"""FastAPI entrypoint"""

from fastapi import Depends, FastAPI
from langchain_core.language_models import BaseChatModel

from app.config import get_llm
from app.schemas import AskRequest, AskResponse
from lantern.chains import build_qa_chain

app = FastAPI(title="Lantern", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, llm: BaseChatModel = Depends(get_llm)) -> AskResponse:
    chain = build_qa_chain(llm)
    answer = await chain.ainvoke({"question": req.question})
    return AskResponse(answer=answer)