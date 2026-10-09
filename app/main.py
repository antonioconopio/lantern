"""FastAPI entrypoint"""

import shutil
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

from app.config import Settings, get_llm, get_settings, get_vector_store
from app.schemas import AskRequest, AskResponse, DocumentInfo, Source
from lantern.chains import build_rag_chain
from lantern.ingest import (
    SUPPORTED_EXTENSIONS,
    UnsupportedFileType,
    delete_document,
    ingest_file,
    list_documents,
)
from lantern.retrieval import build_retriever

app = FastAPI(title="Lantern", version="0.2.0")

SNIPPET_CHARS = 300


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/documents", response_model=DocumentInfo, status_code=201)
def upload_document(
    file: UploadFile,
    store: VectorStore = Depends(get_vector_store),
    settings: Settings = Depends(get_settings),
) -> DocumentInfo:
    filename = Path(file.filename or "").name  # strip any client-sent path
    if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            415, f"Supported file types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / filename
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    try:
        doc_id, n_chunks = ingest_file(
            dest, filename, store, settings.chunk_size, settings.chunk_overlap
        )
    except UnsupportedFileType as e:
        raise HTTPException(415, str(e))

    if n_chunks == 0:
        raise HTTPException(422, "No extractable text found (scanned PDFs need OCR).")
    return DocumentInfo(id=doc_id, filename=filename, chunks=n_chunks)


@app.get("/documents", response_model=list[DocumentInfo])
def get_documents(store: VectorStore = Depends(get_vector_store)) -> list[DocumentInfo]:
    return [DocumentInfo(**d) for d in list_documents(store)]


@app.delete("/documents/{document_id}", status_code=204)
def remove_document(document_id: str, store: VectorStore = Depends(get_vector_store)):
    if delete_document(store, document_id) == 0:
        raise HTTPException(404, "Document not found")


@app.post("/ask", response_model=AskResponse)
async def ask(
    req: AskRequest,
    llm: BaseChatModel = Depends(get_llm),
    store: VectorStore = Depends(get_vector_store),
    settings: Settings = Depends(get_settings),
) -> AskResponse:
    retriever = build_retriever(store, k=settings.retrieval_k)
    chain = build_rag_chain(llm, retriever)
    result = await chain.ainvoke({"question": req.question})
    answer, docs = result["answer"], result["docs"]
 
    # Return only the sources the model says it used. Ignore citation numbers
    # that don't match a retrieved chunk: models occasionally invent them.
    cited = sorted({n for n in answer.citations if 1 <= n <= len(docs)})
    sources = [
        Source(
            index=n,
            document_id=docs[n - 1].metadata["document_id"],
            document=docs[n - 1].metadata["filename"],
            page=docs[n - 1].metadata["page"],
            snippet=docs[n - 1].page_content[:SNIPPET_CHARS],
        )
        for n in cited
    ]
    return AskResponse(
        answer=answer.answer,
        answerable=answer.answerable,
        confidence=answer.confidence,
        sources=sources,
    )