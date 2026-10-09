"""Ingestion pipeline"""

import uuid
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}


class UnsupportedFileType(ValueError):
    pass


def load_file(path: Path, filename: str, document_id: str) -> list[Document]:
    """Load a file into one Document per page (one Document for text files)."""
    ext = Path(filename).suffix.lower()
    base = {"document_id": document_id, "filename": filename}

    if ext == ".pdf":
        reader = PdfReader(path)
        docs = []
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:  # skip blank or image-only pages
                docs.append(Document(page_content=text, metadata={**base, "page": i}))
        return docs

    if ext in {".md", ".txt"}:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        return [Document(page_content=text, metadata={**base, "page": 1})] if text else []

    raise UnsupportedFileType(f"Unsupported file type: {ext or 'none'}")


def split_documents(
    docs: list[Document], chunk_size: int = 1000, chunk_overlap: int = 150
) -> list[Document]:
    """Split pages into overlapping chunks. Metadata is copied onto each chunk."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,  # character offset of the chunk within its page
    )
    return splitter.split_documents(docs)


def ingest_file(
    path: Path,
    filename: str,
    store: VectorStore,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> tuple[str, int]:
    """Load, split and store a file. Returns (document_id, number_of_chunks)."""
    document_id = uuid.uuid4().hex
    pages = load_file(path, filename, document_id)
    chunks = split_documents(pages, chunk_size, chunk_overlap)
    if chunks:
        ids = [f"{document_id}:{i}" for i in range(len(chunks))]
        store.add_documents(chunks, ids=ids)  # embeds each chunk, then stores it
    return document_id, len(chunks)


def list_documents(store) -> list[dict]:
    """Summarise stored documents: id, filename and chunk count."""
    result = store.get(include=["metadatas"])
    docs: dict[str, dict] = {}
    for meta in result["metadatas"]:
        entry = docs.setdefault(
            meta["document_id"],
            {"id": meta["document_id"], "filename": meta["filename"], "chunks": 0},
        )
        entry["chunks"] += 1
    return sorted(docs.values(), key=lambda d: d["filename"])


def delete_document(store, document_id: str) -> int:
    """Remove every chunk of a document. Returns how many chunks were deleted."""
    ids = store.get(where={"document_id": document_id})["ids"]
    if ids:
        store.delete(ids=ids)
    return len(ids)