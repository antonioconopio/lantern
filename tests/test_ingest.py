"""Ingestion pipeline tests: loading, splitting, storing, listing, deleting."""

import pytest

from lantern.ingest import (
    UnsupportedFileType,
    delete_document,
    ingest_file,
    list_documents,
    load_file,
    split_documents,
)


def make_pdf(path, pages):
    """Write a simple multi-page PDF (reportlab is a test-only dependency)."""
    canvas = pytest.importorskip("reportlab.pdfgen.canvas")
    c = canvas.Canvas(str(path))
    for text in pages:
        c.drawString(72, 720, text)
        c.showPage()
    c.save()


def test_load_text_file(tmp_path):
    f = tmp_path / "notes.md"
    f.write_text("# Notes\nSome content.")
    docs = load_file(f, "notes.md", "doc1")
    assert len(docs) == 1
    assert docs[0].metadata == {"document_id": "doc1", "filename": "notes.md", "page": 1}


def test_load_pdf_gives_one_document_per_page_with_page_numbers(tmp_path):
    f = tmp_path / "guide.pdf"
    make_pdf(f, ["First page text", "Second page text"])
    docs = load_file(f, "guide.pdf", "doc1")
    assert [d.metadata["page"] for d in docs] == [1, 2]
    assert "Second page text" in docs[1].page_content


def test_load_rejects_unsupported_type(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("a,b")
    with pytest.raises(UnsupportedFileType):
        load_file(f, "data.csv", "doc1")


def test_split_keeps_metadata_on_every_chunk(tmp_path):
    f = tmp_path / "long.txt"
    f.write_text("word " * 1000)  # ~5000 characters
    docs = load_file(f, "long.txt", "doc1")
    chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(c.metadata["filename"] == "long.txt" for c in chunks)
    assert all(len(c.page_content) <= 500 for c in chunks)


def test_ingest_list_and_delete(tmp_path, store):
    f = tmp_path / "a.txt"
    f.write_text("word " * 600)
    doc_id, n = ingest_file(f, "a.txt", store, chunk_size=500, chunk_overlap=50)

    assert n > 1
    assert list_documents(store) == [{"id": doc_id, "filename": "a.txt", "chunks": n}]

    assert delete_document(store, doc_id) == n
    assert list_documents(store) == []


def test_delete_unknown_document_returns_zero(store):
    assert delete_document(store, "missing") == 0