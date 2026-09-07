import gzip

import pytest

from agent import ingestion
from agent.ingestion import (
    EmptyDocumentError,
    UnsupportedFileTypeError,
    chunk_text,
    extract_text,
    ingest_uploaded_file,
)


def test_extract_text_from_txt():
    assert extract_text("notes.txt", b"hello world") == "hello world"


def test_extract_text_from_md_is_treated_as_plain_text():
    assert extract_text("readme.md", b"# Title\n\nbody") == "# Title\n\nbody"


def test_extract_text_rejects_unsupported_extension():
    with pytest.raises(UnsupportedFileTypeError):
        extract_text("archive.zip", b"whatever")


def test_extract_text_dispatches_pdf(monkeypatch):
    monkeypatch.setattr(ingestion, "_extract_pdf", lambda data: "pdf text")
    assert extract_text("doc.pdf", b"fake-pdf-bytes") == "pdf text"


def test_extract_text_dispatches_pptx(monkeypatch):
    monkeypatch.setattr(ingestion, "_extract_pptx", lambda data: "slide text")
    assert extract_text("deck.pptx", b"fake-pptx-bytes") == "slide text"


def test_extract_text_dispatches_docx(monkeypatch):
    monkeypatch.setattr(ingestion, "_extract_docx", lambda data: "doc text")
    assert extract_text("memo.docx", b"fake-docx-bytes") == "doc text"


def test_chunk_text_respects_overlap():
    chunks = chunk_text("a" * 1000, size=800, overlap=100)
    assert len(chunks) == 2
    assert len(chunks[0]) == 800


def test_ingest_uploaded_file_raises_on_empty_text(monkeypatch):
    monkeypatch.setattr(ingestion, "extract_text", lambda name, data: "   ")
    with pytest.raises(EmptyDocumentError):
        ingest_uploaded_file("blank.txt", b"")


def test_ingest_uploaded_file_archives_compressed_copy_and_upserts(monkeypatch, tmp_path):
    monkeypatch.setattr(ingestion, "UPLOADS_DIR", tmp_path)
    captured = {}

    def fake_upsert(chunks, sources):
        captured["chunks"] = chunks
        captured["sources"] = sources
        return len(chunks)

    monkeypatch.setattr(ingestion, "upsert_documents", fake_upsert)

    result = ingest_uploaded_file("notes.txt", b"some real content here")

    assert result["chunks"] == 1
    assert captured["sources"] == ["notes.txt"]

    archive = tmp_path / "notes.txt.gz"
    assert archive.exists()
    with gzip.open(archive, "rt", encoding="utf-8") as f:
        assert f.read() == "some real content here"
