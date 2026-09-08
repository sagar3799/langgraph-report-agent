"""Turn a raw uploaded file (txt/md/pdf/pptx/docx) into chunks in the vector store.

We deliberately do NOT keep the original file bytes around. For retrieval, only the
extracted text ever matters — a PDF's fonts/images are dead weight once the text is
out, so we gzip just the extracted text for a local audit trail (a fraction of the
original size) and let the binary go.
"""

import gzip
import io
from collections.abc import Callable
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from agent.retrieval.qdrant_store import upsert_documents

# ~2000 chars/~500 tokens matches the 2026 chunking-benchmark consensus default; recursive
# splitting (paragraph -> line -> word -> char fallback) beats naive fixed-size slicing by
# not cutting chunks mid-sentence.
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

# Local CPU embedding cost is ~linear in total characters processed (measured: ~0.175ms/char
# on an 8-core/16-thread Ryzen 7 laptop, regardless of how text is chunked -- multiprocessing
# only bought ~25%, not a multiple of core count). So unlike the old Gemini-quota-based cap,
# this one exists to keep an *interactive chat upload* from blocking the UI for many minutes.
# ~500 chunks is roughly a 1-3 minute worst case. For a document you want fully indexed
# regardless of how long it takes, put it in docs/ and run scripts/ingest_docs.py instead --
# that path passes max_chunks=None (unlimited) since it's an offline batch job, not a chat wait.
MAX_CHUNKS_INTERACTIVE = 500
UPLOADS_DIR = Path(__file__).resolve().parent.parent.parent / "uploads"

SUPPORTED_EXTENSIONS = ("txt", "md", "pdf", "pptx", "docx")

_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


class UnsupportedFileTypeError(ValueError):
    pass


class EmptyDocumentError(ValueError):
    pass


def chunk_text(text: str) -> list[str]:
    return [c.strip() for c in _splitter.split_text(text) if c.strip()]


def extract_text(filename: str, data: bytes) -> str:
    """Extract plain text from raw file bytes based on its extension."""
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if suffix in ("txt", "md"):
        return data.decode("utf-8", errors="replace")
    if suffix == "pdf":
        return _extract_pdf(data)
    if suffix == "pptx":
        return _extract_pptx(data)
    if suffix == "docx":
        return _extract_docx(data)
    raise UnsupportedFileTypeError(
        f"Unsupported file type '.{suffix}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
    )


def _extract_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def _extract_pptx(data: bytes) -> str:
    from pptx import Presentation

    prs = Presentation(io.BytesIO(data))
    parts = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                parts.append(shape.text_frame.text)
    return "\n\n".join(parts)


def _extract_docx(data: bytes) -> str:
    from docx import Document

    doc = Document(io.BytesIO(data))
    return "\n\n".join(p.text for p in doc.paragraphs)


def save_compressed_text(filename: str, text: str) -> Path:
    UPLOADS_DIR.mkdir(exist_ok=True)
    out_path = UPLOADS_DIR / f"{Path(filename).stem}.txt.gz"
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        f.write(text)
    return out_path


def ingest_uploaded_file(
    filename: str,
    data: bytes,
    max_chunks: int | None = MAX_CHUNKS_INTERACTIVE,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    """Extract, locally archive (compressed), chunk, and upsert one uploaded file.

    max_chunks=None means no cap (used by the offline batch script); the interactive
    chat upload path defaults to MAX_CHUNKS_INTERACTIVE so it can't block the UI for
    many minutes on a huge document. progress_callback(done, total), if given, is
    called after each embedding batch.
    """
    text = extract_text(filename, data)
    if not text.strip():
        raise EmptyDocumentError(f"No extractable text found in {filename}")

    archive_path = save_compressed_text(filename, text)
    all_chunks = chunk_text(text)
    truncated = max_chunks is not None and len(all_chunks) > max_chunks
    chunks = all_chunks[:max_chunks] if max_chunks is not None else all_chunks
    chunk_count = upsert_documents(
        chunks, [filename] * len(chunks), progress_callback=progress_callback
    )

    return {
        "filename": filename,
        "chunks": chunk_count,
        "total_chunks_found": len(all_chunks),
        "truncated": truncated,
        "original_bytes": len(data),
        "archived_bytes": archive_path.stat().st_size,
    }
