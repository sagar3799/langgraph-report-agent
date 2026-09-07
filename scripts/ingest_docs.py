"""One-off script: chunk everything in docs/ and upsert into Qdrant.

Usage:
    python scripts/ingest_docs.py
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent.retrieval.qdrant_store import upsert_documents  # noqa: E402

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]


def main() -> None:
    load_dotenv()

    all_chunks: list[str] = []
    all_sources: list[str] = []

    doc_paths = sorted(DOCS_DIR.glob("*.md"))
    if not doc_paths:
        print(f"No .md files found in {DOCS_DIR}")
        return

    for path in doc_paths:
        text = path.read_text(encoding="utf-8")
        for chunk in chunk_text(text):
            all_chunks.append(chunk)
            all_sources.append(path.name)

    count = upsert_documents(all_chunks, all_sources)
    print(f"Upserted {count} chunks from {len(doc_paths)} files: {[p.name for p in doc_paths]}")


if __name__ == "__main__":
    main()
