"""One-off script: chunk everything in docs/ and upsert into Qdrant.

Usage:
    python scripts/ingest_docs.py
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent.ingestion import SUPPORTED_EXTENSIONS, ingest_uploaded_file  # noqa: E402
from agent.logging_config import configure_logging  # noqa: E402

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"


def main() -> None:
    load_dotenv()
    configure_logging()

    doc_paths = sorted(
        p for ext in SUPPORTED_EXTENSIONS for p in DOCS_DIR.glob(f"*.{ext}")
    )
    if not doc_paths:
        print(f"No supported files ({', '.join(SUPPORTED_EXTENSIONS)}) found in {DOCS_DIR}")
        return

    for path in doc_paths:

        def on_progress(done: int, total: int, _name: str = path.name) -> None:
            print(f"  {_name}: {done}/{total} chunks embedded", end="\r")

        result = ingest_uploaded_file(
            path.name, path.read_bytes(), max_chunks=None, progress_callback=on_progress
        )
        print(f"{path.name}: {result['chunks']} chunks upserted" + " " * 20)


if __name__ == "__main__":
    main()
