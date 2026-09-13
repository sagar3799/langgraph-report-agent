"""Run at Docker build time (not at container startup) to bake the fastembed models into the
image. Without this, the first request after every cold start pays a ~350MB download penalty
on top of the unavoidable model-load time -- this removes the download half of that cost.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent.retrieval.qdrant_store import EMBEDDING_MODEL, RERANKER_MODEL  # noqa: E402


def main() -> None:
    from fastembed import TextEmbedding
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    print(f"Prefetching embedding model {EMBEDDING_MODEL}...")
    TextEmbedding(model_name=EMBEDDING_MODEL)
    print(f"Prefetching reranker model {RERANKER_MODEL}...")
    TextCrossEncoder(model_name=RERANKER_MODEL)
    print("Done -- both models cached into the image.")


if __name__ == "__main__":
    main()
