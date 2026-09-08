"""Qdrant wrapper: local embeddings + reranking via fastembed (no external API calls).

Embeddings and reranking run entirely on-device via ONNX (fastembed, maintained by
Qdrant itself) instead of a rate-limited API. This also lets retrieval widen its net
(RETRIEVE_CANDIDATES) and rerank down to the best few with a cross-encoder, which is
the standard fix for "vector search alone isn't precise enough."
"""

import os
import uuid

from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
EMBED_BATCH_SIZE = 32
RETRIEVE_CANDIDATES = 15  # widen the net before reranking down to top_k

_embedder: TextEmbedding | None = None
_reranker: TextCrossEncoder | None = None


def get_qdrant_client() -> QdrantClient:
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key = os.environ.get("QDRANT_API_KEY") or None
    # Default client timeout (5s) is too short for larger batch upserts over a real network.
    return QdrantClient(url=url, api_key=api_key, timeout=60)


def get_collection_name() -> str:
    return os.environ.get("QDRANT_COLLECTION", "report_agent_docs")


def get_embedder() -> TextEmbedding:
    """Lazily load and cache the local embedding model (loaded once per process)."""
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
    return _embedder


def get_reranker() -> TextCrossEncoder:
    """Lazily load and cache the local cross-encoder reranker."""
    global _reranker
    if _reranker is None:
        _reranker = TextCrossEncoder(model_name=RERANKER_MODEL)
    return _reranker


def ensure_collection(client: QdrantClient, collection: str) -> None:
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )


def upsert_documents(chunks: list[str], sources: list[str]) -> int:
    """Embed and upsert `chunks` (parallel to `sources`) into the collection. Returns count."""
    if len(chunks) != len(sources):
        raise ValueError("chunks and sources must be the same length")
    if not chunks:
        return 0

    client = get_qdrant_client()
    collection = get_collection_name()
    ensure_collection(client, collection)
    embedder = get_embedder()

    total = 0
    for start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch_chunks = chunks[start : start + EMBED_BATCH_SIZE]
        batch_sources = sources[start : start + EMBED_BATCH_SIZE]
        vectors = embedder.embed(batch_chunks)

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector.tolist(),
                payload={"text": chunk, "source": source},
            )
            for vector, chunk, source in zip(vectors, batch_chunks, batch_sources, strict=True)
        ]
        client.upsert(collection_name=collection, points=points)
        total += len(points)

    return total


def list_sources() -> list[str]:
    """Return the distinct source filenames currently stored, for display in the UI."""
    client = get_qdrant_client()
    collection = get_collection_name()
    if not client.collection_exists(collection):
        return []

    sources: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=200,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        sources.update(p.payload["source"] for p in points if p.payload and "source" in p.payload)
        if offset is None:
            break
    return sorted(sources)


def search(query: str, top_k: int = 4) -> list[dict]:
    """Vector search for RETRIEVE_CANDIDATES, then rerank down to top_k with a cross-encoder."""
    client = get_qdrant_client()
    collection = get_collection_name()
    if not client.collection_exists(collection):
        return []

    embedder = get_embedder()
    query_vector = next(iter(embedder.query_embed(query))).tolist()

    hits = client.query_points(
        collection_name=collection, query=query_vector, limit=RETRIEVE_CANDIDATES
    ).points
    if not hits:
        return []

    reranker = get_reranker()
    texts = [hit.payload["text"] for hit in hits]
    scores = reranker.rerank(query, texts)

    reranked = sorted(zip(hits, scores, strict=True), key=lambda pair: pair[1], reverse=True)

    return [
        {"text": hit.payload["text"], "source": hit.payload["source"], "score": float(score)}
        for hit, score in reranked[:top_k]
    ]
