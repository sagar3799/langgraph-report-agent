"""Qdrant wrapper: local embeddings + reranking via fastembed (no external API calls).

Embeddings and reranking run entirely on-device via ONNX (fastembed, maintained by
Qdrant itself) instead of a rate-limited API. This also lets retrieval widen its net
(RETRIEVE_CANDIDATES) and rerank down to the best few with a cross-encoder, which is
the standard fix for "vector search alone isn't precise enough."
"""

import logging
import os
import time
import uuid
from collections.abc import Callable

from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
# Measured ~0.175ms/char of embedding cost on CPU regardless of batch size, so this is purely
# about progress-update granularity (smaller batch = more frequent callback ticks), not throughput.
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
        logger.info("Loading embedding model %s (first call downloads it)...", EMBEDDING_MODEL)
        t0 = time.monotonic()
        _embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
        logger.info("Embedding model ready in %.1fs", time.monotonic() - t0)
    return _embedder


def get_reranker() -> TextCrossEncoder:
    """Lazily load and cache the local cross-encoder reranker."""
    global _reranker
    if _reranker is None:
        logger.info("Loading local reranker model %s (first call downloads it)...", RERANKER_MODEL)
        t0 = time.monotonic()
        _reranker = TextCrossEncoder(model_name=RERANKER_MODEL)
        logger.info("Reranker model ready in %.1fs", time.monotonic() - t0)
    return _reranker


def ensure_collection(client: QdrantClient, collection: str) -> None:
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )


def upsert_documents(
    chunks: list[str],
    sources: list[str],
    progress_callback: Callable[[int, int], None] | None = None,
) -> int:
    """Embed and upsert `chunks` (parallel to `sources`) into the collection. Returns count.

    progress_callback(chunks_done, chunks_total), if given, is called after each batch --
    embedding a large document can take minutes on CPU, so callers (e.g. the chat UI) can
    show real progress instead of a plain spinner.
    """
    if len(chunks) != len(sources):
        raise ValueError("chunks and sources must be the same length")
    if not chunks:
        return 0

    client = get_qdrant_client()
    collection = get_collection_name()
    ensure_collection(client, collection)
    embedder = get_embedder()

    n_batches = (len(chunks) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
    logger.info(
        "Embedding %d chunks in %d batch(es) of up to %d...",
        len(chunks), n_batches, EMBED_BATCH_SIZE,
    )

    total = 0
    for batch_num, start in enumerate(range(0, len(chunks), EMBED_BATCH_SIZE), start=1):
        batch_chunks = chunks[start : start + EMBED_BATCH_SIZE]
        batch_sources = sources[start : start + EMBED_BATCH_SIZE]

        t0 = time.monotonic()
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
        logger.info(
            "  batch %d/%d: embedded + upserted %d chunks in %.2fs",
            batch_num, n_batches, len(points), time.monotonic() - t0,
        )
        if progress_callback is not None:
            progress_callback(total, len(chunks))

    logger.info("Upsert complete: %d chunks now in collection '%s'", total, collection)
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
        logger.warning("Collection '%s' doesn't exist yet — returning no results", collection)
        return []

    logger.info("Embedding query locally: %r", query)
    embedder = get_embedder()
    t0 = time.monotonic()
    query_vector = next(iter(embedder.query_embed(query))).tolist()
    logger.info("  query embedded in %.2fs", time.monotonic() - t0)

    t0 = time.monotonic()
    hits = client.query_points(
        collection_name=collection, query=query_vector, limit=RETRIEVE_CANDIDATES
    ).points
    logger.info(
        "Qdrant vector search: %d candidate(s) from '%s' in %.2fs",
        len(hits), collection, time.monotonic() - t0,
    )
    if not hits:
        return []

    t0 = time.monotonic()
    reranker = get_reranker()
    texts = [hit.payload["text"] for hit in hits]
    scores = reranker.rerank(query, texts)

    reranked = sorted(zip(hits, scores, strict=True), key=lambda pair: pair[1], reverse=True)
    logger.info(
        "Reranked %d candidates in %.2fs, keeping top %d (scores: %s)",
        len(hits), time.monotonic() - t0, top_k,
        [round(float(s), 2) for _, s in reranked[:top_k]],
    )

    return [
        {"text": hit.payload["text"], "source": hit.payload["source"], "score": float(score)}
        for hit, score in reranked[:top_k]
    ]
