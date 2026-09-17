"""Qdrant wrapper: local embeddings + reranking via fastembed (no external API calls).

Embeddings and reranking run entirely on-device via ONNX (fastembed, maintained by
Qdrant itself) instead of a rate-limited API. This also lets retrieval widen its net
(RETRIEVE_CANDIDATES) and rerank down to the best few with a cross-encoder, which is
the standard fix for "vector search alone isn't precise enough."
"""

import logging
import math
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


def _sigmoid(x: float) -> float:
    """Squash an unbounded cross-encoder score into [0,1] -- a heuristic confidence signal,
    not a calibrated probability (raw scores run roughly -10 to +7 in testing)."""
    return 1.0 / (1.0 + math.exp(-x))


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


def search(query: str, top_k: int = 4, extra_candidates: list[dict] | None = None) -> list[dict]:
    """Vector search for RETRIEVE_CANDIDATES, merge in `extra_candidates`, then rerank down to
    top_k with a cross-encoder.

    `extra_candidates` (each needs "id", "text", "source") lets a caller fold in chunks already
    retrieved elsewhere in the same run -- e.g. a multi-section research run's shared evidence
    pool -- so they compete fairly for the top_k slots instead of triggering a second, separate
    reranking. Candidates already present in the fresh Qdrant hits (same id) aren't duplicated.
    """
    client = get_qdrant_client()
    collection = get_collection_name()

    ids: list[str] = []
    texts: list[str] = []
    sources: list[str] = []

    if not client.collection_exists(collection):
        logger.warning(
            "Collection '%s' doesn't exist yet — using only extra candidates", collection
        )
    else:
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
        ids = [str(hit.id) for hit in hits]
        texts = [hit.payload["text"] for hit in hits]
        sources = [hit.payload["source"] for hit in hits]

    seen_ids = set(ids)
    merged_from_memory = 0
    for cand in extra_candidates or []:
        if cand["id"] in seen_ids:
            continue
        ids.append(cand["id"])
        texts.append(cand["text"])
        sources.append(cand["source"])
        seen_ids.add(cand["id"])
        merged_from_memory += 1
    if merged_from_memory:
        logger.info("Merged %d candidate(s) from shared research memory", merged_from_memory)

    if not texts:
        return []

    t0 = time.monotonic()
    reranker = get_reranker()
    scores = reranker.rerank(query, texts)

    reranked = sorted(
        zip(ids, texts, sources, scores, strict=True), key=lambda row: row[3], reverse=True
    )
    logger.info(
        "Reranked %d candidates in %.2fs, keeping top %d (scores: %s)",
        len(texts), time.monotonic() - t0, top_k,
        [round(float(s), 2) for *_, s in reranked[:top_k]],
    )

    return [
        {
            "id": doc_id,
            "text": text,
            "source": source,
            "score": float(score),
            "normalized_confidence": _sigmoid(float(score)),
        }
        for doc_id, text, source, score in reranked[:top_k]
    ]
