"""Thin wrapper around Qdrant: embed with Gemini, upsert, and similarity-search."""

import os
import uuid

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIM = 768  # gemini-embedding-001 defaults to 3072; truncated via output_dimensionality


def get_qdrant_client() -> QdrantClient:
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key = os.environ.get("QDRANT_API_KEY") or None
    return QdrantClient(url=url, api_key=api_key)


def get_collection_name() -> str:
    return os.environ.get("QDRANT_COLLECTION", "report_agent_docs")


def get_embedder() -> GoogleGenerativeAIEmbeddings:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in the environment (.env)")
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL, google_api_key=api_key, output_dimensionality=EMBEDDING_DIM
    )


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

    client = get_qdrant_client()
    collection = get_collection_name()
    ensure_collection(client, collection)

    embedder = get_embedder()
    vectors = embedder.embed_documents(chunks)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={"text": chunk, "source": source},
        )
        for vector, chunk, source in zip(vectors, chunks, sources, strict=True)
    ]
    client.upsert(collection_name=collection, points=points)
    return len(points)


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
    """Return top_k matches as [{"text", "source", "score"}, ...]."""
    client = get_qdrant_client()
    collection = get_collection_name()
    if not client.collection_exists(collection):
        return []

    embedder = get_embedder()
    query_vector = embedder.embed_query(query)

    hits = client.query_points(
        collection_name=collection, query=query_vector, limit=top_k
    ).points

    return [
        {"text": hit.payload["text"], "source": hit.payload["source"], "score": hit.score}
        for hit in hits
    ]
