"""Semantic retrieval over the KAIRIX Qdrant knowledge collection."""

from __future__ import annotations

from typing import Any, Dict, List

from .embedder import SentenceTransformerEmbedder
from .qdrant_store import QdrantKnowledgeStore


class VectorRetriever:
    """Embed a natural-language query and retrieve evidence from Qdrant."""

    def __init__(
        self,
        *,
        store: QdrantKnowledgeStore | None = None,
        embedder: SentenceTransformerEmbedder | None = None,
    ) -> None:
        self.store = store or QdrantKnowledgeStore()
        self.embedder = embedder or SentenceTransformerEmbedder()
        self._owns_store = store is None

    def connect(self) -> None:
        self.store.connect()

    def close(self) -> None:
        if self._owns_store:
            self.store.close()

    def index(self, chunks: List[Dict[str, Any]]) -> int:
        if not chunks:
            return 0
        self.store.ensure_collection(self.embedder.dimension)
        vectors = self.embedder.embed([chunk["text"] for chunk in chunks])
        return self.store.upsert(chunks, vectors)

    def delete_artifact(self, artifact_id: str) -> None:
        self.store.delete_by_artifact_id(artifact_id)

    def index_artifact(self, artifact_id: str, chunks: List[Dict[str, Any]]) -> int:
        self.store.ensure_collection(self.embedder.dimension)
        self.store.delete_by_artifact_id(artifact_id)
        if not chunks:
            return 0
        vectors = self.embedder.embed([chunk["text"] for chunk in chunks])
        return self.store.upsert(chunks, vectors)

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        artifact_id: str | None = None,
        kind: str | None = None,
    ) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        vector = self.embedder.embed_one(query.strip())
        return self.store.search(vector, limit=limit, artifact_id=artifact_id, kind=kind)
