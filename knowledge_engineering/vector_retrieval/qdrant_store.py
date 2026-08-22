"""Qdrant persistence for provenance-rich KAIRIX knowledge chunks."""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models


DEFAULT_COLLECTION = "kairix_knowledge"


class QdrantKnowledgeStore:
    """Thin, configurable Qdrant store with no domain-specific collection logic."""

    def __init__(
        self,
        *,
        url: str | None = None,
        api_key: str | None = None,
        collection_name: str | None = None,
        client: QdrantClient | None = None,
    ) -> None:
        self.url = url or os.getenv("QDRANT_URL")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION)
        self.client = client
        self._owns_client = client is None

    def connect(self) -> None:
        if self.client is not None:
            return
        if not self.url:
            raise RuntimeError("QDRANT_URL is not configured.")
        if not self.api_key:
            raise RuntimeError("QDRANT_API_KEY is not configured.")
        self.client = QdrantClient(url=self.url, api_key=self.api_key)
        self.client.get_collections()

    def close(self) -> None:
        if self.client is not None and self._owns_client:
            self.client.close()
        self.client = None if self._owns_client else self.client

    def ensure_collection(self, vector_size: int) -> None:
        self._require_client()
        assert self.client is not None
        if self.client.collection_exists(self.collection_name):
            info = self.client.get_collection(self.collection_name)
            existing_size = self._vector_size(info)
            if existing_size != vector_size:
                raise ValueError(
                    f"Collection '{self.collection_name}' expects vector size {existing_size}, "
                    f"but embedder produced {vector_size}. Use a new collection or matching model."
                )
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
        for field, schema in (
            ("artifact_id", models.PayloadSchemaType.KEYWORD),
            ("kind", models.PayloadSchemaType.KEYWORD),
            ("source_id", models.PayloadSchemaType.KEYWORD),
        ):
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=schema,
            )

    def upsert(self, chunks: Iterable[Dict[str, Any]], vectors: Iterable[List[float]]) -> int:
        self._require_client()
        assert self.client is not None
        chunk_list = list(chunks)
        vector_list = list(vectors)
        if len(chunk_list) != len(vector_list):
            raise ValueError("chunks and vectors must have the same length")
        if not chunk_list:
            return 0
        points = []
        for chunk, vector in zip(chunk_list, vector_list):
            payload = {
                "text": chunk["text"],
                "kind": chunk.get("kind"),
                "source_id": str(chunk.get("source_id")) if chunk.get("source_id") is not None else None,
                "artifact_id": str(chunk.get("artifact_id")) if chunk.get("artifact_id") is not None else None,
                "metadata": chunk.get("metadata", {}),
            }
            payload = {key: value for key, value in payload.items() if value is not None}
            points.append(models.PointStruct(id=chunk["id"], vector=vector, payload=payload))
        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
        return len(points)

    def search(
        self,
        vector: List[float],
        *,
        limit: int = 10,
        artifact_id: str | None = None,
        kind: str | None = None,
    ) -> List[Dict[str, Any]]:
        self._require_client()
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        assert self.client is not None
        filters = []
        if artifact_id:
            filters.append(models.FieldCondition(key="artifact_id", match=models.MatchValue(value=artifact_id)))
        if kind:
            filters.append(models.FieldCondition(key="kind", match=models.MatchValue(value=kind)))
        query_filter = models.Filter(must=filters) if filters else None
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        return [
            {"id": str(item.id), "score": float(item.score), **(item.payload or {})}
            for item in response.points
        ]

    def count(self) -> int:
        self._require_client()
        assert self.client is not None
        return int(self.client.count(collection_name=self.collection_name, exact=True).count)

    def _require_client(self) -> None:
        if self.client is None:
            raise RuntimeError("QdrantKnowledgeStore is not connected. Call connect() first.")

    @staticmethod
    def _vector_size(info: Any) -> int:
        config = info.config.params.vectors
        if isinstance(config, dict):
            first = next(iter(config.values()))
            return int(first.size)
        return int(config.size)
