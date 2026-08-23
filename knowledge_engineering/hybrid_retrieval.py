"""Hybrid semantic + graph retrieval for KAIRIX."""

from __future__ import annotations

from typing import Any, Dict, List

from .env import load_environment
from .knowledge_graph import KnowledgeGraphQuery
from .vector_retrieval import VectorRetriever


class HybridRetriever:
    """Combine Qdrant semantic evidence with Neo4j graph evidence."""

    def __init__(
        self,
        *,
        vector_retriever: VectorRetriever | None = None,
        graph_query: KnowledgeGraphQuery | None = None,
    ) -> None:
        load_environment()

        self.vector = vector_retriever or VectorRetriever()
        self.graph = graph_query or KnowledgeGraphQuery()
        self._owns_vector = vector_retriever is None
        self._owns_graph = graph_query is None

    def connect(self) -> None:
        self.vector.connect()
        self.graph.store.connect()

    def close(self) -> None:
        if self._owns_vector:
            self.vector.close()
        if self._owns_graph:
            self.graph.store.close()

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        graph_hops: int = 1,
    ) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("query must not be empty")

        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")

        if not 1 <= graph_hops <= 4:
            raise ValueError("graph_hops must be between 1 and 4")

        vector_results = self.vector.search(
            query.strip(),
            limit=limit,
        )

        results: List[Dict[str, Any]] = []

        for result in vector_results:
            source_id = result.get("source_id")

            graph_evidence: List[Dict[str, Any]] = []

            if source_id:
                try:
                    graph_evidence = self.graph.traverse(
                        str(source_id),
                        hops=graph_hops,
                        limit=20,
                    )
                except Exception:
                    graph_evidence = []

            vector_score = float(result.get("score", 0.0))

            results.append(
                {
                    "id": result.get("id"),
                    "score": vector_score,
                    "text": result.get("text"),
                    "kind": result.get("kind"),
                    "source_id": source_id,
                    "artifact_id": result.get("artifact_id"),
                    "metadata": result.get("metadata", {}),
                    "graph_evidence": graph_evidence,
                    "graph_evidence_count": len(graph_evidence),
                }
            )

        return results