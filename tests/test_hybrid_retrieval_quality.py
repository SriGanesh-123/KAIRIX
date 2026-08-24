from __future__ import annotations

from typing import Any

from knowledge_engineering.hybrid_retrieval import HybridRetriever


class FakeVector:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def connect(self) -> None:
        pass

    def close(self) -> None:
        pass

    def search(self, query: str, *, limit: int, **_: Any) -> list[dict[str, Any]]:
        self.calls.append((query, limit))
        return [
            {
                "id": "vector-1",
                "score": 0.80 if query.startswith("What is") else 0.70,
                "text": "Semantic transaction evidence",
                "kind": "entity",
                "source_id": "entity:transaction",
                "artifact_id": "artifact:claimcenter",
                "metadata": {},
            }
        ]


class FakeStore:
    def connect(self) -> None:
        pass

    def close(self) -> None:
        pass

    def _require_connection(self) -> None:
        pass


class FakeGraph:
    def __init__(self) -> None:
        self.store = FakeStore()
        self.search_calls: list[str] = []
        self.traverse_calls: list[str] = []

    def search_nodes(self, text: str, limit: int = 20) -> list[dict[str, Any]]:
        self.search_calls.append(text)
        if text == "policy":
            return [
                {
                    "id": "entity:policy",
                    "name": "Policy",
                    "type": "TABLE",
                    "artifact_id": "artifact:claimcenter",
                }
            ]
        return []

    def traverse(self, node_id: str, hops: int = 1, limit: int = 20) -> list[dict[str, Any]]:
        self.traverse_calls.append(node_id)
        if node_id == "entity:transaction":
            return [
                {
                    "nodes": [
                        {"id": "entity:transaction", "name": "Transaction", "type": "TABLE"},
                        {"id": "entity:policy", "name": "Policy", "type": "TABLE"},
                    ],
                    "relationships": [
                        {
                            "id": "rel-1",
                            "relationship_type": "REFERENCES",
                            "validation_status": "SUPPORTED",
                        }
                    ],
                }
            ]
        if node_id == "entity:policy":
            return [
                {
                    "nodes": [
                        {"id": "entity:policy", "name": "Policy", "type": "TABLE"},
                        {"id": "entity:transaction", "name": "Transaction", "type": "TABLE"},
                    ],
                    "relationships": [
                        {
                            "id": "rel-1",
                            "relationship_type": "REFERENCES",
                            "validation_status": "SUPPORTED",
                        }
                    ],
                }
            ]
        return []


def test_hybrid_retriever_expands_graph_terms_and_merges_graph_evidence() -> None:
    vector = FakeVector()
    graph = FakeGraph()
    retriever = HybridRetriever(vector_retriever=vector, graph_query=graph)

    query = "What is the relationship between transaction and policy?"
    results = retriever.search(query, limit=2, graph_hops=1)

    assert len(vector.calls) == 2
    assert vector.calls[0] == (query, 6)
    assert vector.calls[1][0] == "transaction policy relationship dependency"
    assert vector.calls[1][1] == 6
    assert "policy" in graph.search_calls
    assert any(item["source_id"] == "entity:policy" for item in results)
    assert any(item["graph_evidence_count"] > 0 for item in results)


def test_hybrid_retriever_keeps_qdrant_relevance_primary() -> None:
    vector = FakeVector()
    graph = FakeGraph()
    retriever = HybridRetriever(vector_retriever=vector, graph_query=graph)

    results = retriever.search("policy", limit=2, graph_hops=1)

    assert len(vector.calls) == 1
    assert results[0]["source_id"] == "entity:transaction"
    assert results[0]["score"] >= results[1]["score"]


def test_hybrid_retriever_expands_calculation_queries() -> None:
    vector = FakeVector()
    graph = FakeGraph()
    retriever = HybridRetriever(vector_retriever=vector, graph_query=graph)

    retriever.search("How is premium calculated?", limit=2, graph_hops=1)

    assert len(vector.calls) == 2
    assert vector.calls[1][0] == "premium calculated calculation formula business rule"
