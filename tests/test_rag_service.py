from __future__ import annotations

from typing import Any

from knowledge_engineering.rag_service import RAGService


class FakeRetriever:
    def __init__(self, evidence: list[dict[str, Any]]) -> None:
        self.evidence = evidence
        self.calls: list[tuple[str, int, int]] = []

    def connect(self) -> None:
        pass

    def close(self) -> None:
        pass

    def search(self, query: str, *, limit: int, graph_hops: int) -> list[dict[str, Any]]:
        self.calls.append((query, limit, graph_hops))
        return self.evidence


class FakeGenerator:
    provider = "fake"
    model = "test"

    def __init__(self, response: str) -> None:
        self.response = response
        self.prompt = ""

    def generate(self, prompt: str) -> str:
        self.prompt = prompt
        return self.response


def _evidence() -> list[dict[str, Any]]:
    return [
        {
            "id": "vector-1",
            "score": 0.9,
            "text": "Entity: vw_curr_cc_transaction. Type: TABLE.",
            "kind": "entity",
            "source_id": "entity:transaction",
            "artifact_id": "artifact:claimcenter",
            "graph_evidence": [
                {
                    "nodes": [
                        {"id": "entity:transaction", "name": "vw_curr_cc_transaction", "type": "TABLE"},
                        {"id": "artifact:claimcenter", "name": "ClaimCenter_CPP_Breakdown_metadata", "type": "ARTIFACT"},
                    ],
                    "relationships": [
                        {"id": "rel-1", "relationship_type": "USES", "validation_status": "SUPPORTED"}
                    ],
                }
            ],
        },
        {
            "id": "vector-2",
            "score": 0.8,
            "text": "Entity: vw_curr_cc_policy. Type: TABLE.",
            "kind": "entity",
            "source_id": "entity:policy",
            "artifact_id": "artifact:claimcenter",
            "graph_evidence": [],
        },
    ]


def test_rag_service_returns_only_retrieved_evidence_ids() -> None:
    retriever = FakeRetriever(_evidence())
    generator = FakeGenerator(
        '{"answer":"The evidence does not establish a direct relationship.",'
        '"evidence_ids":["entity:transaction","not-retrieved","entity:policy"]}'
    )
    service = RAGService(retriever=retriever, generator=generator)

    result = service.ask("How are transaction and policy related?", limit=2, graph_hops=1)

    assert result["answer"] == "The evidence does not establish a direct relationship."
    assert result["evidence_ids"] == ["entity:transaction", "entity:policy"]
    assert [item["source_id"] for item in result["evidence"]] == [
        "entity:transaction",
        "entity:policy",
    ]
    assert result["retrieved_count"] == 2
    assert retriever.calls == [("How are transaction and policy related?", 2, 1)]


def test_rag_prompt_contains_graph_relationship_evidence() -> None:
    retriever = FakeRetriever(_evidence())
    generator = FakeGenerator(
        '{"answer":"Transaction uses the ClaimCenter artifact.",'
        '"evidence_ids":["entity:transaction"]}'
    )
    service = RAGService(retriever=retriever, generator=generator)

    service.ask("What does the transaction table use?", limit=1, graph_hops=1)

    assert "vw_curr_cc_transaction" in generator.prompt
    assert "USES" in generator.prompt
    assert "SUPPORTED" in generator.prompt
    assert "Do not infer a direct relationship" in generator.prompt


def test_rag_service_rejects_empty_query() -> None:
    service = RAGService(
        retriever=FakeRetriever([]),
        generator=FakeGenerator('{"answer":"unused","evidence_ids":[]}'),
    )

    try:
        service.ask("   ")
    except ValueError as exc:
        assert str(exc) == "query must not be empty"
    else:
        raise AssertionError("Expected ValueError for an empty query")
