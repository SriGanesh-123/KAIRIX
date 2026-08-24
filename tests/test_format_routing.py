from __future__ import annotations

from typing import Any

from knowledge_engineering.rag_service import RAGService


class FakeRetriever:
    def connect(self) -> None:
        pass

    def close(self) -> None:
        pass

    def search(self, query: str, *, limit: int = 5, graph_hops: int = 1) -> list[dict[str, Any]]:
        return [
            {
                "source_id": "entity:test",
                "id": "vector:test",
                "score": 0.9,
                "kind": "entity",
                "artifact_id": "artifact:test",
                "text": "test evidence",
                "metadata": {},
                "graph_evidence": [
                    {
                        "nodes": [{"id": "entity:test", "name": "Test", "type": "TABLE"}],
                        "relationships": [],
                    }
                ],
                "graph_evidence_count": 1,
            }
        ]


class FakeGenerator:
    provider = "fake"
    model = "test"

    def generate(self, prompt: str) -> str:
        if "investigation planning component" in prompt:
            return '{"intent":"test","objectives":[],"retrieval_queries":[],"evidence_requirements":[],"requested_output_format":"","constraints":[]}'
        if "evidence-sufficiency component" in prompt:
            return '{"sufficient":true,"knowledge_gaps":[],"follow_up_queries":[]}'
        return '{"answer":"grounded","evidence_ids":["entity:test"],"confidence":0.8,"knowledge_gaps":[]}'


def test_normal_question_ignores_planner_response_format() -> None:
    result = RAGService(retriever=FakeRetriever(), generator=FakeGenerator()).investigate("How is premium calculated?")
    assert result["format"] is None
    assert result["format_valid"] is True


def test_explicit_predefined_format_is_validated() -> None:
    result = RAGService(retriever=FakeRetriever(), generator=FakeGenerator()).investigate(
        {"question": "What does this depend on?", "output_format": "Dependency Report"}
    )
    assert result["format"]["name"] == "Dependency Report"
    assert result["format_valid"] is True
