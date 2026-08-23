from __future__ import annotations

from typing import Any

from knowledge_engineering.investigation_agent import InvestigationAgent


class FakeRetriever:
    def __init__(self, batches: dict[str, list[dict[str, Any]]]) -> None:
        self.batches = batches
        self.calls: list[tuple[str, int, int]] = []

    def search(self, query: str, *, limit: int = 5, graph_hops: int = 1) -> list[dict[str, Any]]:
        self.calls.append((query, limit, graph_hops))
        return self.batches.get(query, [])


class FakeGenerator:
    provider = "fake"
    model = "test"

    def __init__(self, answer: str, plan_queries: list[str] | None = None) -> None:
        self.answer = answer
        self.plan_queries = plan_queries or []
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if "investigation planning component" in prompt:
            return (
                '{"intent":"investigate dependencies",'
                '"objectives":["identify supported connections"],'
                f'"retrieval_queries":{self.plan_queries!r},'
                '"evidence_requirements":["graph-backed evidence"],'
                '"requested_output_format":"answer",'
                '"constraints":["use supplied evidence only"]}'
            ).replace("'", '"')
        return self.answer


def _item(source_id: str, *, graph: bool = False) -> dict[str, Any]:
    return {
        "id": f"vector:{source_id}",
        "source_id": source_id,
        "score": 0.9,
        "kind": "entity",
        "artifact_id": "artifact:generic",
        "text": f"Evidence for {source_id}",
        "metadata": {},
        "graph_evidence": [{"relationships": []}] if graph else [],
        "graph_evidence_count": 1 if graph else 0,
    }


def test_investigation_stops_when_initial_evidence_has_graph_support() -> None:
    query = "How are source entities connected?"
    retriever = FakeRetriever({query: [_item("entity:source", graph=True)]})
    generator = FakeGenerator(
        '{"answer":"Supported by graph evidence.","evidence_ids":["entity:source"],"confidence":0.9,"knowledge_gaps":[]}',
        [query],
    )

    result = InvestigationAgent(retriever=retriever, generator=generator).investigate(query)

    assert result["investigation_triggered"] is False
    assert len(result["steps"]) == 1
    assert retriever.calls == [(query, 5, 1)]
    assert result["evidence_ids"] == ["entity:source"]
    assert result["confidence_level"] == "HIGH"
    assert result["trace_references"][0]["evidence_ids"] == ["entity:source"]


def test_investigation_escalates_using_llm_plan() -> None:
    query = "Explain the dependency between components."
    follow_one = f"{query} direct relationships"
    follow_two = f"{query} cross-artifact lineage"
    retriever = FakeRetriever(
        {
            query: [_item("entity:initial")],
            follow_one: [_item("entity:follow", graph=True)],
            follow_two: [_item("entity:other", graph=True)],
        }
    )
    generator = FakeGenerator(
        '{"answer":"The relationship is supported by the investigated evidence.",'
        '"evidence_ids":["entity:follow","entity:other"],"confidence":0.82,"knowledge_gaps":[]}',
        [follow_one, follow_two],
    )

    result = InvestigationAgent(retriever=retriever, generator=generator).investigate(
        query, limit=3, initial_graph_hops=1, max_graph_hops=3
    )

    assert result["investigation_triggered"] is True
    assert len(result["steps"]) == 3
    assert retriever.calls == [
        (query, 3, 1),
        (follow_one, 3, 2),
        (follow_two, 3, 2),
    ]
    assert result["retrieved_count"] == 3
    assert result["plan"]["objectives"] == ["identify supported connections"]


def test_investigation_filters_generator_citations_to_retrieved_evidence() -> None:
    query = "What evidence supports this dependency?"
    retriever = FakeRetriever({query: [_item("entity:known", graph=True)]})
    generator = FakeGenerator(
        '{"answer":"Only known evidence is available.",'
        '"evidence_ids":["entity:known","entity:not-retrieved"],"confidence":0.7,"knowledge_gaps":[]}',
        [query],
    )

    result = InvestigationAgent(retriever=retriever, generator=generator).investigate(query)

    assert result["evidence_ids"] == ["entity:known"]
    assert [item["source_id"] for item in result["evidence"]] == ["entity:known"]


def test_investigation_accepts_structured_request_without_domain_hardcoding() -> None:
    request = {
        "question": "How does component A depend on component B?",
        "document_type": "lineage report",
        "output_format": "structured summary",
    }
    retriever = FakeRetriever({request["question"]: [_item("entity:a", graph=True)]})
    generator = FakeGenerator(
        '{"answer":"The evidence is sufficient.","evidence_ids":["entity:a"],"confidence":0.8,"knowledge_gaps":[]}',
        [request["question"]],
    )

    result = InvestigationAgent(retriever=retriever, generator=generator).investigate(request)

    assert result["request"]["document_type"] == "lineage report"
    assert result["request"]["output_format"] == "structured summary"
    assert "component A" in generator.prompts[-1]
    assert "component B" in generator.prompts[-1]
    assert "ClaimCenter" not in generator.prompts[-1]
    assert "PolicyCenter" not in generator.prompts[-1]
