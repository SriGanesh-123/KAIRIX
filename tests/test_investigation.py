from __future__ import annotations

from knowledge_engineering.investigation import InvestigationAgent


class FakeRetriever:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def search(self, query, *, limit=5, graph_hops=1):
        self.calls.append((query, limit, graph_hops))
        return self.responses.get(query, [])


class FakeGenerator:
    provider = "fake"
    model = "test"

    def __init__(self, evidence_id, plan_queries=None):
        self.evidence_id = evidence_id
        self.plan_queries = plan_queries or []
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        if "investigation planning component" in prompt:
            queries = str(self.plan_queries).replace("'", '"')
            return (
                '{"intent":"investigate", "objectives":[], '
                f'"retrieval_queries":{queries}, '
                '"evidence_requirements":[], "requested_output_format":"", "constraints":[]}'
            )
        return (
            '{"answer":"grounded","evidence_ids":["' + self.evidence_id + '"],'
            '"confidence":0.8,"knowledge_gaps":[]}'
        )


def evidence(source_id, graph_count=0):
    return {
        "source_id": source_id,
        "id": source_id + ":vector",
        "score": 0.8,
        "kind": "entity",
        "artifact_id": "artifact:a",
        "text": "evidence",
        "metadata": {},
        "graph_evidence": [],
        "graph_evidence_count": graph_count,
    }


def test_sufficient_initial_evidence_does_not_trigger_investigation():
    query = "transaction policy"
    retriever = FakeRetriever({query: [evidence("entity:a", graph_count=1)]})
    generator = FakeGenerator("entity:a", [query])
    agent = InvestigationAgent(retriever=retriever, generator=generator)

    result = agent.investigate(query)

    assert result["investigation_triggered"] is False
    assert len(retriever.calls) == 1
    assert result["evidence_ids"] == ["entity:a"]


def test_insufficient_initial_evidence_triggers_follow_up_searches():
    query = "transaction policy"
    follow_one = f"{query} direct relationships dependencies"
    follow_two = f"{query} cross-artifact relationships lineage"
    retriever = FakeRetriever({
        query: [evidence("entity:a")],
        follow_one: [evidence("entity:b", graph_count=1)],
        follow_two: [evidence("entity:c")],
    })
    generator = FakeGenerator("entity:b", [follow_one, follow_two])
    agent = InvestigationAgent(retriever=retriever, generator=generator)

    result = agent.investigate(query, initial_graph_hops=1, max_graph_hops=3)

    assert result["investigation_triggered"] is True
    assert len(retriever.calls) == 3
    assert retriever.calls[1][2] == 2
    assert result["evidence_ids"] == ["entity:b"]


def test_investigation_deduplicates_evidence():
    query = "transaction policy"
    shared = evidence("entity:a")
    follow_one = f"{query} direct relationships dependencies"
    follow_two = f"{query} cross-artifact relationships lineage"
    retriever = FakeRetriever({
        query: [shared],
        follow_one: [shared],
        follow_two: [evidence("entity:b")],
    })
    generator = FakeGenerator("entity:a", [follow_one, follow_two])
    agent = InvestigationAgent(retriever=retriever, generator=generator)

    result = agent.investigate(query)

    assert result["retrieved_count"] == 2


def test_invalid_investigation_arguments_are_rejected():
    retriever = FakeRetriever({})
    generator = FakeGenerator("entity:a")
    agent = InvestigationAgent(retriever=retriever, generator=generator)

    for kwargs in ({"limit": 0}, {"initial_graph_hops": 0}, {"initial_graph_hops": 3, "max_graph_hops": 2}):
        try:
            agent.investigate("query", **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
