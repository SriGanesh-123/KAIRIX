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

    def __init__(self, evidence_id):
        self.evidence_id = evidence_id
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return '{"answer":"grounded","evidence_ids":["' + self.evidence_id + '"]}'


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
    generator = FakeGenerator("entity:a")
    agent = InvestigationAgent(retriever=retriever, generator=generator)

    result = agent.investigate(query)

    assert result["investigation_triggered"] is False
    assert len(retriever.calls) == 1
    assert result["evidence_ids"] == ["entity:a"]


def test_insufficient_initial_evidence_triggers_follow_up_searches():
    query = "transaction policy"
    retriever = FakeRetriever({
        query: [evidence("entity:a")],
        f"{query} direct relationships dependencies": [evidence("entity:b", graph_count=1)],
        f"{query} cross-artifact relationships lineage": [evidence("entity:c")],
    })
    generator = FakeGenerator("entity:b")
    agent = InvestigationAgent(retriever=retriever, generator=generator)

    result = agent.investigate(query, initial_graph_hops=1, max_graph_hops=3)

    assert result["investigation_triggered"] is True
    assert len(retriever.calls) == 3
    assert retriever.calls[1][2] == 2
    assert result["evidence_ids"] == ["entity:b"]


def test_investigation_deduplicates_evidence():
    query = "transaction policy"
    shared = evidence("entity:a")
    retriever = FakeRetriever({
        query: [shared],
        f"{query} direct relationships dependencies": [shared],
        f"{query} cross-artifact relationships lineage": [evidence("entity:b")],
    })
    generator = FakeGenerator("entity:a")
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
