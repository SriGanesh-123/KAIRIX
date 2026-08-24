from __future__ import annotations

from typing import Any

from knowledge_engineering.investigation_agent import InvestigationAgent


class Retriever:
    def search(self, query: str, *, limit: int = 5, graph_hops: int = 1) -> list[dict[str, Any]]:
        return [
            {
                "id": "entity:source",
                "source_id": "entity:source",
                "score": 0.95,
                "kind": "entity",
                "artifact_id": "artifact:generic",
                "text": "The source is connected to the target through a supported relationship.",
                "metadata": {},
                "graph_evidence": [{"relationships": [{"id": "relationship:1", "validation_status": "SUPPORTED"}]}],
                "graph_evidence_count": 1,
            }
        ]


class VerifyingGenerator:
    provider = "fake"
    model = "test"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        if "investigation planning component" in prompt:
            return '{"intent":"investigate","objectives":["answer the request"],"retrieval_queries":["source target relationship"],"evidence_requirements":["supported relationship"],"requested_output_format":"answer","constraints":[]}'
        if "evidence-sufficiency component" in prompt:
            return '{"sufficient":true,"knowledge_gaps":[],"follow_up_queries":[]}'
        if "final evidence verifier" in prompt:
            return '{"answer":"The source is connected to the target through a supported relationship.","evidence_ids":["entity:source"],"confidence":0.93,"knowledge_gaps":[],"verified":true}'
        return '{"answer":"The source is connected to the target.","evidence_ids":["entity:source"],"confidence":0.9,"knowledge_gaps":[]}'


def test_investigation_verifies_final_answer_and_exposes_status() -> None:
    generator = VerifyingGenerator()
    result = InvestigationAgent(retriever=Retriever(), generator=generator).investigate(
        "How are the source and target connected?"
    )

    assert result["answer_verified"] is True
    assert result["confidence"] == 0.93
    assert result["evidence_ids"] == ["entity:source"]
    assert result["confidence_level"] == "HIGH"
    assert any("final evidence verifier" in prompt for prompt in generator.calls)


def test_verifier_cannot_introduce_unretrieved_evidence_ids() -> None:
    class UnsafeVerifier(VerifyingGenerator):
        def generate(self, prompt: str) -> str:
            if "final evidence verifier" in prompt:
                return '{"answer":"Unsupported claim.","evidence_ids":["entity:missing"],"confidence":0.9,"knowledge_gaps":[],"verified":true}'
            return super().generate(prompt)

    result = InvestigationAgent(retriever=Retriever(), generator=UnsafeVerifier()).investigate(
        "What evidence supports the connection?"
    )

    assert result["answer_verified"] is True
    assert result["evidence_ids"] == []
    assert result["confidence_level"] == "HIGH"
    assert any("No retrieved evidence was cited" in gap for gap in result["knowledge_gaps"])
