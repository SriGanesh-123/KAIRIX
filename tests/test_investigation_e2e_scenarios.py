"""End-to-end domain-neutral scenario tests for the investigation agent."""
from __future__ import annotations

import json
from typing import Any
import pytest

from knowledge_engineering.investigation_agent import InvestigationAgent


class ScenarioRetriever:
    """Configurable synthetic retriever for testing different investigation scenarios."""

    def __init__(
        self,
        evidence_map: dict[str, list[dict[str, Any]]],
        *,
        fail_on: set[str] | None = None,
    ) -> None:
        self.evidence_map = evidence_map
        self.fail_on = fail_on or set()
        self.calls: list[str] = []

    def search(self, query: str, *, limit: int = 5, graph_hops: int = 1) -> list[dict[str, Any]]:
        self.calls.append(query)
        if query in self.fail_on:
            raise RuntimeError(f"Simulated retrieval failure for {query!r}")
        return self.evidence_map.get(query, [])


class ScenarioGenerator:
    """Configurable synthetic generator returning typed responses for stages."""

    provider = "scenario_mock"
    model = "mock-v1"

    def __init__(
        self,
        *,
        plan_response: dict[str, Any] | None = None,
        sufficiency_response: dict[str, Any] | None = None,
        answer_response: dict[str, Any] | None = None,
        verifier_response: dict[str, Any] | None = None,
        fail_stages: set[str] | None = None,
    ) -> None:
        self.plan_response = plan_response
        self.sufficiency_response = sufficiency_response
        self.answer_response = answer_response
        self.verifier_response = verifier_response
        self.fail_stages = fail_stages or set()
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)

        if "investigation planning component" in prompt:
            if "plan" in self.fail_stages:
                raise RuntimeError("Simulated planner failure")
            resp = self.plan_response or {
                "intent": "analyze relationships",
                "objectives": ["identify connections"],
                "retrieval_queries": ["query follow-up"],
                "evidence_requirements": ["graph links"],
                "requested_output_format": "text",
                "constraints": [],
            }
            return json.dumps(resp)

        if "evidence-sufficiency component" in prompt:
            if "sufficiency" in self.fail_stages:
                raise RuntimeError("Simulated sufficiency failure")
            resp = self.sufficiency_response or {
                "sufficient": True,
                "knowledge_gaps": [],
                "follow_up_queries": [],
            }
            return json.dumps(resp)

        if "final evidence verifier" in prompt:
            if "verifier" in self.fail_stages:
                raise RuntimeError("Simulated verifier failure")
            if self.verifier_response:
                return json.dumps(self.verifier_response)
            ans = self.answer_response or {"answer": "Grounded conclusion from evidence.", "evidence_ids": ["entity:node_a"], "confidence": 0.8, "knowledge_gaps": []}
            return json.dumps({
                "verified": True,
                "answer": ans.get("answer", "Verified conclusion."),
                "evidence_ids": ans.get("evidence_ids", []),
                "confidence": ans.get("confidence", 0.8),
                "knowledge_gaps": ans.get("knowledge_gaps", []),
            })

        # Default: answer generation stage
        if "answer" in self.fail_stages:
            raise RuntimeError("Simulated answer generator failure")
        resp = self.answer_response or {
            "answer": "Grounded conclusion from evidence.",
            "evidence_ids": ["entity:node_a"],
            "confidence": 0.8,
            "knowledge_gaps": [],
        }
        return json.dumps(resp)


def _synthetic_item(
    source_id: str,
    text: str,
    *,
    kind: str = "entity",
    has_graph: bool = False,
    score: float = 0.9,
) -> dict[str, Any]:
    return {
        "id": f"id-{source_id}",
        "source_id": source_id,
        "score": score,
        "kind": kind,
        "artifact_id": "artifact:module_1",
        "text": text,
        "metadata": {"synthetic": True},
        "graph_evidence": [
            {
                "nodes": [{"id": source_id, "name": source_id}, {"id": "target_1", "name": "target_1"}],
                "relationships": [{"relationship_type": "CONNECTS_TO", "validation_status": "SUPPORTED"}],
            }
        ] if has_graph else [],
        "graph_evidence_count": 1 if has_graph else 0,
    }


def test_scenario_relationship_investigation() -> None:
    query = "How is module Alpha related to module Beta?"
    ev_initial = [_synthetic_item("entity:alpha", "Module Alpha invokes module Beta.", has_graph=True)]
    retriever = ScenarioRetriever({query: ev_initial})
    generator = ScenarioGenerator(
        answer_response={
            "answer": "Module Alpha directly invokes module Beta via call interface.",
            "evidence_ids": ["entity:alpha"],
            "confidence": 0.9,
            "knowledge_gaps": [],
        },
        verifier_response={
            "verified": True,
            "answer": "Module Alpha directly invokes module Beta via call interface.",
            "evidence_ids": ["entity:alpha"],
            "confidence": 0.9,
            "knowledge_gaps": [],
        },
    )

    agent = InvestigationAgent(retriever=retriever, generator=generator)
    result = agent.investigate(query)

    assert result["answer_verified"] is True
    assert "Module Alpha directly invokes" in result["answer"]
    assert result["evidence_ids"] == ["entity:alpha"]
    assert result["confidence_level"] == "HIGH"


def test_scenario_calculation_investigation() -> None:
    query = "What logic calculates metric Gamma in the system?"
    ev_initial = [_synthetic_item("entity:gamma", "Metric Gamma is calculated by multiplying factors X and Y.")]
    retriever = ScenarioRetriever({query: ev_initial})
    generator = ScenarioGenerator(
        answer_response={
            "answer": "Metric Gamma is computed as the product of factor X and factor Y.",
            "evidence_ids": ["entity:gamma"],
            "confidence": 0.85,
            "knowledge_gaps": [],
        },
        verifier_response={
            "verified": True,
            "answer": "Metric Gamma is computed as the product of factor X and factor Y.",
            "evidence_ids": ["entity:gamma"],
            "confidence": 0.85,
            "knowledge_gaps": [],
        },
    )

    agent = InvestigationAgent(retriever=retriever, generator=generator)
    result = agent.investigate(query)

    assert "product of factor X and factor Y" in result["answer"]
    assert result["evidence_ids"] == ["entity:gamma"]
    assert result["confidence"] == 0.85


def test_scenario_lineage_and_dependency_escalation() -> None:
    query = "Trace the data lineage from input feed to aggregate table."
    follow_up = "Find transform steps between input feed and aggregate table."
    retriever = ScenarioRetriever({
        query: [_synthetic_item("entity:feed", "Input feed receives raw records.")],
        follow_up: [_synthetic_item("entity:transform", "Transform step aggregates raw records to aggregate table.", has_graph=True)],
    })
    generator = ScenarioGenerator(
        sufficiency_response={
            "sufficient": False,
            "knowledge_gaps": ["Missing intermediate transform step."],
            "follow_up_queries": [follow_up],
        },
        answer_response={
            "answer": "Data flows from input feed through the transform step into the aggregate table.",
            "evidence_ids": ["entity:feed", "entity:transform"],
            "confidence": 0.88,
            "knowledge_gaps": [],
        },
        verifier_response={
            "verified": True,
            "answer": "Data flows from input feed through the transform step into the aggregate table.",
            "evidence_ids": ["entity:feed", "entity:transform"],
            "confidence": 0.88,
            "knowledge_gaps": [],
        },
    )

    agent = InvestigationAgent(retriever=retriever, generator=generator)
    result = agent.investigate(query)

    assert result["investigation_triggered"] is True
    assert len(result["steps"]) == 2
    assert set(result["evidence_ids"]) == {"entity:feed", "entity:transform"}
    assert result["retrieved_count"] == 2


def test_scenario_insufficient_evidence_produces_structured_gaps() -> None:
    query = "Find evidence of unpublished protocol Zeta."
    retriever = ScenarioRetriever({query: []})
    generator = ScenarioGenerator()

    agent = InvestigationAgent(retriever=retriever, generator=generator)
    result = agent.investigate(query)

    assert result["confidence"] == 0.0
    assert result["confidence_level"] == "LOW"
    assert result["evidence_ids"] == []
    assert len(result["knowledge_gaps"]) > 0


def test_scenario_provider_failure_is_handled_gracefully() -> None:
    query = "Explain component interactions."
    retriever = ScenarioRetriever({query: [_synthetic_item("entity:comp", "Component details.")]})
    generator = ScenarioGenerator(fail_stages={"answer"})

    agent = InvestigationAgent(retriever=retriever, generator=generator)
    result = agent.investigate(query)

    assert result["confidence"] == 0.0
    assert result["confidence_level"] == "LOW"
    assert result["evidence_ids"] == []
    assert any("answer-generation step" in gap or "error" in gap.lower() for gap in result["knowledge_gaps"])


def test_scenario_retrieval_failure_on_followup_continues() -> None:
    query = "Explain component linkage."
    follow_up_fail = "failed query"
    retriever = ScenarioRetriever(
        {query: [_synthetic_item("entity:comp1", "Component 1 details.")]},
        fail_on={follow_up_fail},
    )
    generator = ScenarioGenerator(
        sufficiency_response={
            "sufficient": False,
            "knowledge_gaps": ["Need more details."],
            "follow_up_queries": [follow_up_fail],
        },
        answer_response={
            "answer": "Component 1 is documented.",
            "evidence_ids": ["entity:comp1"],
            "confidence": 0.6,
            "knowledge_gaps": ["Need more details."],
        },
    )

    agent = InvestigationAgent(retriever=retriever, generator=generator)
    # Must not raise an unhandled exception when follow-up query encounters retrieval failure
    result = agent.investigate(query)
    assert result["answer_verified"] is True
    assert result["evidence_ids"] == ["entity:comp1"]
