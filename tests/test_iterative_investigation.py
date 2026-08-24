"""Tests for iterative multi-round investigation, claim-level verification, and status classification."""
from __future__ import annotations

import json
import pytest
from typing import Any

from knowledge_engineering.investigation_agent.agent import InvestigationAgent
from knowledge_engineering.investigation_agent.config import InvestigationConfig
from knowledge_engineering.investigation_agent.contracts import (
    ClaimVerification,
    GroundedAnswer,
    calculate_grounded_confidence,
    parse_and_validate_verification,
)
from knowledge_engineering.llm.errors import LLMRateLimitError


class MockSequenceRetriever:
    """Retriever returning different batches on successive calls."""

    def __init__(self, responses: list[list[dict[str, Any]]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def search(self, query: str, *, limit: int = 5, graph_hops: int = 1) -> list[dict[str, Any]]:
        self.calls.append({"query": query, "limit": limit, "graph_hops": graph_hops})
        if self.responses:
            return self.responses.pop(0)
        return []


class MockSequenceGenerator:
    """Generator returning predetermined text responses in order."""

    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        if not self.responses:
            raise RuntimeError("No mock responses remaining")
        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


def test_investigation_config_validation() -> None:
    cfg = InvestigationConfig(max_investigation_rounds=3, max_follow_up_queries=4)
    assert cfg.max_investigation_rounds == 3
    assert cfg.max_follow_up_queries == 4

    with pytest.raises(ValueError, match="confidence thresholds"):
        InvestigationConfig(confidence_thresholds=(0.8, 0.2))

    with pytest.raises(ValueError, match="graph hop range"):
        InvestigationConfig(initial_graph_hops=3, max_graph_hops=2)


def test_iterative_multi_round_investigation_escalation() -> None:
    # Round 1: Plan -> Initial search -> Sufficiency Check (Insufficient, 1 follow up)
    # Round 2: Follow up search -> Sufficiency Check (Sufficient!) -> Answer -> Verification
    plan_json = json.dumps({
        "intent": "Investigate formula components",
        "retrieval_queries": ["search_initial"],
    })
    suff_round1 = json.dumps({
        "sufficient": False,
        "knowledge_gaps": ["Missing transformation details"],
        "follow_up_queries": ["search_follow_up"],
    })
    suff_round2 = json.dumps({
        "sufficient": True,
        "knowledge_gaps": [],
        "follow_up_queries": [],
    })
    ans_json = json.dumps({
        "answer": "Formula transforms input X to output Y using rule R1.",
        "evidence_ids": ["e_init", "e_follow"],
        "confidence": 0.90,
        "knowledge_gaps": [],
    })
    ver_json = json.dumps({
        "verified": True,
        "answer": "Formula transforms input X to output Y using rule R1.",
        "evidence_ids": ["e_init", "e_follow"],
        "confidence": 0.90,
        "knowledge_gaps": [],
        "claims": [
            {"claim": "Transforms input X to Y", "supported": "YES", "evidence_id": "e_init", "reason": "Explicit"},
            {"claim": "Uses rule R1", "supported": "YES", "evidence_id": "e_follow", "reason": "Confirmed"},
        ],
    })

    retriever = MockSequenceRetriever([
        [{"id": "e_init", "source_id": "e_init", "score": 0.85, "text": "Input X details", "graph_evidence_count": 1}],
        [{"id": "e_follow", "source_id": "e_follow", "score": 0.90, "text": "Rule R1 details", "graph_evidence_count": 2}],
    ])
    generator = MockSequenceGenerator([plan_json, suff_round1, suff_round2, ans_json, ver_json])
    agent = InvestigationAgent(
        retriever=retriever,
        generator=generator,
        config=InvestigationConfig(max_investigation_rounds=2),
    )

    res = agent.investigate("How does formula calculate output Y?")
    assert res["status"] == "SUCCESS"
    assert res["failure_code"] is None
    assert res["answer_verified"] is True
    assert set(res["evidence_ids"]) == {"e_init", "e_follow"}
    assert len(res["steps"]) == 2
    assert len(res["claims"]) == 2
    assert res["confidence"] >= 0.85


def test_claim_level_verification_parsing_and_filtering() -> None:
    draft = GroundedAnswer(
        answer="Component A connects to B and modifies variable V.",
        evidence_ids=["e1", "e2"],
        confidence=0.90,
        knowledge_gaps=[],
    )
    raw_ver = json.dumps({
        "verified": True,
        "answer": "Component A connects to B.",
        "evidence_ids": ["e1"],
        "confidence": 0.80,
        "knowledge_gaps": ["Modification of variable V is not in evidence"],
        "claims": [
            {"claim": "Component A connects to B", "supported": "YES", "evidence_id": "e1", "reason": "Graph edge"},
            {"claim": "Modifies variable V", "supported": "NO", "evidence_id": "e_fake", "reason": "Absent"},
        ],
    })
    res = parse_and_validate_verification(raw_ver, draft, allowed_evidence_ids={"e1", "e2"})
    assert res.verified is True
    assert res.evidence_ids == ["e1"]
    assert len(res.claims) == 2
    assert res.claims[0].supported == "YES"
    assert res.claims[0].evidence_id == "e1"
    # e_fake must be stripped because it was not in allowed evidence
    assert res.claims[1].supported == "NO"
    assert res.claims[1].evidence_id is None


def test_top_level_status_rate_limited_distinction() -> None:
    retriever = MockSequenceRetriever([
        [{"id": "e1", "source_id": "e1", "score": 0.8, "text": "Evidence 1 text"}],
    ])
    # Generator throws rate limit error during answer generation
    generator = MockSequenceGenerator([
        json.dumps({"intent": "plan", "retrieval_queries": ["q1"]}),
        json.dumps({"sufficient": True, "knowledge_gaps": []}),
        LLMRateLimitError("Rate limit reached: try again in 2s"),
    ])
    agent = InvestigationAgent(retriever=retriever, generator=generator)

    res = agent.investigate("Explain business logic for rule R")
    # Must distinguish RATE_LIMITED from INSUFFICIENT_EVIDENCE
    assert res["status"] == "RATE_LIMITED"
    assert res["failure_code"] == "LLM_RATE_LIMIT"
    assert "rate limit" in res["answer"].lower()
    assert res["confidence"] == 0.0


def test_top_level_status_insufficient_evidence() -> None:
    retriever = MockSequenceRetriever([[]])  # Zero results
    generator = MockSequenceGenerator([])
    agent = InvestigationAgent(retriever=retriever, generator=generator)

    res = agent.investigate("What is unknown component X?")
    assert res["status"] == "INSUFFICIENT_EVIDENCE"
    assert res["failure_code"] == "EVIDENCE_INSUFFICIENT"
    assert res["confidence"] == 0.0
    assert "No evidence was found" in res["answer"]
