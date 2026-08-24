"""Unit tests for investigation stage contracts and schema validation."""
from __future__ import annotations

import json
import pytest

from knowledge_engineering.investigation_agent.contracts import (
    GroundedAnswer,
    InvestigationPlan,
    SufficiencyAssessment,
    VerificationResult,
    extract_json_payload,
    parse_and_validate_answer,
    parse_and_validate_plan,
    parse_and_validate_sufficiency,
    parse_and_validate_verification,
)
from knowledge_engineering.llm.generator import generate_structured


def test_extract_json_payload_from_markdown_fence() -> None:
    text = "Here is your plan:\n```json\n{\"intent\": \"test intent\", \"objectives\": [\"obj1\"]}\n```\nHope this helps!"
    payload = extract_json_payload(text)
    assert payload == {"intent": "test intent", "objectives": ["obj1"]}


def test_extract_json_payload_from_embedded_text() -> None:
    text = "Preamble {\"answer\": \"valid result\", \"evidence_ids\": [\"e1\"]} Postscript"
    payload = extract_json_payload(text)
    assert payload["answer"] == "valid result"
    assert payload["evidence_ids"] == ["e1"]


def test_extract_json_payload_malformed_raises() -> None:
    with pytest.raises(ValueError, match="No valid JSON object"):
        extract_json_payload("There is no json in here at all.")


def test_planner_contract_valid_response() -> None:
    raw = json.dumps({
        "intent": "analyze system connections",
        "objectives": ["obj1", "obj2"],
        "retrieval_queries": ["q1", "q2"],
        "evidence_requirements": ["req1"],
        "requested_output_format": "json",
        "constraints": ["c1"],
    })
    plan = parse_and_validate_plan(raw, "fallback query")
    assert isinstance(plan, InvestigationPlan)
    assert plan.intent == "analyze system connections"
    assert plan.objectives == ["obj1", "obj2"]
    assert plan.retrieval_queries == ["q1", "q2"]


def test_planner_contract_malformed_uses_fallback_query() -> None:
    raw = json.dumps({"intent": "", "retrieval_queries": []})
    plan = parse_and_validate_plan(raw, "fallback query")
    assert plan.retrieval_queries == ["fallback query"]
    assert "Investigate" in plan.intent


def test_sufficiency_contract_valid_response() -> None:
    raw = json.dumps({
        "sufficient": True,
        "knowledge_gaps": [],
        "follow_up_queries": [],
    })
    assessment = parse_and_validate_sufficiency(raw)
    assert isinstance(assessment, SufficiencyAssessment)
    assert assessment.sufficient is True
    assert assessment.knowledge_gaps == []


def test_sufficiency_contract_insufficient_with_gaps() -> None:
    raw = json.dumps({
        "sufficient": False,
        "knowledge_gaps": ["Missing relationship data"],
        "follow_up_queries": ["query for relationship"],
    })
    assessment = parse_and_validate_sufficiency(raw)
    assert assessment.sufficient is False
    assert assessment.knowledge_gaps == ["Missing relationship data"]
    assert assessment.follow_up_queries == ["query for relationship"]


def test_sufficiency_contract_missing_field_raises() -> None:
    raw = json.dumps({"knowledge_gaps": ["gap"]})
    with pytest.raises(ValueError, match="Missing required boolean field 'sufficient'"):
        parse_and_validate_sufficiency(raw)


def test_answer_contract_valid_response() -> None:
    raw = json.dumps({
        "answer": "The calculation uses rule R1.",
        "evidence_ids": ["e1", "e2"],
        "confidence": 0.85,
        "knowledge_gaps": [],
    })
    answer = parse_and_validate_answer(raw, allowed_evidence_ids={"e1", "e2", "e3"})
    assert isinstance(answer, GroundedAnswer)
    assert answer.answer == "The calculation uses rule R1."
    assert answer.evidence_ids == ["e1", "e2"]
    assert answer.confidence == 0.85


def test_answer_contract_filters_unknown_evidence_ids() -> None:
    raw = json.dumps({
        "answer": "Direct answer",
        "evidence_ids": ["e1", "e_fake", "e_hallucinated"],
        "confidence": 0.9,
        "knowledge_gaps": [],
    })
    answer = parse_and_validate_answer(raw, allowed_evidence_ids={"e1", "e2"})
    assert answer.evidence_ids == ["e1"]
    assert "e_fake" not in answer.evidence_ids
    assert "e_hallucinated" not in answer.evidence_ids


def test_answer_contract_deduplicates_evidence_ids() -> None:
    raw = json.dumps({
        "answer": "Direct answer",
        "evidence_ids": ["e1", "e1", "e2", "e1"],
        "confidence": 0.75,
        "knowledge_gaps": [],
    })
    answer = parse_and_validate_answer(raw, allowed_evidence_ids={"e1", "e2"})
    assert answer.evidence_ids == ["e1", "e2"]


def test_answer_contract_empty_answer_raises() -> None:
    raw = json.dumps({"answer": "   ", "evidence_ids": ["e1"], "confidence": 0.5})
    with pytest.raises(ValueError, match="non-empty 'answer'"):
        parse_and_validate_answer(raw)


def test_answer_contract_clamps_invalid_confidence() -> None:
    raw_high = json.dumps({"answer": "ans", "confidence": 1.5})
    ans_high = parse_and_validate_answer(raw_high)
    assert ans_high.confidence == 1.0

    raw_neg = json.dumps({"answer": "ans", "confidence": -0.5})
    ans_neg = parse_and_validate_answer(raw_neg)
    assert ans_neg.confidence == 0.0

    raw_invalid = json.dumps({"answer": "ans", "confidence": "unknown"})
    ans_inv = parse_and_validate_answer(raw_invalid)
    assert ans_inv.confidence == 0.0


def test_verifier_contract_valid_response() -> None:
    draft = GroundedAnswer(answer="Draft text", evidence_ids=["e1"], confidence=0.7, knowledge_gaps=[])
    raw = json.dumps({
        "verified": True,
        "answer": "Verified text",
        "evidence_ids": ["e1"],
        "confidence": 0.8,
        "knowledge_gaps": [],
    })
    result = parse_and_validate_verification(raw, draft, allowed_evidence_ids={"e1"})
    assert isinstance(result, VerificationResult)
    assert result.verified is True
    assert result.answer == "Verified text"
    assert result.evidence_ids == ["e1"]
    assert result.confidence == 0.8


def test_verifier_contract_rejects_unretrieved_ids() -> None:
    draft = GroundedAnswer(answer="Draft text", evidence_ids=["e1"], confidence=0.7, knowledge_gaps=[])
    raw = json.dumps({
        "verified": True,
        "answer": "Modified text",
        "evidence_ids": ["e1", "unretrieved_id"],
        "confidence": 0.8,
        "knowledge_gaps": [],
    })
    result = parse_and_validate_verification(raw, draft, allowed_evidence_ids={"e1"})
    assert result.evidence_ids == ["e1"]
    assert "unretrieved_id" not in result.evidence_ids


class MockGeneratorWithRepair:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        if self.responses:
            return self.responses.pop(0)
        return "{}"


def test_generate_structured_retry_success() -> None:
    # First response is invalid JSON, second response is valid JSON
    generator = MockGeneratorWithRepair([
        "Invalid non-json response",
        json.dumps({"intent": "repaired intent", "retrieval_queries": ["q1"]}),
    ])
    result = generate_structured(
        generator,
        "Initial prompt",
        lambda c: parse_and_validate_plan(c, "fallback"),
        max_repair_retries=1,
    )
    assert result.intent == "repaired intent"
    assert len(generator.calls) == 2
    assert "Validation error" in generator.calls[1]


def test_generate_structured_retry_exhaustion() -> None:
    generator = MockGeneratorWithRepair(["Invalid response 1", "Invalid response 2"])
    with pytest.raises(ValueError, match="Failed to produce valid structured output after 2 attempts"):
        generate_structured(
            generator,
            "Initial prompt",
            lambda c: parse_and_validate_plan(c, "fallback"),
            max_repair_retries=1,
        )
