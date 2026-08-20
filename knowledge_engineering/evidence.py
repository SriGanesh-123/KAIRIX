"""Evidence coverage and confidence assessment for knowledge findings."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict


def assess_evidence(document: Dict[str, Any]) -> Dict[str, Any]:
    evidence_by_artifact: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in document.get("evidence", []):
        evidence_by_artifact[item.get("artifact_id", "")].append(item)

    entity_evidence = {item.get("id"): len(item.get("evidence_ids", [])) for item in document.get("entities", [])}
    relationship_evidence = {item.get("id"): len(item.get("evidence_ids", [])) for item in document.get("relationships", [])}
    rule_evidence = {item.get("id"): len(item.get("evidence_ids", [])) for item in document.get("business_rules", [])}

    findings = []
    for artifact in document.get("artifacts", []):
        artifact_id = artifact["id"]
        artifact_entities = [e for e in document.get("entities", []) if e.get("artifact_id") == artifact_id]
        artifact_rels = [r for r in document.get("relationships", []) if r.get("artifact_id") == artifact_id]
        artifact_rules = [r for r in document.get("business_rules", []) if r.get("artifact_id") == artifact_id]
        source_evidence = evidence_by_artifact.get(artifact_id, [])

        score = _score(
            has_source_evidence=bool(source_evidence),
            entity_count=len(artifact_entities),
            relationship_count=len(artifact_rels),
            rule_count=len(artifact_rules),
        )
        findings.append(
            {
                "artifact_id": artifact_id,
                "confidence": score,
                "confidence_level": _level(score),
                "evidence_count": len(source_evidence),
                "entities_with_direct_evidence": sum(entity_evidence.get(e.get("id"), 0) > 0 for e in artifact_entities),
                "relationships_with_direct_evidence": sum(relationship_evidence.get(r.get("id"), 0) > 0 for r in artifact_rels),
                "business_rules_with_direct_evidence": sum(rule_evidence.get(r.get("id"), 0) > 0 for r in artifact_rules),
                "validation_notes": _notes(score, bool(source_evidence), len(artifact_rules)),
            }
        )

    return {
        "artifact_assessments": findings,
        "summary": {
            "artifacts": len(findings),
            "high_confidence": sum(item["confidence_level"] == "HIGH" for item in findings),
            "medium_confidence": sum(item["confidence_level"] == "MEDIUM" for item in findings),
            "low_confidence": sum(item["confidence_level"] == "LOW" for item in findings),
        },
    }


def _score(*, has_source_evidence: bool, entity_count: int, relationship_count: int, rule_count: int) -> float:
    score = 0.65 if has_source_evidence else 0.45
    if entity_count:
        score += 0.10
    if relationship_count:
        score += 0.10
    if rule_count:
        score += 0.05
    return min(round(score, 2), 0.95)


def _level(score: float) -> str:
    if score >= 0.80:
        return "HIGH"
    if score >= 0.60:
        return "MEDIUM"
    return "LOW"


def _notes(score: float, has_source_evidence: bool, rule_count: int) -> list[str]:
    notes = []
    if not has_source_evidence:
        notes.append("No explicit evidence item is attached to this artifact.")
    if rule_count:
        notes.append("Business rules are present and should retain source evidence during enrichment.")
    if score < 0.80:
        notes.append("LLM review or deeper artifact analysis can increase confidence.")
    return notes
