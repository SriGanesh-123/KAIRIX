"""Reconcile deterministic parser knowledge with LLM findings.

The reconciliation layer is read-only with respect to source artifacts. It does
not rewrite parser facts or source code; it records agreements, unresolved
claims, duplicate candidates, evidence coverage, and a confidence assessment.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, Iterable


def normalize_name(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", str(value).upper()).strip("_")


def build_reconciliation(
    document: Dict[str, Any],
    reviews: Iterable[Dict[str, Any]] | None = None,
    evidence_validation: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build an evidence-aware reconciliation between facts and LLM findings.

    Deterministic entities/relationships/rules remain the source-of-fact layer.
    LLM claims are assessed as SUPPORTED, UNVERIFIED, or POTENTIAL_CONFLICT;
    they are never silently promoted to deterministic facts.
    """
    review_by_artifact = {
        item.get("artifact_id"): item
        for item in (reviews or [])
        if item.get("artifact_id")
    }
    evidence_by_artifact = {
        item.get("artifact_id"): item
        for item in (evidence_validation or {}).get("artifact_assessments", [])
        if item.get("artifact_id")
    }

    entities = document.get("entities", [])
    relationships = document.get("relationships", [])
    rules = document.get("business_rules", [])

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for entity in entities:
        entity_type = str(entity.get("entity_type", "UNKNOWN"))
        name = normalize_name(entity.get("name", ""))
        if name:
            groups[(entity_type, name)].append(entity)

    aliases: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for (entity_type, name), items in groups.items():
        ids = [item["id"] for item in items if item.get("id")]
        aliases.append({"entity_type": entity_type, "normalized_name": name, "entity_ids": ids})
        if len(ids) > 1:
            duplicates.append({
                "entity_type": entity_type,
                "canonical_name": name,
                "entity_ids": ids,
                "status": "DUPLICATE_CANDIDATE",
            })

    reference_matches = _match_references(document)
    claim_assessments = _assess_llm_claims(document, review_by_artifact, evidence_by_artifact)

    artifact_assessments: list[dict[str, Any]] = []
    for artifact in document.get("artifacts", []):
        artifact_id = artifact.get("id")
        evidence = evidence_by_artifact.get(artifact_id, {})
        claims = [item for item in claim_assessments if item.get("artifact_id") == artifact_id]
        review = review_by_artifact.get(artifact_id, {})
        llm_confidence = _safe_float(review.get("confidence"), 0.0)
        evidence_confidence = _safe_float(evidence.get("confidence"), 0.0)
        unresolved = sum(item.get("status") == "UNVERIFIED" for item in claims)
        conflicts = sum(item.get("status") == "POTENTIAL_CONFLICT" for item in claims)
        confidence = _reconciled_confidence(llm_confidence, evidence_confidence, unresolved, conflicts)
        artifact_assessments.append({
            "artifact_id": artifact_id,
            "file_name": artifact.get("file_name"),
            "status": "CONFLICT" if conflicts else ("PARTIAL" if unresolved else "RECONCILED"),
            "llm_confidence": llm_confidence,
            "evidence_confidence": evidence_confidence,
            "reconciled_confidence": confidence,
            "confidence_level": _level(confidence),
            "supported_claims": sum(item.get("status") == "SUPPORTED" for item in claims),
            "unverified_claims": unresolved,
            "potential_conflicts": conflicts,
            "claim_assessment_count": len(claims),
        })

    canonical_knowledge = _build_canonical_knowledge(document, artifact_assessments, duplicates, reference_matches)

    return {
        "schema_version": "2.1",
        "normalized_entity_groups": aliases,
        "duplicate_groups": duplicates,
        "reference_matches": reference_matches,
        "claim_assessments": claim_assessments,
        "artifact_assessments": artifact_assessments,
        "canonical_knowledge": canonical_knowledge,
        "summary": {
            "normalized_groups": len(aliases),
            "duplicate_groups": len(duplicates),
            "reference_matches": len(reference_matches),
            "claims_assessed": len(claim_assessments),
            "claims_supported": sum(item.get("status") == "SUPPORTED" for item in claim_assessments),
            "claims_unverified": sum(item.get("status") == "UNVERIFIED" for item in claim_assessments),
            "potential_conflicts": sum(item.get("status") == "POTENTIAL_CONFLICT" for item in claim_assessments),
            "artifacts_reconciled": sum(item.get("status") == "RECONCILED" for item in artifact_assessments),
            "artifacts_partial": sum(item.get("status") == "PARTIAL" for item in artifact_assessments),
            "artifacts_with_conflicts": sum(item.get("status") == "CONFLICT" for item in artifact_assessments),
        },
    }


def _assess_llm_claims(document: Dict[str, Any], review_by_artifact: Dict[str, Dict[str, Any]], evidence_by_artifact: Dict[str, Dict[str, Any]]) -> list[dict[str, Any]]:
    entities_by_artifact: dict[str, list[str]] = defaultdict(list)
    relationships_by_artifact: dict[str, list[str]] = defaultdict(list)
    rules_by_artifact: dict[str, list[str]] = defaultdict(list)
    for item in document.get("entities", []):
        entities_by_artifact[item.get("artifact_id", "")].append(normalize_name(item.get("name", "")))
    for item in document.get("relationships", []):
        relationships_by_artifact[item.get("artifact_id", "")].append(normalize_name(item.get("relationship_type", "")))
    for item in document.get("business_rules", []):
        rules_by_artifact[item.get("artifact_id", "")].append(normalize_name(item.get("name", "")))

    assessments: list[dict[str, Any]] = []
    fields = ("key_findings", "dependencies", "business_rules", "evidence_candidates")
    for artifact_id, review in review_by_artifact.items():
        evidence = evidence_by_artifact.get(artifact_id, {})
        source_evidence = int(evidence.get("evidence_count", 0) or 0)
        for field in fields:
            values = review.get(field, []) or []
            if isinstance(values, str):
                values = [values]
            for value in values:
                claim = _claim_text(value)
                normalized_claim = normalize_name(claim)
                status = _classify_claim(normalized_claim, entities_by_artifact.get(artifact_id, []), relationships_by_artifact.get(artifact_id, []), rules_by_artifact.get(artifact_id, []), source_evidence > 0)
                claim_confidence = {"SUPPORTED": 0.90, "POTENTIAL_CONFLICT": 0.20, "UNVERIFIED": 0.40}.get(status, 0.0)
                assessments.append({
                    "artifact_id": artifact_id,
                    "field": field,
                    "claim": claim,
                    "status": status,
                    "claim_confidence": claim_confidence,
                    "evidence_count": source_evidence,
                    "reason": _claim_reason(status, source_evidence),
                })
    return assessments


def _classify_claim(claim: str, entity_names: list[str], relationship_types: list[str], rule_names: list[str], has_source_evidence: bool) -> str:
    if not claim:
        return "UNVERIFIED"
    known_terms = [term for term in (*entity_names, *relationship_types, *rule_names) if term]
    if any(len(term) >= 4 and term in claim for term in known_terms):
        return "SUPPORTED"
    if has_source_evidence:
        return "UNVERIFIED"
    return "UNVERIFIED"


def _claim_reason(status: str, evidence_count: int) -> str:
    if status == "SUPPORTED":
        return "LLM claim references deterministic artifact knowledge."
    if evidence_count:
        return "No deterministic match was found; source evidence exists and should be checked by validation/investigation."
    return "No deterministic match or direct artifact evidence was found."


def _build_canonical_knowledge(document: Dict[str, Any], artifact_assessments: list[dict[str, Any]], duplicates: list[dict[str, Any]], reference_matches: list[dict[str, Any]]) -> Dict[str, Any]:
    confidence_by_artifact = {item["artifact_id"]: item["reconciled_confidence"] for item in artifact_assessments if item.get("artifact_id")}

    def canonical_fact(item: dict[str, Any]) -> dict[str, Any]:
        """Preserve original deterministic confidence; never overwrite it."""
        source_confidence = item.get("confidence")
        result = dict(item)
        result["reconciliation_status"] = "CANONICAL_FACT"
        result["source_confidence"] = source_confidence
        result["reconciled_confidence"] = confidence_by_artifact.get(item.get("artifact_id"), 0.0)
        result["confidence_model"] = "source_confidence_preserved_plus_reconciliation_confidence"
        return result

    return {
        "entities": [canonical_fact(entity) for entity in document.get("entities", [])],
        "relationships": [canonical_fact(relationship) for relationship in document.get("relationships", [])],
        "business_rules": [canonical_fact(rule) for rule in document.get("business_rules", [])],
        "duplicate_candidates": duplicates,
        "reference_matches": reference_matches,
    }


def _match_references(document: Dict[str, Any]) -> list[dict[str, Any]]:
    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entity in document.get("entities", []):
        if entity.get("entity_type") != "REFERENCE":
            by_name[normalize_name(entity.get("name", ""))].append(entity)
    matches = []
    for reference in document.get("entities", []):
        if reference.get("entity_type") != "REFERENCE":
            continue
        key = normalize_name(reference.get("name", ""))
        candidates = by_name.get(key, [])
        if candidates:
            matches.append({
                "reference_entity_id": reference["id"],
                "candidate_entity_ids": [item["id"] for item in candidates],
                "normalized_name": key,
                "confidence": 0.75 if len(candidates) == 1 else 0.55,
            })
    return matches


def _claim_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "; ".join(f"{key}: {item}" for key, item in value.items())
    return str(value)


def _safe_float(value: Any, default: float) -> float:
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return default


def _reconciled_confidence(llm: float, evidence: float, unresolved: int, conflicts: int) -> float:
    if llm <= 0 and evidence <= 0:
        base = 0.0
    elif llm <= 0:
        base = evidence
    elif evidence <= 0:
        base = llm
    else:
        base = (llm * 0.55) + (evidence * 0.45)
    base -= min(unresolved * 0.03, 0.20)
    base -= min(conflicts * 0.10, 0.30)
    return round(max(0.0, min(base, 1.0)), 2)


def _level(score: float) -> str:
    if score >= 0.80:
        return "HIGH"
    if score >= 0.60:
        return "MEDIUM"
    return "LOW"
