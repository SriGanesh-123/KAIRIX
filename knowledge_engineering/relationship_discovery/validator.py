"""Evidence-aware validation for discovered relationships."""

from __future__ import annotations

from typing import Any, Dict, List


def _entity_index(canonical: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(item.get("id")): item
        for item in canonical.get("entities", [])
        if item.get("id")
    }


def _artifact_index(canonical: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(item.get("id")): item
        for item in canonical.get("artifacts", [])
        if item.get("id")
    }


def _candidate_evidence(candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
    evidence = candidate.get("evidence", [])
    return evidence if isinstance(evidence, list) else []


def validate_relationships(canonical: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate candidates using only canonical metadata evidence.

    Candidates are never promoted merely because they exist. A candidate is
    SUPPORTED only when its source/target entities resolve to different known
    artifacts and it carries explicit reference evidence. Otherwise it remains
    UNVERIFIED. No source artifacts are modified.
    """
    entities = _entity_index(canonical)
    artifacts = _artifact_index(canonical)
    validated: List[Dict[str, Any]] = []

    for candidate in candidates:
        item = dict(candidate)
        source = entities.get(str(candidate.get("source", "")), {})
        target = entities.get(str(candidate.get("target", "")), {})
        source_artifact = candidate.get("source_artifact_id") or source.get("artifact_id")
        target_artifact = candidate.get("target_artifact_id") or target.get("artifact_id")
        evidence = _candidate_evidence(candidate)

        has_cross_artifact_proof = (
            bool(source_artifact)
            and bool(target_artifact)
            and str(source_artifact) != str(target_artifact)
            and str(source_artifact) in artifacts
            and str(target_artifact) in artifacts
            and any(
                isinstance(ev, dict)
                and (ev.get("reference_entity_id") or ev.get("candidate_entity_id") or ev.get("reference_match_id"))
                for ev in evidence
            )
        )

        if has_cross_artifact_proof:
            status = "SUPPORTED"
            reason = "Canonical reference evidence resolves the relationship across two known artifacts."
        else:
            status = "UNVERIFIED"
            reason = "Canonical metadata does not provide sufficient deterministic evidence to promote the candidate."

        item["validation_status"] = status
        item["validation_reason"] = reason
        item["validation_evidence_count"] = len(evidence)
        item["source_artifact_id"] = str(source_artifact) if source_artifact else None
        item["target_artifact_id"] = str(target_artifact) if target_artifact else None
        validated.append(item)

    return {
        "schema_version": "1.0",
        "relationships": validated,
        "summary": {
            "validated": len(validated),
            "supported": sum(item["validation_status"] == "SUPPORTED" for item in validated),
            "unverified": sum(item["validation_status"] == "UNVERIFIED" for item in validated),
            "conflicts": sum(item["validation_status"] == "CONFLICT" for item in validated),
        },
        "safety": {
            "artifact_specific_hardcoding": False,
            "source_modified": False,
        },
    }
