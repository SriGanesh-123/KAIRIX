"""Evidence-aware validation for discovered relationships."""

from __future__ import annotations

from typing import Any, Dict, List


def _entity_index(canonical: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(item.get("id")): item for item in canonical.get("entities", []) if item.get("id")}


def _artifact_index(canonical: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(item.get("id")): item for item in canonical.get("artifacts", []) if item.get("id")}


def _candidate_evidence(candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
    evidence = candidate.get("evidence", [])
    return evidence if isinstance(evidence, list) else []


def _has_reference_proof(evidence: List[Dict[str, Any]]) -> bool:
    """Return whether evidence explicitly identifies a canonical reference match."""
    for item in evidence:
        if not isinstance(item, dict):
            continue
        if item.get("reference_entity_id") or item.get("candidate_entity_id") or item.get("reference_match_id"):
            return True
        nested = item.get("reference_match")
        if isinstance(nested, dict) and (
            nested.get("reference_entity_id") or nested.get("candidate_entity_id")
        ):
            return True
    return False


def _has_explicit_cross_artifact_proof(evidence: List[Dict[str, Any]]) -> bool:
    """Return whether evidence explicitly records both artifact endpoints."""
    for item in evidence:
        if not isinstance(item, dict):
            continue
        if item.get("source_artifact_id") and item.get("target_artifact_id"):
            return str(item["source_artifact_id"]) != str(item["target_artifact_id"])
    return False


def _reference_match_cardinality(canonical: Dict[str, Any], source_id: str, target_id: str) -> int:
    """Return the number of candidates attached to the matching reference match."""
    for match in canonical.get("reference_matches", []):
        if not isinstance(match, dict):
            continue
        if str(match.get("reference_entity_id")) != str(source_id):
            continue
        candidate_ids = match.get("candidate_entity_ids", [])
        if not isinstance(candidate_ids, list):
            continue
        if str(target_id) in {str(item) for item in candidate_ids}:
            return len(candidate_ids)
    return 0


def validate_relationships(canonical: Dict[str, Any], relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate only newly discovered candidates; preserve canonical facts."""
    entities = _entity_index(canonical)
    artifacts = _artifact_index(canonical)
    validated: List[Dict[str, Any]] = []

    for candidate in relationships:
        item = dict(candidate)

        if candidate.get("validation_status") == "SUPPORTED" and candidate.get("discovery_method") == "canonical_metadata":
            item["validation_status"] = "SUPPORTED"
            item["validation_reason"] = "Canonical reconciliation marked this relationship as CANONICAL_FACT."
            validated.append(item)
            continue

        source = entities.get(str(candidate.get("source", "")), {})
        target = entities.get(str(candidate.get("target", "")), {})
        source_id = str(candidate.get("source", ""))
        target_id = str(candidate.get("target", ""))
        source_artifact = candidate.get("source_artifact_id") or source.get("artifact_id")
        target_artifact = candidate.get("target_artifact_id") or target.get("artifact_id")
        evidence = _candidate_evidence(candidate)

        has_known_cross_artifact_endpoints = (
            bool(source_artifact)
            and bool(target_artifact)
            and str(source_artifact) != str(target_artifact)
            and str(source_artifact) in artifacts
            and str(target_artifact) in artifacts
        )
        has_reference_proof = _has_reference_proof(evidence)
        has_explicit_cross_artifact_proof = _has_explicit_cross_artifact_proof(evidence)
        ambiguous_reference = has_reference_proof and _reference_match_cardinality(canonical, source_id, target_id) > 1
        is_canonical_explicit_relationship = candidate.get("discovery_method") == "canonical_metadata"

        if is_canonical_explicit_relationship and has_known_cross_artifact_endpoints:
            status = "SUPPORTED"
            reason = "Canonical relationship explicitly identifies endpoints in two known artifacts."
        elif has_known_cross_artifact_endpoints and has_reference_proof and not ambiguous_reference:
            status = "SUPPORTED"
            reason = "Canonical reference evidence resolves the relationship across two known artifacts."
        elif has_known_cross_artifact_endpoints and has_explicit_cross_artifact_proof and not has_reference_proof:
            status = "SUPPORTED"
            reason = "Canonical relationship evidence explicitly identifies endpoints in two known artifacts."
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
        "schema_version": "1.1",
        "relationships": validated,
        "summary": {
            "validated": len(validated),
            "supported": sum(item["validation_status"] == "SUPPORTED" for item in validated),
            "unverified": sum(item["validation_status"] == "UNVERIFIED" for item in validated),
            "conflicts": sum(item["validation_status"] == "CONFLICT" for item in validated),
        },
        "safety": {"artifact_specific_hardcoding": False, "source_modified": False},
    }
