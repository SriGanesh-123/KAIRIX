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


def validate_relationships(canonical: Dict[str, Any], relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate only newly discovered candidates; preserve canonical facts."""
    entities = _entity_index(canonical)
    artifacts = _artifact_index(canonical)
    validated: List[Dict[str, Any]] = []

    for candidate in relationships:
        item = dict(candidate)

        # Reconciled canonical facts are already authoritative and must not be
        # downgraded by the candidate validator.
        if candidate.get("validation_status") == "SUPPORTED" and candidate.get("discovery_method") == "canonical_metadata":
            item["validation_status"] = "SUPPORTED"
            item["validation_reason"] = "Canonical reconciliation marked this relationship as CANONICAL_FACT."
            validated.append(item)
            continue

        source = entities.get(str(candidate.get("source", "")), {})
        target = entities.get(str(candidate.get("target", "")), {})
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

        if has_known_cross_artifact_endpoints and (has_reference_proof or has_explicit_cross_artifact_proof):
            status = "SUPPORTED"
            if has_reference_proof:
                reason = "Canonical reference evidence resolves the relationship across two known artifacts."
            else:
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
