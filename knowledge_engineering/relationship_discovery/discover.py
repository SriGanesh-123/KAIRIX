"""Data-driven relationship discovery primitives."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

from .candidate_discovery import discover_reference_candidates
from .schema import relationship


def _entity_index(canonical: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(entity.get("id")): entity
        for entity in canonical.get("entities", [])
        if entity.get("id")
    }


def _artifact_for_entity(entity: Dict[str, Any]) -> str | None:
    artifact_id = entity.get("artifact_id")
    if artifact_id:
        return str(artifact_id)
    provenance = entity.get("provenance")
    if isinstance(provenance, dict):
        artifact = provenance.get("artifact")
        if isinstance(artifact, dict) and artifact.get("artifact_id"):
            return str(artifact["artifact_id"])
    return None


def _normalize_existing(canonical: Dict[str, Any]) -> List[Dict[str, Any]]:
    entities = _entity_index(canonical)
    output: List[Dict[str, Any]] = []

    for rel in canonical.get("relationships", []):
        source_id = rel.get("source_entity_id") or rel.get("source_id") or rel.get("source")
        target_id = rel.get("target_entity_id") or rel.get("target_id") or rel.get("target")
        relation_type = rel.get("relationship_type") or rel.get("relationship") or rel.get("type")
        if not source_id or not target_id or not relation_type:
            continue

        source_id = str(source_id)
        target_id = str(target_id)
        source_entity = entities.get(source_id, {})
        target_entity = entities.get(target_id, {})
        source_artifact = _artifact_for_entity(source_entity) or rel.get("artifact_id")
        target_artifact = _artifact_for_entity(target_entity) or rel.get("target_artifact_id")

        evidence = list(rel.get("evidence", [])) if isinstance(rel.get("evidence"), list) else []
        evidence_ids = rel.get("evidence_ids")
        if isinstance(evidence_ids, list):
            evidence.extend({"evidence_id": item} for item in evidence_ids)

        output.append({
            **relationship(
                source_id,
                str(relation_type),
                target_id,
                evidence=evidence,
                confidence=float(rel.get("reconciled_confidence", rel.get("confidence", 0.0)) or 0.0),
                discovery_method=rel.get("discovery_method", "canonical_metadata"),
                validation_status=("SUPPORTED" if rel.get("reconciliation_status") == "CANONICAL_FACT" else rel.get("validation_status", "UNVERIFIED")),
            ),
            "source_artifact_id": str(source_artifact) if source_artifact else None,
            "target_artifact_id": str(target_artifact) if target_artifact else None,
            "canonical_relationship_id": rel.get("id"),
        })
    return output


def _discover_cross_artifact(canonical: Dict[str, Any], existing: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Discover explicit cross-artifact edges from canonical metadata only."""
    entities = _entity_index(canonical)
    known = {(item["source"], item["relationship"], item["target"]) for item in existing}
    found: List[Dict[str, Any]] = []

    for rel in canonical.get("relationships", []):
        source_id = rel.get("source_entity_id") or rel.get("source_id") or rel.get("source")
        target_id = rel.get("target_entity_id") or rel.get("target_id") or rel.get("target")
        relation_type = rel.get("relationship_type") or rel.get("relationship") or rel.get("type")
        if not source_id or not target_id or not relation_type:
            continue
        source_entity = entities.get(str(source_id), {})
        target_entity = entities.get(str(target_id), {})
        source_artifact = _artifact_for_entity(source_entity) or rel.get("artifact_id")
        target_artifact = _artifact_for_entity(target_entity) or rel.get("target_artifact_id")
        if not source_artifact or not target_artifact or str(source_artifact) == str(target_artifact):
            continue
        key = (str(source_id), str(relation_type), str(target_id))
        if key in known:
            continue
        evidence_ids = rel.get("evidence_ids") if isinstance(rel.get("evidence_ids"), list) else []
        evidence = [{"evidence_id": item} for item in evidence_ids]
        evidence.append({"source_artifact_id": str(source_artifact), "target_artifact_id": str(target_artifact), "relationship_id": rel.get("id"), "relationship_type": relation_type})
        found.append({
            **relationship(str(source_id), str(relation_type), str(target_id), evidence=evidence,
                           confidence=float(rel.get("reconciled_confidence", rel.get("confidence", 0.0)) or 0.0),
                           discovery_method="canonical_entity_provenance",
                           validation_status="SUPPORTED" if rel.get("reconciliation_status") == "CANONICAL_FACT" else "UNVERIFIED"),
            "source_artifact_id": str(source_artifact),
            "target_artifact_id": str(target_artifact),
            "canonical_relationship_id": rel.get("id"),
        })
        known.add(key)

    found.extend(discover_reference_candidates(canonical, known))
    return found


def _validate_relationships(relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate without inventing evidence or upgrading ambiguous candidates."""
    validated: List[Dict[str, Any]] = []
    for item in relationships:
        result = dict(item)
        evidence = item.get("evidence", [])
        status = str(item.get("validation_status", "UNVERIFIED"))

        if status == "SUPPORTED":
            result["validation_status"] = "SUPPORTED"
        elif item.get("discovery_method") == "canonical_reference_match":
            confidence = float(item.get("confidence", 0.0) or 0.0)
            candidate_ids = []
            for evidence_item in evidence if isinstance(evidence, list) else []:
                if not isinstance(evidence_item, dict):
                    continue
                reference_match = evidence_item.get("reference_match")
                if isinstance(reference_match, dict) and reference_match.get("candidate_entity_id"):
                    candidate_ids.append(str(reference_match["candidate_entity_id"]))
            result["validation_status"] = (
                "SUPPORTED" if len(set(candidate_ids)) == 1 and confidence >= 0.75 else "UNVERIFIED"
            )
        elif status in {"UNVERIFIED", "CONFLICT"}:
            result["validation_status"] = status
        else:
            result["validation_status"] = "UNVERIFIED"

        if isinstance(evidence, list) and any(
            isinstance(evidence_item, dict) and evidence_item.get("conflict") is True
            for evidence_item in evidence
        ):
            result["validation_status"] = "CONFLICT"
        validated.append(result)
    return validated


def discover(canonical: Dict[str, Any]) -> Dict[str, Any]:
    existing = _normalize_existing(canonical)
    cross_artifact = _discover_cross_artifact(canonical, existing)

    all_relationships = _validate_relationships(existing + cross_artifact)
    unique: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str]] = set()
    for item in all_relationships:
        key = (item["source"], item["relationship"], item["target"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)

    cross_keys = {
        (item["source"], item["relationship"], item["target"])
        for item in cross_artifact
    }
    validated_cross_artifact = [
        item for item in unique
        if (item["source"], item["relationship"], item["target"]) in cross_keys
    ]

    by_type: Dict[str, int] = defaultdict(int)
    by_method: Dict[str, int] = defaultdict(int)
    for item in unique:
        by_type[item["relationship"]] += 1
        by_method[item["discovery_method"]] += 1

    return {
        "schema_version": "1.2",
        "relationships": unique,
        "cross_artifact_relationships": validated_cross_artifact,
        "summary": {
            "relationships_discovered": len(unique),
            "cross_artifact_discovered": len(validated_cross_artifact),
            "supported": sum(item["validation_status"] == "SUPPORTED" for item in unique),
            "unverified": sum(item["validation_status"] == "UNVERIFIED" for item in unique),
            "conflicts": sum(item["validation_status"] == "CONFLICT" for item in unique),
            "by_type": dict(sorted(by_type.items())),
            "by_method": dict(sorted(by_method.items())),
        },
    }
