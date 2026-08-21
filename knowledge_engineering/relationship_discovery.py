"""Discover evidence-backed relationships from canonical metadata.

This module is deliberately data-driven: it never contains artifact-specific
names, tables, columns, or relationships. It derives candidates from the
canonical metadata supplied by the Knowledge Engineering layer.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple


RELATIONSHIP_TYPES = {
    "CALLS", "READS_FROM", "WRITES_TO", "DEPENDS_ON", "DERIVED_FROM",
    "MAPS_TO", "TRANSFORMS", "USES", "IMPLEMENTS", "CONTAINS",
    "REFERENCES", "LOADS_TO",
}


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _name_values(entity: Dict[str, Any]) -> List[str]:
    values: List[str] = []
    for key in ("name", "canonical_name", "normalized_name", "file_name", "table", "table_name", "column", "column_name", "id", "entity_id"):
        value = entity.get(key)
        if value:
            values.append(str(value))
    return list(dict.fromkeys(values))


def _entity_index(entities: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    index: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for entity in entities:
        for value in _name_values(entity):
            normalized = _norm(value)
            if normalized:
                index[normalized].append(entity)
    return index


def _existing_relationships(canonical: Dict[str, Any]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for rel in canonical.get("relationships", []):
        result.append({
            "source": rel.get("source") or rel.get("source_id") or rel.get("from") or rel.get("from_id"),
            "relationship": rel.get("relationship") or rel.get("type") or rel.get("relation"),
            "target": rel.get("target") or rel.get("target_id") or rel.get("to") or rel.get("to_id"),
            "evidence": rel.get("evidence", []),
            "confidence": rel.get("confidence", rel.get("source_confidence", 0.0)),
            "discovery_method": rel.get("discovery_method", "canonical_metadata"),
            "validation_status": rel.get("validation_status", "SUPPORTED"),
        })
    return result


def _artifact_name_index(canonical: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for artifact in canonical.get("artifacts", []):
        for value in (artifact.get("artifact_id"), artifact.get("file_name")):
            key = _norm(value)
            if key:
                result[key] = artifact
    return result


def _cross_artifact_candidates(canonical: Dict[str, Any], existing: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Discover cross-artifact candidates from explicit metadata references.

    The current canonical contract may contain artifact/entity references under
    relationships, evidence, or dependency-like fields. Those references are
    promoted into relationship candidates only when they identify two distinct
    artifacts. Plain string similarity is never enough to make a relationship.
    """
    artifacts = canonical.get("artifacts", [])
    if len(artifacts) < 2:
        return []

    artifact_by_id = {a.get("artifact_id"): a for a in artifacts if a.get("artifact_id")}
    artifact_by_name: Dict[str, str] = {}
    for artifact in artifacts:
        aid = artifact.get("artifact_id")
        for value in (artifact.get("file_name"), artifact.get("artifact_id")):
            key = _norm(value)
            if aid and key:
                artifact_by_name[key] = aid

    known_pairs = {(r.get("source"), r.get("relationship"), r.get("target")) for r in existing}
    candidates: List[Dict[str, Any]] = []

    def add_candidate(source: Any, relationship: str, target: Any, evidence: Any, confidence: float = 0.55) -> None:
        source_id = source if source in artifact_by_id else artifact_by_name.get(_norm(source))
        target_id = target if target in artifact_by_id else artifact_by_name.get(_norm(target))
        if not source_id or not target_id or source_id == target_id:
            return
        key = (source_id, relationship, target_id)
        if key in known_pairs:
            return
        candidates.append({
            "source": source_id,
            "relationship": relationship,
            "target": target_id,
            "evidence": evidence if isinstance(evidence, list) else [evidence],
            "confidence": confidence,
            "discovery_method": "canonical_cross_artifact_reference",
            "validation_status": "UNVERIFIED",
        })
        known_pairs.add(key)

    # Inspect generic reference-bearing fields from artifacts and existing
    # metadata. No project-specific field names are required beyond common
    # structural names already present in the canonical contract.
    reference_fields = ("depends_on", "dependency_ids", "source_artifact_id", "target_artifact_id", "artifact_references", "references")
    for artifact in artifacts:
        source_id = artifact.get("artifact_id")
        for field in reference_fields:
            values = artifact.get(field, [])
            if values is None:
                continue
            if not isinstance(values, list):
                values = [values]
            for value in values:
                if isinstance(value, dict):
                    target = value.get("artifact_id") or value.get("target_artifact_id") or value.get("file_name") or value.get("name")
                    relationship = value.get("relationship") or value.get("type") or "DEPENDS_ON"
                    evidence = value.get("evidence") or {"field": field, "value": value}
                else:
                    target = value
                    relationship = "DEPENDS_ON"
                    evidence = {"field": field, "value": value}
                add_candidate(source_id, relationship, target, evidence)

    # Dependency/reference information can also appear on canonical entities.
    for entity in canonical.get("entities", []):
        source_artifact = entity.get("artifact_id") or entity.get("source_artifact_id")
        for field in reference_fields:
            values = entity.get(field, [])
            if values is None:
                continue
            if not isinstance(values, list):
                values = [values]
            for value in values:
                target = value.get("artifact_id") if isinstance(value, dict) else value
                if isinstance(value, dict):
                    target = target or value.get("target_artifact_id") or value.get("file_name")
                    relationship = value.get("relationship") or value.get("type") or "REFERENCES"
                    evidence = value.get("evidence") or {"field": field, "entity_id": entity.get("id"), "value": value}
                else:
                    relationship = "REFERENCES"
                    evidence = {"field": field, "entity_id": entity.get("id"), "value": value}
                add_candidate(source_artifact, relationship, target, evidence, 0.50)

    return candidates


def discover_relationships(canonical_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Return a structured relationship-discovery result from canonical metadata."""
    canonical = canonical_metadata
    existing = _existing_relationships(canonical)
    candidates = _cross_artifact_candidates(canonical, existing)
    all_relationships = existing + candidates

    seen: set[Tuple[Any, Any, Any]] = set()
    relationships: List[Dict[str, Any]] = []
    for rel in all_relationships:
        key = (rel.get("source"), rel.get("relationship"), rel.get("target"))
        if not key[0] or not key[1] or not key[2] or key in seen:
            continue
        seen.add(key)
        relationships.append(rel)

    supported = sum(r.get("validation_status") == "SUPPORTED" for r in relationships)
    unverified = sum(r.get("validation_status") == "UNVERIFIED" for r in relationships)
    conflicts = sum(r.get("validation_status") == "CONFLICT" for r in relationships)
    by_type: Dict[str, int] = defaultdict(int)
    by_method: Dict[str, int] = defaultdict(int)
    for rel in relationships:
        by_type[str(rel.get("relationship"))] += 1
        by_method[str(rel.get("discovery_method"))] += 1

    return {
        "schema_version": "1.1",
        "agent": {"name": "relationship_discovery", "version": "1.1.0", "artifact_specific_hardcoding": False},
        "source": {"canonical_schema_version": canonical.get("schema_version", "1.0"), "artifact_count": len(canonical.get("artifacts", []))},
        "relationship_types": sorted(RELATIONSHIP_TYPES),
        "relationships": relationships,
        "summary": {
            "relationships_discovered": len(relationships),
            "supported": supported,
            "unverified": unverified,
            "conflicts": conflicts,
            "by_type": dict(sorted(by_type.items())),
            "by_method": dict(sorted(by_method.items())),
        },
        "safety": {
            "source_modified": False,
            "artifact_specific_hardcoding": False,
            "unsupported_inferences_are_unverified": True,
        },
    }
