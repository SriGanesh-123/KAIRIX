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
    text = str(value or "").strip().lower()
    return re.sub(r"[^a-z0-9]+", "", text)


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
    """Find only evidence-backed cross-artifact links already represented in metadata.

    We intentionally do not guess a relationship from string similarity alone.
    Matching names are retained as candidate evidence, but remain UNVERIFIED.
    """
    artifacts = canonical.get("artifacts", [])
    if len(artifacts) < 2:
        return []
    entity_index = _entity_index(canonical.get("entities", []))
    known_pairs = {(r.get("source"), r.get("relationship"), r.get("target")) for r in existing}
    candidates: List[Dict[str, Any]] = []

    artifact_ids = {a.get("artifact_id") for a in artifacts}
    for left in artifacts:
        left_id = left.get("artifact_id")
        left_names = [_norm(left.get("artifact_id")), _norm(left.get("file_name"))]
        left_names = [x for x in left_names if x]
        for name in left_names:
            for entity in entity_index.get(name, []):
                entity_artifact = entity.get("artifact_id") or entity.get("source_artifact_id")
                if entity_artifact and entity_artifact != left_id and entity_artifact in artifact_ids:
                    key = (left_id, "REFERENCES", entity.get("id") or entity.get("entity_id"))
                    if key not in known_pairs:
                        candidates.append({
                            "source": left_id,
                            "relationship": "REFERENCES",
                            "target": entity.get("id") or entity.get("entity_id") or entity.get("name"),
                            "evidence": [{"type": "cross_artifact_name_match", "artifact_id": left_id, "matched_value": name}],
                            "confidence": 0.50,
                            "discovery_method": "cross_artifact_matching",
                            "validation_status": "UNVERIFIED",
                        })
    return candidates


def discover_relationships(canonical_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Return a structured relationship-discovery result from canonical metadata."""
    canonical = canonical_metadata
    existing = _existing_relationships(canonical)
    candidates = _cross_artifact_candidates(canonical, existing)
    all_relationships = existing + candidates

    # Deduplicate without assuming any project-specific identifiers.
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
        "schema_version": "1.0",
        "agent": {"name": "relationship_discovery", "version": "1.0.0", "artifact_specific_hardcoding": False},
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
