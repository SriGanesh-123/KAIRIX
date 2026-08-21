"""Schema helpers for a provenance-preserving knowledge graph."""

from __future__ import annotations

from typing import Any, Dict

SCHEMA_VERSION = "1.0"
TRUSTED_STATUS = "SUPPORTED"
CANDIDATE_STATUS = "UNVERIFIED"
CONFLICT_STATUS = "CONFLICT"


def node_from_entity(entity: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": entity.get("id"),
        "type": entity.get("entity_type", "ENTITY"),
        "name": entity.get("name", ""),
        "artifact_id": entity.get("artifact_id"),
        "properties": entity.get("properties", {}),
        "source_confidence": entity.get("source_confidence"),
        "reconciled_confidence": entity.get("reconciled_confidence"),
    }


def edge_from_relationship(relationship: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": relationship.get("id"),
        "source_entity_id": relationship.get("source_entity_id", relationship.get("source")),
        "target_entity_id": relationship.get("target_entity_id", relationship.get("target")),
        "relationship_type": relationship.get("relationship_type", relationship.get("relationship")),
        "artifact_id": relationship.get("artifact_id"),
        "source_artifact_id": relationship.get("source_artifact_id"),
        "target_artifact_id": relationship.get("target_artifact_id"),
        "evidence_ids": relationship.get("evidence_ids", []),
        "evidence": relationship.get("evidence", []),
        "properties": relationship.get("properties", {}),
        "confidence": relationship.get("confidence", relationship.get("reconciled_confidence")),
        "validation_status": relationship.get("validation_status", "UNVERIFIED"),
        "discovery_method": relationship.get("discovery_method", "canonical_metadata"),
    }
