"""Generic provenance helpers for canonical knowledge."""

from __future__ import annotations

from typing import Any, Dict


def artifact_ref(artifact: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "artifact_id": artifact.get("artifact_id"),
        "file_name": artifact.get("file_name"),
        "source_type": artifact.get("source_type"),
    }


def attach_entity_provenance(entity: Dict[str, Any], artifact: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(entity)
    result["provenance"] = {
        "artifact": artifact_ref(artifact),
        "source_location": entity.get("source_location") or entity.get("location"),
    }
    return result


def attach_relationship_provenance(relationship: Dict[str, Any], source_artifact: Dict[str, Any] | None = None,
                                   target_artifact: Dict[str, Any] | None = None) -> Dict[str, Any]:
    result = dict(relationship)
    result["provenance"] = {
        "source_artifact": artifact_ref(source_artifact or {}),
        "target_artifact": artifact_ref(target_artifact or {}),
        "evidence": result.get("evidence", []),
    }
    return result
