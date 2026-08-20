"""Build technology-neutral knowledge profiles from canonical metadata."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict


def build_artifact_profiles(document: Dict[str, Any]) -> list[Dict[str, Any]]:
    entities = document.get("entities", [])
    relationships = document.get("relationships", [])
    evidence = document.get("evidence", [])
    rules = document.get("business_rules", [])

    entity_by_artifact: dict[str, list[dict[str, Any]]] = {}
    rel_by_artifact: dict[str, list[dict[str, Any]]] = {}
    evidence_by_artifact: dict[str, list[dict[str, Any]]] = {}
    rules_by_artifact: dict[str, list[dict[str, Any]]] = {}

    for item in entities:
        entity_by_artifact.setdefault(item.get("artifact_id", ""), []).append(item)
    for item in relationships:
        rel_by_artifact.setdefault(item.get("artifact_id", ""), []).append(item)
    for item in evidence:
        evidence_by_artifact.setdefault(item.get("artifact_id", ""), []).append(item)
    for item in rules:
        rules_by_artifact.setdefault(item.get("artifact_id", ""), []).append(item)

    profiles = []
    for artifact in document.get("artifacts", []):
        artifact_id = artifact["id"]
        items = entity_by_artifact.get(artifact_id, [])
        rels = rel_by_artifact.get(artifact_id, [])
        evs = evidence_by_artifact.get(artifact_id, [])
        brs = rules_by_artifact.get(artifact_id, [])
        type_counts = Counter(item.get("entity_type", "UNKNOWN") for item in items)
        relationship_counts = Counter(item.get("relationship_type", "RELATED_TO") for item in rels)

        profiles.append(
            {
                "artifact_id": artifact_id,
                "source_type": artifact.get("source_type"),
                "file_name": artifact.get("file_name"),
                "purpose": _infer_purpose(artifact, type_counts, relationship_counts),
                "entity_counts": dict(type_counts),
                "relationship_counts": dict(relationship_counts),
                "entity_count": len(items),
                "relationship_count": len(rels),
                "evidence_count": len(evs),
                "business_rule_count": len(brs),
                "key_entities": [
                    {"id": item["id"], "type": item["entity_type"], "name": item["name"]}
                    for item in items[:20]
                ],
                "business_rules": [rule.get("name") for rule in brs[:20]],
            }
        )
    return profiles


def _infer_purpose(
    artifact: Dict[str, Any],
    entity_counts: Counter,
    relationship_counts: Counter,
) -> str:
    source_type = artifact.get("source_type")
    if source_type == "cobol":
        if entity_counts.get("FILE", 0) and entity_counts.get("RECORD", 0):
            return "COBOL program with file and record processing"
        return "COBOL application program"
    if source_type == "sql":
        if entity_counts.get("TABLE", 0) > 1:
            return "SQL transformation or query over relational tables"
        return "SQL script over relational data"
    if source_type == "ssis":
        return "SSIS package containing ETL tasks and connections"
    return "Legacy application artifact"
