"""Data-driven relationship discovery primitives."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

from .schema import relationship


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _entity_values(entity: Dict[str, Any]) -> Iterable[str]:
    for value in entity.values():
        if isinstance(value, (str, int)) and value:
            yield str(value)


def _artifact_ids(canonical: Dict[str, Any]) -> set[str]:
    return {str(a.get("artifact_id")) for a in canonical.get("artifacts", []) if a.get("artifact_id")}


def _normalize_existing(canonical: Dict[str, Any]) -> List[Dict[str, Any]]:
    output = []
    for rel in canonical.get("relationships", []):
        source = rel.get("source") or rel.get("source_id") or rel.get("from") or rel.get("from_id")
        target = rel.get("target") or rel.get("target_id") or rel.get("to") or rel.get("to_id")
        relation_type = rel.get("relationship") or rel.get("type") or rel.get("relation")
        if source and target and relation_type:
            output.append(relationship(
                str(source), str(relation_type), str(target),
                evidence=rel.get("evidence", []),
                confidence=float(rel.get("confidence", rel.get("source_confidence", 0.0)) or 0.0),
                discovery_method=rel.get("discovery_method", "canonical_metadata"),
                validation_status=rel.get("validation_status", "SUPPORTED"),
            ))
    return output


def _cross_artifact(canonical: Dict[str, Any], existing: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract only explicit cross-artifact references; never infer from names alone."""
    artifacts = canonical.get("artifacts", [])
    ids = _artifact_ids(canonical)
    names = {_norm(a.get("file_name")): str(a.get("artifact_id")) for a in artifacts if a.get("file_name") and a.get("artifact_id")}
    known = {(x["source"], x["relationship"], x["target"]) for x in existing}
    found: List[Dict[str, Any]] = []

    def resolve(value: Any) -> str | None:
        if value is None:
            return None
        value = str(value)
        return value if value in ids else names.get(_norm(value))

    def add(source: Any, target: Any, relation_type: str, evidence: Any, confidence: float) -> None:
        source_id, target_id = resolve(source), resolve(target)
        if not source_id or not target_id or source_id == target_id:
            return
        key = (source_id, relation_type, target_id)
        if key in known:
            return
        found.append(relationship(source_id, relation_type, target_id,
                                   evidence=[evidence] if not isinstance(evidence, list) else evidence,
                                   confidence=confidence,
                                   discovery_method="canonical_cross_artifact_reference",
                                   validation_status="UNVERIFIED"))
        known.add(key)

    # Inspect structured reference-bearing values generically. This intentionally
    # accepts common metadata shapes without encoding project-specific artifacts.
    reference_keys = {"artifact_id", "source_artifact_id", "target_artifact_id", "file_name", "artifact_reference", "depends_on", "dependency_ids", "references"}
    for container_name in ("artifacts", "entities", "relationships", "business_rules", "claims"):
        for item in canonical.get(container_name, []):
            if not isinstance(item, dict):
                continue
            source = item.get("artifact_id") or item.get("source_artifact_id") or item.get("source")
            for key, value in item.items():
                if key not in reference_keys and not key.endswith("_artifact_id"):
                    continue
                values = value if isinstance(value, list) else [value]
                for ref in values:
                    target = ref.get("artifact_id") if isinstance(ref, dict) else ref
                    if isinstance(ref, dict):
                        target = target or ref.get("target_artifact_id") or ref.get("file_name")
                        rel_type = ref.get("relationship") or ref.get("type") or "REFERENCES"
                        evidence = ref.get("evidence") or {"container": container_name, "field": key, "value": ref}
                    else:
                        rel_type = "DEPENDS_ON"
                        evidence = {"container": container_name, "field": key, "value": ref}
                    add(source, target, rel_type, evidence, 0.50)
    return found


def discover(canonical: Dict[str, Any]) -> Dict[str, Any]:
    existing = _normalize_existing(canonical)
    discovered = existing + _cross_artifact(canonical, existing)
    unique: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str]] = set()
    for item in discovered:
        key = (item["source"], item["relationship"], item["target"])
        if key not in seen:
            seen.add(key)
            unique.append(item)

    by_type: Dict[str, int] = defaultdict(int)
    by_method: Dict[str, int] = defaultdict(int)
    for item in unique:
        by_type[item["relationship"]] += 1
        by_method[item["discovery_method"]] += 1
    return {
        "schema_version": "1.0",
        "relationships": unique,
        "summary": {
            "relationships_discovered": len(unique),
            "supported": sum(x["validation_status"] == "SUPPORTED" for x in unique),
            "unverified": sum(x["validation_status"] == "UNVERIFIED" for x in unique),
            "conflicts": sum(x["validation_status"] == "CONFLICT" for x in unique),
            "by_type": dict(sorted(by_type.items())),
            "by_method": dict(sorted(by_method.items())),
        },
    }
