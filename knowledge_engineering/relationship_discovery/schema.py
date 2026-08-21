"""Schemas and controlled vocabulary for relationship discovery."""

from __future__ import annotations

from typing import Any, Dict

SCHEMA_VERSION = "1.0"
RELATIONSHIP_TYPES = frozenset({
    "CALLS", "READS_FROM", "WRITES_TO", "DEPENDS_ON", "DERIVED_FROM",
    "MAPS_TO", "TRANSFORMS", "USES", "IMPLEMENTS", "CONTAINS",
    "REFERENCES", "LOADS_TO",
})


def relationship(source: str, relation: str, target: str, *, evidence: list | None = None,
                 confidence: float = 0.0, discovery_method: str = "unknown",
                 validation_status: str = "UNVERIFIED") -> Dict[str, Any]:
    return {
        "source": source,
        "relationship": relation,
        "target": target,
        "evidence": evidence or [],
        "confidence": confidence,
        "discovery_method": discovery_method,
        "validation_status": validation_status,
    }
