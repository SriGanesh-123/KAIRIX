"""Standalone Relationship Discovery Agent."""

from __future__ import annotations

from typing import Any, Dict

from .discover import discover
from .schema import RELATIONSHIP_TYPES, SCHEMA_VERSION


class RelationshipDiscoveryAgent:
    """Discover relationships from canonical metadata without source mutation."""

    VERSION = "1.0.0"

    def run(self, canonical_metadata: Dict[str, Any]) -> Dict[str, Any]:
        result = discover(canonical_metadata)
        result.update({
            "agent": {
                "name": "relationship_discovery",
                "version": self.VERSION,
                "artifact_specific_hardcoding": False,
            },
            "source": {
                "canonical_schema_version": canonical_metadata.get("schema_version", SCHEMA_VERSION),
                "artifact_count": len(canonical_metadata.get("artifacts", [])),
            },
            "relationship_types": sorted(RELATIONSHIP_TYPES),
            "safety": {
                "source_modified": False,
                "artifact_specific_hardcoding": False,
                "unsupported_inferences_are_unverified": True,
            },
        })
        return result
