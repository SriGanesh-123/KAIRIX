"""Build a stable downstream canonical metadata contract.

This layer is read-only with respect to source artifacts. It converts the
reconciled knowledge into a versioned, explicit contract for relationship
discovery, graph storage, vector indexing, validation and investigation.
"""

from __future__ import annotations

from typing import Any, Dict


SCHEMA_VERSION = "1.0"


def build_canonical_metadata(document: Dict[str, Any], reconciliation: Dict[str, Any]) -> Dict[str, Any]:
    """Create canonical metadata without modifying original source facts."""
    canonical = reconciliation.get("canonical_knowledge", {})
    assessments = reconciliation.get("artifact_assessments", [])
    assessment_by_id = {x.get("artifact_id"): x for x in assessments if x.get("artifact_id")}

    artifacts = []
    for artifact in document.get("artifacts", []):
        artifact_id = artifact.get("id")
        assessment = assessment_by_id.get(artifact_id, {})
        artifacts.append({
            "artifact_id": artifact_id,
            "file_name": artifact.get("file_name"),
            "source_type": artifact.get("source_type"),
            "artifact_kind": artifact.get("artifact_kind"),
            "status": assessment.get("status", "UNASSESSED"),
            "reconciled_confidence": assessment.get("reconciled_confidence", 0.0),
            "confidence_level": assessment.get("confidence_level", "LOW"),
            "supported_claims": assessment.get("supported_claims", 0),
            "unverified_claims": assessment.get("unverified_claims", 0),
            "potential_conflicts": assessment.get("potential_conflicts", 0),
        })

    claims = []
    for claim in reconciliation.get("claim_assessments", []):
        claims.append({
            "artifact_id": claim.get("artifact_id"),
            "field": claim.get("field"),
            "claim": claim.get("claim"),
            "status": claim.get("status"),
            "claim_confidence": claim.get("claim_confidence", 0.0),
            "evidence_count": claim.get("evidence_count", 0),
            "reason": claim.get("reason", ""),
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "contract": {
            "name": "kairix_canonical_metadata",
            "version": SCHEMA_VERSION,
            "source_read_only": True,
            "purpose": "Stable downstream contract for validation, relationship discovery, graph/vector indexing and investigation.",
        },
        "source": {
            "canonical_schema_version": document.get("schema_version", "1.0"),
            "artifact_count": len(document.get("artifacts", [])),
        },
        "artifacts": artifacts,
        "entities": canonical.get("entities", []),
        "relationships": canonical.get("relationships", []),
        "business_rules": canonical.get("business_rules", []),
        "claims": claims,
        "duplicate_candidates": canonical.get("duplicate_candidates", []),
        "reference_matches": canonical.get("reference_matches", []),
        "knowledge_gaps": [
            {
                "artifact_id": item.get("artifact_id"),
                "type": item.get("type"),
                "reason": item.get("reason"),
            }
            for item in reconciliation.get("artifact_assessments", [])
            if item.get("status") in {"PARTIAL", "CONFLICT"}
        ],
        "code_fix_policy": {
            "source_modified": False,
            "action_owner": "DEVELOPER",
            "description": "Potential code fixes are findings for developers; canonical metadata never modifies source code.",
        },
        "statistics": {
            "artifacts": len(artifacts),
            "entities": len(canonical.get("entities", [])),
            "relationships": len(canonical.get("relationships", [])),
            "business_rules": len(canonical.get("business_rules", [])),
            "claims": len(claims),
            "duplicate_candidates": len(canonical.get("duplicate_candidates", [])),
            "reference_matches": len(canonical.get("reference_matches", [])),
            "partial_or_conflicted_artifacts": sum(1 for x in assessments if x.get("status") in {"PARTIAL", "CONFLICT"}),
        },
    }
