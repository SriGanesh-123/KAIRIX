"""Knowledge Engineering Agent orchestration.

The first implementation is deterministic and provider-neutral. An LLM reviewer can
be plugged in later without changing the canonical metadata contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Protocol

from .evidence import assess_evidence
from .profile import build_artifact_profiles
from .reconcile import build_reconciliation


class ArtifactReviewer(Protocol):
    """Optional LLM review contract."""

    def review(self, artifact: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
        ...


class KnowledgeEngineeringAgent:
    def __init__(self, reviewer: ArtifactReviewer | None = None) -> None:
        self.reviewer = reviewer

    def run(self, canonical: Dict[str, Any]) -> Dict[str, Any]:
        profiles = build_artifact_profiles(canonical)
        reconciliation = build_reconciliation(canonical)
        evidence = assess_evidence(canonical)

        reviews = []
        gaps = []
        artifacts = {item["id"]: item for item in canonical.get("artifacts", [])}

        for profile in profiles:
            artifact = artifacts[profile["artifact_id"]]
            if self.reviewer is not None:
                review = self.reviewer.review(artifact, profile)
                mode = "llm"
            else:
                review = self._deterministic_review(profile)
                mode = "deterministic_baseline"
            reviews.append({"artifact_id": profile["artifact_id"], "mode": mode, **review})

            if review.get("needs_deeper_analysis"):
                gaps.append(
                    {
                        "artifact_id": profile["artifact_id"],
                        "type": "DEEPER_ANALYSIS",
                        "reason": review.get("reason", "Artifact requires additional analysis."),
                    }
                )

        for group in reconciliation.get("duplicate_groups", []):
            gaps.append(
                {
                    "type": "DUPLICATE_ENTITY_CANDIDATE",
                    "entity_type": group["entity_type"],
                    "normalized_name": group["canonical_name"],
                    "entity_ids": group["entity_ids"],
                }
            )

        return {
            "schema_version": "1.0",
            "agent": {
                "name": "knowledge_engineering_agent",
                "version": "0.1.0",
                "mode": "llm_enabled" if self.reviewer else "deterministic_baseline",
            },
            "source": {
                "canonical_schema_version": canonical.get("schema_version", "1.0"),
                "artifact_count": len(canonical.get("artifacts", [])),
            },
            "artifact_profiles": profiles,
            "artifact_reviews": reviews,
            "reconciliation": reconciliation,
            "evidence_validation": evidence,
            "knowledge_gaps": gaps,
            "summary": {
                "profiles": len(profiles),
                "reviews": len(reviews),
                "knowledge_gaps": len(gaps),
                "deeper_analysis_required": sum(
                    item.get("type") == "DEEPER_ANALYSIS" for item in gaps
                ),
            },
        }

    @staticmethod
    def _deterministic_review(profile: Dict[str, Any]) -> Dict[str, Any]:
        source_type = profile.get("source_type")
        has_rules = profile.get("business_rule_count", 0) > 0
        has_relationships = profile.get("relationship_count", 0) > 0
        if not has_relationships:
            return {
                "purpose": profile.get("purpose"),
                "key_findings": ["Artifact has no canonical relationships."],
                "needs_deeper_analysis": True,
                "reason": "No relationships were extracted; deeper artifact review may recover dependencies.",
            }
        findings = [
            f"Identified {profile.get('entity_count', 0)} canonical entities.",
            f"Identified {profile.get('relationship_count', 0)} canonical relationships.",
        ]
        if has_rules:
            findings.append(f"Artifact contains {profile['business_rule_count']} business rules.")
        return {
            "purpose": profile.get("purpose"),
            "key_findings": findings,
            "source_type": source_type,
            "needs_deeper_analysis": False,
        }


def load_canonical(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_enrichment(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
