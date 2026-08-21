"""Knowledge Engineering Agent orchestration.

The agent is the technology-neutral orchestrator for the knowledge-engineering
layer. It identifies artifacts, selects the appropriate existing parser,
combines deterministic profiles with optional LLM review, reconciles the
results, and records evidence and knowledge gaps for downstream analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Protocol

from .evidence import assess_evidence
from .identification import identify_artifacts
from .parser_registry import select_parsers
from .profile import build_artifact_profiles
from .reconcile import build_reconciliation


class ArtifactReviewer(Protocol):
    """Optional LLM review contract."""

    def review(self, artifact: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
        ...


class KnowledgeEngineeringAgent:
    """Orchestrate the complete knowledge-engineering control flow."""

    VERSION = "0.4.0"

    def __init__(self, reviewer: ArtifactReviewer | None = None) -> None:
        self.reviewer = reviewer

    def run(self, canonical: Dict[str, Any]) -> Dict[str, Any]:
        artifacts = canonical.get("artifacts", [])

        # Stage 1: identify the canonical artifacts.
        identifications = identify_artifacts(artifacts)

        # Stage 2: deterministically select the existing parser. The registry
        # does not reimplement parser logic; it provides the orchestration
        # contract over parsers already present under parsers/.
        parser_selections = select_parsers(artifacts)

        # Stage 3-5: consume canonical deterministic knowledge, then review it
        # with the configured LLM when available.
        profiles = build_artifact_profiles(canonical)
        reconciliation = build_reconciliation(canonical)
        evidence = assess_evidence(canonical)

        reviews = []
        gaps = []
        artifacts_by_id = {item["id"]: item for item in artifacts}
        selection_by_id = {item["artifact_id"]: item for item in parser_selections}
        identification_by_id = {item["artifact_id"]: item for item in identifications}

        for item in identifications:
            if item["status"] != "IDENTIFIED":
                gaps.append({
                    "artifact_id": item["artifact_id"],
                    "type": "ARTIFACT_IDENTIFICATION",
                    "reason": item["reason"],
                })

        for selection in parser_selections:
            if selection["status"] == "UNSUPPORTED":
                gaps.append({
                    "artifact_id": selection["artifact_id"],
                    "type": "PARSER_UNSUPPORTED",
                    "reason": selection["reason"],
                })

        for profile in profiles:
            artifact = artifacts_by_id[profile["artifact_id"]]
            selection = selection_by_id.get(profile["artifact_id"], {})
            identification = identification_by_id.get(profile["artifact_id"], {})
            review_status = "LLM_REVIEWED"
            if self.reviewer is not None:
                try:
                    review = self.reviewer.review(artifact, profile)
                    mode = "llm"
                except Exception as exc:
                    review = {
                        "purpose": profile.get("purpose", ""),
                        "summary": "LLM review could not be completed.",
                        "key_findings": [],
                        "dependencies": [],
                        "business_rules": [],
                        "semantic_gaps": [str(exc)],
                        "evidence_candidates": [],
                        "confidence": 0.0,
                        "needs_deeper_analysis": True,
                        "reason": "LLM provider was unavailable; retry the artifact review later.",
                    }
                    mode = "llm_pending"
                    review_status = "LLM_REVIEW_PENDING"
            else:
                review = self._deterministic_review(profile)
                mode = "deterministic_baseline"
                review_status = "DETERMINISTIC_REVIEW"

            reviews.append({
                "artifact_id": profile["artifact_id"],
                "mode": mode,
                "status": review_status,
                "artifact_identification": identification,
                "parser_selection": selection,
                **review,
            })

            if review.get("needs_deeper_analysis"):
                gaps.append({
                    "artifact_id": profile["artifact_id"],
                    "type": "DEEPER_ANALYSIS",
                    "reason": review.get("reason", "Artifact requires additional analysis."),
                })

            if review_status == "LLM_REVIEW_PENDING":
                gaps.append({
                    "artifact_id": profile["artifact_id"],
                    "type": "LLM_REVIEW_PENDING",
                    "reason": "LLM provider unavailable; review is pending retry.",
                })

        for group in reconciliation.get("duplicate_groups", []):
            gaps.append({
                "type": "DUPLICATE_ENTITY_CANDIDATE",
                "entity_type": group["entity_type"],
                "normalized_name": group["canonical_name"],
                "entity_ids": group["entity_ids"],
            })

        parser_counts: dict[str, int] = {}
        for selection in parser_selections:
            parser_name = selection.get("parser") or "UNSUPPORTED"
            parser_counts[parser_name] = parser_counts.get(parser_name, 0) + 1

        return {
            "schema_version": "1.0",
            "agent": {
                "name": "knowledge_engineering_agent",
                "version": self.VERSION,
                "mode": "llm_enabled" if self.reviewer else "deterministic_baseline",
                "stages": [
                    "artifact_identification",
                    "parser_selection",
                    "deterministic_profile",
                    "llm_review",
                    "reconciliation",
                    "evidence_validation",
                    "knowledge_gap_detection",
                ],
            },
            "source": {
                "canonical_schema_version": canonical.get("schema_version", "1.0"),
                "artifact_count": len(artifacts),
            },
            "artifact_identification": {
                "total": len(identifications),
                "identified": sum(item["status"] == "IDENTIFIED" for item in identifications),
                "incomplete": sum(item["status"] != "IDENTIFIED" for item in identifications),
                "items": identifications,
            },
            "parser_selection": {
                "total": len(parser_selections),
                "selected": sum(item["status"] == "SELECTED" for item in parser_selections),
                "unsupported": sum(item["status"] == "UNSUPPORTED" for item in parser_selections),
                "by_parser": parser_counts,
                "items": parser_selections,
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
                "artifacts_identified": sum(item["status"] == "IDENTIFIED" for item in identifications),
                "parsers_selected": sum(item["status"] == "SELECTED" for item in parser_selections),
                "parsers_unsupported": sum(item["status"] == "UNSUPPORTED" for item in parser_selections),
                "llm_reviews_completed": sum(item.get("status") == "LLM_REVIEWED" for item in reviews),
                "llm_reviews_pending": sum(item.get("status") == "LLM_REVIEW_PENDING" for item in reviews),
                "deeper_analysis_required": sum(item.get("type") == "DEEPER_ANALYSIS" for item in gaps),
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
