"""Knowledge Engineering Agent orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Protocol

from .canonical_metadata import build_canonical_metadata
from .evidence import assess_evidence
from .identification import identify_artifacts
from .knowledge_graph import KnowledgeGraphAgent
from .parser_executor import execute_parser
from .parser_registry import select_parsers
from .profile import build_artifact_profiles
from .reconcile import build_reconciliation
from .relationship_discovery import discover_relationships


class ArtifactReviewer(Protocol):
    def review(self, artifact: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]: ...


class KnowledgeEngineeringAgent:
    """Orchestrate the complete knowledge-engineering control flow."""

    VERSION = "1.0.0"

    def __init__(self, reviewer: ArtifactReviewer | None = None, *, execute_parsers: bool = False, project_root: Path | None = None) -> None:
        self.reviewer = reviewer
        self.execute_parsers = execute_parsers
        self.project_root = project_root or Path(__file__).resolve().parents[1]
        self.knowledge_graph_agent = KnowledgeGraphAgent()

    def run(self, canonical: Dict[str, Any]) -> Dict[str, Any]:
        artifacts = canonical.get("artifacts", [])
        identifications = identify_artifacts(artifacts)
        parser_selections = select_parsers(artifacts)
        parser_executions = []
        if self.execute_parsers:
            for selection in parser_selections:
                result = execute_parser(selection, self.project_root) if selection.get("status") == "SELECTED" else {"status": "NOT_EXECUTED", "reason": selection.get("reason")}
                parser_executions.append({"artifact_id": selection.get("artifact_id"), "parser": selection.get("parser"), **result})

        profiles = build_artifact_profiles(canonical)
        evidence = assess_evidence(canonical)
        reviews = []
        gaps = []
        artifacts_by_id = {item["id"]: item for item in artifacts}
        selection_by_id = {item["artifact_id"]: item for item in parser_selections}
        identification_by_id = {item["artifact_id"]: item for item in identifications}
        execution_by_id = {item["artifact_id"]: item for item in parser_executions}

        for item in identifications:
            if item["status"] != "IDENTIFIED":
                gaps.append({"artifact_id": item["artifact_id"], "type": "ARTIFACT_IDENTIFICATION", "reason": item["reason"]})
        for selection in parser_selections:
            if selection["status"] == "UNSUPPORTED":
                gaps.append({"artifact_id": selection["artifact_id"], "type": "PARSER_UNSUPPORTED", "reason": selection["reason"]})
        for execution in parser_executions:
            if execution["status"] not in {"EXECUTED", "NOT_EXECUTED"}:
                gaps.append({"artifact_id": execution["artifact_id"], "type": "PARSER_EXECUTION", "reason": execution.get("reason", "Selected parser execution failed.")})

        for profile in profiles:
            artifact = artifacts_by_id[profile["artifact_id"]]
            selection = selection_by_id.get(profile["artifact_id"], {})
            identification = identification_by_id.get(profile["artifact_id"], {})
            execution = execution_by_id.get(profile["artifact_id"], {})
            review_status = "LLM_REVIEWED"
            if self.reviewer is not None:
                try:
                    review = self.reviewer.review(artifact, profile)
                    mode = "llm"
                except Exception as exc:
                    review = {"purpose": profile.get("purpose", ""), "summary": "LLM review could not be completed.", "key_findings": [], "dependencies": [], "business_rules": [], "semantic_gaps": [str(exc)], "evidence_candidates": [], "confidence": 0.0, "needs_deeper_analysis": True, "reason": "LLM provider was unavailable; retry the artifact review later."}
                    mode = "llm_pending"
                    review_status = "LLM_REVIEW_PENDING"
            else:
                review = self._deterministic_review(profile)
                mode = "deterministic_baseline"
                review_status = "DETERMINISTIC_REVIEW"
            reviews.append({"artifact_id": profile["artifact_id"], "mode": mode, "status": review_status, "artifact_identification": identification, "parser_selection": selection, "parser_execution": execution, **review})
            if review.get("needs_deeper_analysis"):
                gaps.append({"artifact_id": profile["artifact_id"], "type": "DEEPER_ANALYSIS", "reason": review.get("reason", "Artifact requires additional analysis.")})
            if review_status == "LLM_REVIEW_PENDING":
                gaps.append({"artifact_id": profile["artifact_id"], "type": "LLM_REVIEW_PENDING", "reason": "LLM provider unavailable; review is pending retry."})

        reconciliation = build_reconciliation(canonical, reviews, evidence)
        for group in reconciliation.get("duplicate_groups", []):
            gaps.append({"type": "DUPLICATE_ENTITY_CANDIDATE", "entity_type": group["entity_type"], "normalized_name": group["canonical_name"], "entity_ids": group["entity_ids"]})
        for item in reconciliation.get("artifact_assessments", []):
            if item.get("status") in {"PARTIAL", "CONFLICT"}:
                gaps.append({"artifact_id": item.get("artifact_id"), "type": "RECONCILIATION", "reason": "Deterministic facts and LLM findings require validation/investigation." if item.get("status") == "PARTIAL" else "Potential conflict detected between deterministic knowledge and LLM findings."})

        canonical_metadata = build_canonical_metadata(canonical, reconciliation)
        relationship_discovery = discover_relationships(canonical_metadata)
        if relationship_discovery.get("summary", {}).get("unverified", 0):
            gaps.append({"type": "RELATIONSHIP_DISCOVERY", "reason": "Some discovered relationships require downstream validation.", "count": relationship_discovery["summary"]["unverified"]})

        # The graph is built from the canonical metadata and the validated
        # relationship-discovery payload produced immediately above. It is
        # included in the same enrichment document for end-to-end consumers.
        knowledge_graph = self.knowledge_graph_agent.run(canonical_metadata, relationship_discovery)

        parser_counts: dict[str, int] = {}
        for selection in parser_selections:
            parser_name = selection.get("parser") or "UNSUPPORTED"
            parser_counts[parser_name] = parser_counts.get(parser_name, 0) + 1
        execution_counts: dict[str, int] = {}
        for execution in parser_executions:
            status = execution.get("status", "UNKNOWN")
            execution_counts[status] = execution_counts.get(status, 0) + 1

        return {
            "schema_version": "1.5",
            "agent": {"name": "knowledge_engineering_agent", "version": self.VERSION, "mode": "llm_enabled" if self.reviewer else "deterministic_baseline", "stages": ["artifact_identification", "parser_selection", "parser_execution", "deterministic_profile", "llm_review", "evidence_validation", "reconciliation", "canonical_metadata", "relationship_discovery", "knowledge_graph", "knowledge_gap_detection"]},
            "source": {"canonical_schema_version": canonical.get("schema_version", "1.0"), "artifact_count": len(artifacts)},
            "artifact_identification": {"total": len(identifications), "identified": sum(item["status"] == "IDENTIFIED" for item in identifications), "incomplete": sum(item["status"] != "IDENTIFIED" for item in identifications), "items": identifications},
            "parser_selection": {"total": len(parser_selections), "selected": sum(item["status"] == "SELECTED" for item in parser_selections), "unsupported": sum(item["status"] == "UNSUPPORTED" for item in parser_selections), "by_parser": parser_counts, "items": parser_selections},
            "parser_execution": {"enabled": self.execute_parsers, "total": len(parser_executions), "by_status": execution_counts, "items": parser_executions},
            "artifact_profiles": profiles,
            "artifact_reviews": reviews,
            "evidence_validation": evidence,
            "reconciliation": reconciliation,
            "canonical_metadata": canonical_metadata,
            "relationship_discovery": relationship_discovery,
            "knowledge_graph": knowledge_graph,
            "knowledge_gaps": gaps,
            "summary": {"profiles": len(profiles), "reviews": len(reviews), "knowledge_gaps": len(gaps), "artifacts_identified": sum(item["status"] == "IDENTIFIED" for item in identifications), "parsers_selected": sum(item["status"] == "SELECTED" for item in parser_selections), "parsers_unsupported": sum(item["status"] == "UNSUPPORTED" for item in parser_selections), "parser_executions": len(parser_executions), "parser_executions_successful": sum(item.get("status") == "EXECUTED" for item in parser_executions), "parser_executions_failed": sum(item.get("status") in {"FAILED", "TIMEOUT", "EXECUTION_ERROR"} for item in parser_executions), "llm_reviews_completed": sum(item.get("status") == "LLM_REVIEWED" for item in reviews), "llm_reviews_pending": sum(item.get("status") == "LLM_REVIEW_PENDING" for item in reviews), "deeper_analysis_required": sum(item.get("type") == "DEEPER_ANALYSIS" for item in gaps), "reconciliation_claims_assessed": reconciliation.get("summary", {}).get("claims_assessed", 0), "reconciliation_claims_supported": reconciliation.get("summary", {}).get("claims_supported", 0), "reconciliation_claims_unverified": reconciliation.get("summary", {}).get("claims_unverified", 0), "reconciliation_potential_conflicts": reconciliation.get("summary", {}).get("potential_conflicts", 0), "canonical_metadata_artifacts": canonical_metadata.get("statistics", {}).get("artifacts", 0), "relationships_discovered": relationship_discovery.get("summary", {}).get("relationships_discovered", 0), "relationships_supported": relationship_discovery.get("summary", {}).get("supported", 0), "relationships_unverified": relationship_discovery.get("summary", {}).get("unverified", 0), "relationship_conflicts": relationship_discovery.get("summary", {}).get("conflicts", 0), "knowledge_graph_nodes": knowledge_graph.get("statistics", {}).get("nodes", 0), "knowledge_graph_edges": knowledge_graph.get("statistics", {}).get("edges", 0), "knowledge_graph_supported_edges": knowledge_graph.get("statistics", {}).get("supported_edges", 0), "knowledge_graph_unverified_edges": knowledge_graph.get("statistics", {}).get("unverified_edges", 0), "knowledge_graph_conflict_edges": knowledge_graph.get("statistics", {}).get("conflict_edges", 0)}
        }

    @staticmethod
    def _deterministic_review(profile: Dict[str, Any]) -> Dict[str, Any]:
        source_type = profile.get("source_type")
        has_rules = profile.get("business_rule_count", 0) > 0
        has_relationships = profile.get("relationship_count", 0) > 0
        if not has_relationships:
            return {"purpose": profile.get("purpose"), "key_findings": ["Artifact has no canonical relationships."], "needs_deeper_analysis": True, "reason": "No relationships were extracted; deeper artifact review may recover dependencies."}
        findings = [f"Identified {profile.get('entity_count', 0)} canonical entities.", f"Identified {profile.get('relationship_count', 0)} canonical relationships."]
        if has_rules: findings.append(f"Artifact contains {profile['business_rule_count']} business rules.")
        return {"purpose": profile.get("purpose"), "key_findings": findings, "source_type": source_type, "needs_deeper_analysis": False}


def load_canonical(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_enrichment(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
