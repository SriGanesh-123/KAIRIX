"""Atomic single-file incremental knowledge ingestion orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Optional

from .agent import ArtifactReviewer, KnowledgeEngineeringAgent
from .enrichment_store import load_existing, merge_enrichment, write_merged
from .knowledge_graph.agent import KnowledgeGraphAgent
from .knowledge_graph.neo4j_store import Neo4jKnowledgeGraphStore
from .manifest import IngestionManifest, get_default_manifest_path
from .normalize import normalize_source_metadata, stable_id
from .parser_registry import select_parser
from .paths import (
    PROJECT_ROOT,
    compute_file_sha256,
    get_project_root,
    get_source_type_from_path,
    resolve_project_path,
    to_project_relative,
)
from .profile import build_artifact_profiles
from .reconcile import build_reconciliation
from .schema import KnowledgeDocument
from .summary import render_artifact_summary, write_single_summary
from .validate import validate_document
from .vector_retrieval.indexer import index_single_artifact
from .vector_retrieval.retriever import VectorRetriever


@dataclass
class IngestionResult:
    source_path: str
    artifact_id: str
    source_hash: str
    status: str  # "NEW", "MODIFIED", "SKIPPED", "FAILED"
    parser_output: Optional[str] = None
    summary_output: Optional[str] = None
    vector_update_status: Optional[str] = None
    graph_update_status: Optional[str] = None
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


def _load_existing_canonical(canonical_path: Path) -> Dict[str, Any]:
    if not canonical_path.exists():
        return {
            "schema_version": "1.0",
            "artifacts": [],
            "entities": [],
            "relationships": [],
            "evidence": [],
            "business_rules": [],
        }
    try:
        return json.loads(canonical_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "schema_version": "1.0",
            "artifacts": [],
            "entities": [],
            "relationships": [],
            "evidence": [],
            "business_rules": [],
        }


def _merge_canonical_artifact(
    existing: Dict[str, Any],
    new_doc: KnowledgeDocument,
    artifact_id: str,
    source_path: str,
) -> Dict[str, Any]:
    """Replace only records belonging to artifact_id/source_path and append new ones."""
    # Filter out existing items for this artifact
    filtered_artifacts = [
        a for a in existing.get("artifacts", [])
        if a.get("id") != artifact_id and a.get("path") != source_path
    ]
    filtered_entities = [
        e for e in existing.get("entities", [])
        if e.get("artifact_id") != artifact_id
    ]
    filtered_relationships = [
        r for r in existing.get("relationships", [])
        if r.get("artifact_id") != artifact_id and r.get("source_artifact_id") != artifact_id
    ]
    filtered_evidence = [
        ev for ev in existing.get("evidence", [])
        if ev.get("artifact_id") != artifact_id
    ]
    filtered_rules = [
        br for br in existing.get("business_rules", [])
        if br.get("artifact_id") != artifact_id
    ]

    new_dict = new_doc.model_dump()
    merged = {
        "schema_version": existing.get("schema_version", "1.0"),
        "artifacts": filtered_artifacts + new_dict.get("artifacts", []),
        "entities": filtered_entities + new_dict.get("entities", []),
        "relationships": filtered_relationships + new_dict.get("relationships", []),
        "evidence": filtered_evidence + new_dict.get("evidence", []),
        "business_rules": filtered_rules + new_dict.get("business_rules", []),
    }
    return merged


class SingleFileIngestionPipeline:
    """Atomic ingestion pipeline for a single source file."""

    def __init__(
        self,
        project_root: Path | None = None,
        *,
        reviewer: ArtifactReviewer | None = None,
        vector_retriever: VectorRetriever | None = None,
        neo4j_store: Neo4jKnowledgeGraphStore | None = None,
        manifest: IngestionManifest | None = None,
    ) -> None:
        self.project_root = project_root or get_project_root()
        self.reviewer = reviewer
        self.vector_retriever = vector_retriever
        self.neo4j_store = neo4j_store
        self.manifest = manifest or IngestionManifest(project_root=self.project_root)
        self.canonical_path = self.project_root / "output" / "knowledge" / "canonical_metadata.json"
        self.enrichment_path = self.project_root / "output" / "knowledge" / "knowledge_enrichment.json"
        self.summaries_dir = self.project_root / "output" / "knowledge" / "summaries"

    def ingest_file(
        self,
        source_path: str | Path,
        *,
        force: bool = False,
        update_vectors: bool = False,
        update_graph: bool = False,
    ) -> IngestionResult:
        """Process one single source file end-to-end adhering strictly to atomic invariants."""
        # 1. Normalize and resolve source path
        try:
            source_rel = to_project_relative(source_path, self.project_root)
            abs_source = resolve_project_path(source_rel, self.project_root)
        except Exception as exc:
            return IngestionResult(
                source_path=str(source_path),
                artifact_id="",
                source_hash="",
                status="FAILED",
                error=f"Path normalization error: {exc}",
            )

        if not abs_source.exists() or not abs_source.is_file():
            return IngestionResult(
                source_path=source_rel,
                artifact_id="",
                source_hash="",
                status="FAILED",
                error=f"Source file does not exist: {abs_source}",
            )

        # 2. Compute SHA-256 hash and stat
        try:
            source_hash = compute_file_sha256(abs_source, self.project_root)
            stat = abs_source.stat()
            mtime = stat.st_mtime
            size_bytes = stat.st_size
        except Exception as exc:
            return IngestionResult(
                source_path=source_rel,
                artifact_id="",
                source_hash="",
                status="FAILED",
                error=f"Hashing error: {exc}",
            )

        # 3. Check Manifest status
        status_in_manifest = self.manifest.check_status(source_rel, source_hash)
        if status_in_manifest == "UNCHANGED" and not force:
            entry = self.manifest.get_entry(source_rel) or {}
            return IngestionResult(
                source_path=source_rel,
                artifact_id=entry.get("artifact_id", ""),
                source_hash=source_hash,
                status="SKIPPED",
                parser_output=entry.get("parser_output"),
                summary_output=entry.get("summary_output"),
                vector_update_status="SKIPPED",
                graph_update_status="SKIPPED",
                details={"reason": "Content hash unchanged; skipping all downstream stages."},
            )

        operation_status = "NEW" if status_in_manifest == "NEW" else "MODIFIED"

        # 4. Select parser
        try:
            source_type = get_source_type_from_path(source_rel)
            spec = select_parser(source_type=source_type, file_name=source_rel)
        except Exception as exc:
            self.manifest.record_failed(source_rel, error=str(exc), sha256=source_hash)
            return IngestionResult(
                source_path=source_rel,
                artifact_id="",
                source_hash=source_hash,
                status="FAILED",
                error=f"Parser selection failed: {exc}",
            )

        # 5. Parse single file
        output_dir = self.project_root / spec.output_dir
        try:
            if spec.source_type == "sql":
                from parsers.sql.parse import parse_single_file as parse_sql_single
                raw_metadata = parse_sql_single(abs_source, output_dir=output_dir)
            elif spec.source_type == "cobol":
                from parsers.cobol.batch_parse_312_parity import parse_single_file as parse_cobol_single
                raw_metadata = parse_cobol_single(abs_source, output_dir=output_dir)
            elif spec.source_type == "ssis":
                from parsers.ssis.parse import parse_single_file as parse_ssis_single
                raw_metadata = parse_ssis_single(abs_source, output_dir=output_dir)
            else:
                raise ValueError(f"No single-file parser runner for source_type: {spec.source_type}")
        except Exception as exc:
            self.manifest.record_failed(source_rel, error=f"Parser execution failed: {exc}", sha256=source_hash)
            return IngestionResult(
                source_path=source_rel,
                artifact_id="",
                source_hash=source_hash,
                status="FAILED",
                error=f"Parser execution failed: {exc}",
            )

        parser_output_file = output_dir / f"{abs_source.stem}_metadata.json"
        parser_output_rel = to_project_relative(parser_output_file, self.project_root, allow_outside=True)

        # 6. Normalize metadata
        try:
            doc = normalize_source_metadata(raw_metadata, source_rel, source_type, self.project_root)
            artifact = doc.artifacts[0]
            artifact_id = artifact.id
        except Exception as exc:
            self.manifest.record_failed(source_rel, error=f"Metadata normalization failed: {exc}", sha256=source_hash)
            return IngestionResult(
                source_path=source_rel,
                artifact_id="",
                source_hash=source_hash,
                status="FAILED",
                error=f"Metadata normalization failed: {exc}",
            )

        # 7. Merge into canonical_metadata.json
        try:
            existing_canonical = _load_existing_canonical(self.canonical_path)
            merged_canonical = _merge_canonical_artifact(existing_canonical, doc, artifact_id, source_rel)
            self.canonical_path.parent.mkdir(parents=True, exist_ok=True)
            self.canonical_path.write_text(
                json.dumps(merged_canonical, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            self.manifest.record_failed(source_rel, error=f"Canonical update failed: {exc}", sha256=source_hash)
            return IngestionResult(
                source_path=source_rel,
                artifact_id=artifact_id,
                source_hash=source_hash,
                status="FAILED",
                error=f"Canonical update failed: {exc}",
            )

        # 8. Per-Artifact Knowledge Engineering Profile & Review
        try:
            agent = KnowledgeEngineeringAgent(reviewer=self.reviewer, project_root=self.project_root)
            enrichment = agent.run(merged_canonical)
            write_merged(self.enrichment_path, enrichment)
        except Exception as exc:
            self.manifest.record_failed(source_rel, error=f"Enrichment agent failed: {exc}", sha256=source_hash)
            return IngestionResult(
                source_path=source_rel,
                artifact_id=artifact_id,
                source_hash=source_hash,
                status="FAILED",
                error=f"Enrichment agent failed: {exc}",
            )

        # 9. Summary Document Generation for THIS artifact only
        try:
            review = next(
                (r for r in enrichment.get("artifact_reviews", []) if r.get("artifact_id") == artifact_id),
                {"artifact_id": artifact_id, "status": "DETERMINISTIC_REVIEW", "summary": "Deterministic profile."},
            )
            profile = next(
                (p for p in enrichment.get("artifact_profiles", []) if p.get("artifact_id") == artifact_id),
                None,
            )
            summary_path = write_single_summary(
                artifact_id=artifact_id,
                review=review,
                profile=profile,
                output_dir=self.summaries_dir,
                source_path=source_rel,
            )
            summary_output_rel = to_project_relative(summary_path, self.project_root, allow_outside=True)
        except Exception as exc:
            self.manifest.record_failed(source_rel, error=f"Summary generation failed: {exc}", sha256=source_hash)
            return IngestionResult(
                source_path=source_rel,
                artifact_id=artifact_id,
                source_hash=source_hash,
                status="FAILED",
                error=f"Summary generation failed: {exc}",
            )

        # 10. Vector Indexing (Qdrant)
        vector_update_status = "NOT_CONFIGURED"
        if update_vectors and self.vector_retriever is not None:
            try:
                canonical_meta = enrichment.get("canonical_metadata", merged_canonical)
                index_single_artifact(canonical_meta, artifact_id, retriever=self.vector_retriever)
                vector_update_status = "INDEXED"
            except Exception as exc:
                vector_update_status = f"FAILED: {exc}"

        # 11. Graph Update (Neo4j)
        graph_update_status = "NOT_CONFIGURED"
        if update_graph and self.neo4j_store is not None and self.neo4j_store.configured:
            try:
                graph_data = enrichment.get("knowledge_graph", {})
                self.neo4j_store.write_artifact_graph(artifact_id, graph_data)
                graph_update_status = "UPDATED"
            except Exception as exc:
                graph_update_status = f"FAILED: {exc}"

        # 12. Commit Manifest Atomically
        try:
            self.manifest.record_ingested(
                source_rel,
                artifact_id=artifact_id,
                source_type=source_type,
                sha256=source_hash,
                mtime=mtime,
                size_bytes=size_bytes,
                parser_output=parser_output_rel,
                summary_output=summary_output_rel,
            )
        except Exception as exc:
            return IngestionResult(
                source_path=source_rel,
                artifact_id=artifact_id,
                source_hash=source_hash,
                status="FAILED",
                error=f"Manifest commit failed: {exc}",
            )

        return IngestionResult(
            source_path=source_rel,
            artifact_id=artifact_id,
            source_hash=source_hash,
            status=operation_status,
            parser_output=parser_output_rel,
            summary_output=summary_output_rel,
            vector_update_status=vector_update_status,
            graph_update_status=graph_update_status,
            details={
                "source_type": source_type,
                "entities_extracted": len(doc.entities),
                "relationships_extracted": len(doc.relationships),
                "business_rules_extracted": len(doc.business_rules),
            },
        )
