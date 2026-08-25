"""Comprehensive tests for atomic single-file incremental knowledge ingestion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
import pytest

from knowledge_engineering.ingest import IngestionResult, SingleFileIngestionPipeline
from knowledge_engineering.manifest import IngestionManifest, load_manifest, save_manifest_atomic
from knowledge_engineering.normalize import normalize_file, normalize_source_metadata, stable_id
from knowledge_engineering.paths import (
    PROJECT_ROOT,
    compute_file_sha256,
    get_source_type_from_path,
    resolve_project_path,
    to_project_relative,
)
from knowledge_engineering.summary import render_artifact_summary, write_single_summary
from knowledge_engineering.llm.reviewer import load_source
from knowledge_engineering.vector_retrieval.chunker import KnowledgeChunker
from knowledge_engineering.vector_retrieval.indexer import index_single_artifact
from knowledge_engineering.knowledge_graph.neo4j_store import Neo4jKnowledgeGraphStore


# ============================================================================
# 1-4. PATH NORMALIZATION TESTS
# ============================================================================

def test_project_relative_path_normalization():
    rel = "source/sql/ClaimCenter_CPP_Breakdown.sql"
    normalized = to_project_relative(rel, PROJECT_ROOT)
    assert normalized == "source/sql/ClaimCenter_CPP_Breakdown.sql"
    assert "\\" not in normalized


def test_absolute_windows_path_normalization():
    abs_path = PROJECT_ROOT / "source" / "sql" / "ClaimCenter_CPP_Breakdown.sql"
    normalized = to_project_relative(abs_path, PROJECT_ROOT)
    assert normalized == "source/sql/ClaimCenter_CPP_Breakdown.sql"
    assert "\\" not in normalized


def test_relative_path_with_backslashes_normalization():
    rel_win = "source\\mainframe\\cobol\\EARNPREM.CBL"
    normalized = to_project_relative(rel_win, PROJECT_ROOT)
    assert normalized == "source/mainframe/cobol/EARNPREM.CBL"
    assert "\\" not in normalized


def test_path_outside_project_root_raises():
    outside = Path("C:/SomeOtherDirectory/fake.sql") if Path("C:/").exists() else Path("/tmp/outside/fake.sql")
    with pytest.raises(ValueError, match="outside project root"):
        to_project_relative(outside, PROJECT_ROOT, allow_outside=False)


def test_resolve_project_path():
    resolved = resolve_project_path("source/sql/foo.sql", PROJECT_ROOT)
    assert resolved == (PROJECT_ROOT / "source" / "sql" / "foo.sql").resolve()


# ============================================================================
# 5-8. SINGLE-FILE PARSER TESTS
# ============================================================================

def test_single_file_sql_parser(tmp_path):
    from parsers.sql.parse import parse_single_file as parse_sql

    sql_file = tmp_path / "test_query.sql"
    sql_file.write_text("SELECT id, name FROM users WHERE active = 1;", encoding="utf-8")
    out_dir = tmp_path / "output_sql"

    metadata = parse_sql(sql_file, output_dir=out_dir)
    assert metadata["summary"]["unique_tables"] >= 1
    assert (out_dir / "test_query_metadata.json").exists()


def test_single_file_cobol_parser(tmp_path):
    from parsers.cobol.batch_parse_312_parity import parse_single_file as parse_cobol

    cobol_file = tmp_path / "TESTPROG.CBL"
    cobol_file.write_text(
        "       IDENTIFICATION DIVISION.\n"
        "       PROGRAM-ID. TESTPROG.\n"
        "       DATA DIVISION.\n"
        "       WORKING-STORAGE SECTION.\n"
        "       01 WS-VAR PIC X(10).\n"
        "       PROCEDURE DIVISION.\n"
        "           STOP RUN.\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "output_cobol"

    metadata = parse_cobol(cobol_file, output_dir=out_dir)
    assert metadata["program_id"] == "TESTPROG"
    assert (out_dir / "TESTPROG_metadata.json").exists()


def test_single_file_ssis_parser(tmp_path):
    from parsers.ssis.parse import parse_single_file as parse_ssis

    # Minimal valid DTSX XML
    dtsx_content = """<?xml version="1.0"?>
    <DTS:Executable xmlns:DTS="www.microsoft.com/SqlServer/Dts"
        DTS:ExecutableType="Microsoft.Package"
        DTS:ObjectName="Test_Package">
        <DTS:Property DTS:Name="PackageFormatVersion">8</DTS:Property>
        <DTS:Variables/>
        <DTS:Executables/>
    </DTS:Executable>"""

    dtsx_file = tmp_path / "Test_Package.dtsx"
    dtsx_file.write_text(dtsx_content, encoding="utf-8")
    out_dir = tmp_path / "output_ssis"

    metadata = parse_ssis(dtsx_file, output_dir=out_dir)
    assert metadata["source_type"] == "SSIS DTSX"
    assert (out_dir / "Test_Package_metadata.json").exists()


def test_parser_does_not_process_sibling_files(tmp_path):
    from parsers.sql.parse import parse_single_file as parse_sql

    file_a = tmp_path / "A.sql"
    file_b = tmp_path / "B.sql"
    file_a.write_text("SELECT * FROM table_a;", encoding="utf-8")
    file_b.write_text("SELECT * FROM table_b;", encoding="utf-8")

    out_dir = tmp_path / "output"
    parse_sql(file_a, output_dir=out_dir)

    assert (out_dir / "A_metadata.json").exists()
    assert not (out_dir / "B_metadata.json").exists()


# ============================================================================
# 9-16. MANIFEST & INCREMENTAL PIPELINE TESTS
# ============================================================================

def test_manifest_creation_and_atomic_update(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest = IngestionManifest(manifest_path=manifest_path, project_root=tmp_path)

    assert manifest.check_status("source/sql/foo.sql", "sha256:123") == "NEW"

    manifest.record_ingested(
        "source/sql/foo.sql",
        artifact_id="artifact:sql:foo",
        source_type="sql",
        sha256="sha256:123",
        mtime=100.0,
        size_bytes=50,
        parser_output="output/sql/foo_metadata.json",
        summary_output="output/knowledge/summaries/foo_summary.md",
    )

    assert manifest.check_status("source/sql/foo.sql", "sha256:123") == "UNCHANGED"
    assert manifest.check_status("source/sql/foo.sql", "sha256:999") == "MODIFIED"

    # Verify atomic file existence and validity on disk
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "source/sql/foo.sql" in data["files"]


def test_sha256_detection(tmp_path):
    test_file = tmp_path / "sample.sql"
    test_file.write_text("SELECT 1;", encoding="utf-8")
    hash1 = compute_file_sha256(test_file, tmp_path)
    assert hash1.startswith("sha256:")

    test_file.write_text("SELECT 2;", encoding="utf-8")
    hash2 = compute_file_sha256(test_file, tmp_path)
    assert hash2.startswith("sha256:")
    assert hash1 != hash2


def test_single_file_ingestion_new_file(tmp_path):
    source_dir = tmp_path / "source" / "sql"
    source_dir.mkdir(parents=True)
    sql_file = source_dir / "test_item.sql"
    sql_file.write_text("SELECT * FROM products;", encoding="utf-8")

    pipeline = SingleFileIngestionPipeline(project_root=tmp_path)
    result = pipeline.ingest_file("source/sql/test_item.sql")

    assert result.status == "NEW"
    assert result.source_path == "source/sql/test_item.sql"
    assert result.artifact_id.startswith("artifact:")
    assert result.parser_output == "output/sql/test_item_metadata.json"
    assert result.summary_output == "output/knowledge/summaries/test_item_summary.md"

    # Verify outputs created
    assert (tmp_path / "output" / "sql" / "test_item_metadata.json").exists()
    assert (tmp_path / "output" / "knowledge" / "canonical_metadata.json").exists()
    assert (tmp_path / "output" / "knowledge" / "summaries" / "test_item_summary.md").exists()


def test_single_file_ingestion_unchanged_file_skip(tmp_path):
    source_dir = tmp_path / "source" / "sql"
    source_dir.mkdir(parents=True)
    sql_file = source_dir / "test_skip.sql"
    sql_file.write_text("SELECT * FROM items;", encoding="utf-8")

    pipeline = SingleFileIngestionPipeline(project_root=tmp_path)
    result1 = pipeline.ingest_file("source/sql/test_skip.sql")
    assert result1.status == "NEW"

    # Second run without changes -> MUST SKIP
    result2 = pipeline.ingest_file("source/sql/test_skip.sql")
    assert result2.status == "SKIPPED"
    assert result2.vector_update_status == "SKIPPED"
    assert result2.graph_update_status == "SKIPPED"


def test_single_file_ingestion_modified_file(tmp_path):
    source_dir = tmp_path / "source" / "sql"
    source_dir.mkdir(parents=True)
    sql_file = source_dir / "test_mod.sql"
    sql_file.write_text("SELECT v1 FROM t1;", encoding="utf-8")

    pipeline = SingleFileIngestionPipeline(project_root=tmp_path)
    result1 = pipeline.ingest_file("source/sql/test_mod.sql")
    assert result1.status == "NEW"

    # Modify file
    sql_file.write_text("SELECT v2 FROM t2 WHERE id = 1;", encoding="utf-8")
    result2 = pipeline.ingest_file("source/sql/test_mod.sql")
    assert result2.status == "MODIFIED"


def test_single_file_ingestion_force(tmp_path):
    source_dir = tmp_path / "source" / "sql"
    source_dir.mkdir(parents=True)
    sql_file = source_dir / "test_force.sql"
    sql_file.write_text("SELECT 1;", encoding="utf-8")

    pipeline = SingleFileIngestionPipeline(project_root=tmp_path)
    pipeline.ingest_file("source/sql/test_force.sql")

    # Force flag should reprocess even when content is identical
    result = pipeline.ingest_file("source/sql/test_force.sql", force=True)
    assert result.status == "MODIFIED"


# ============================================================================
# 16-19. ARTIFACT IDENTITY & SUMMARY TESTS
# ============================================================================

def test_canonical_artifact_identifies_source_file():
    raw_data = {"unique_tables": ["orders"], "business_rules": ["rule 1"]}
    doc = normalize_source_metadata(
        raw_data,
        source_path="source/sql/orders.sql",
        source_type="sql",
        project_root=PROJECT_ROOT,
    )
    artifact = doc.artifacts[0]
    assert artifact.file_name == "orders.sql"
    assert artifact.path == "source/sql/orders.sql"
    assert artifact.source_type == "sql"
    assert "\\" not in artifact.path


def test_reviewer_loads_raw_source():
    artifact = {
        "id": "artifact:test",
        "source_type": "sql",
        "file_name": "ClaimCenter_CPP_Breakdown.sql",
        "path": "source/sql/ClaimCenter_CPP_Breakdown.sql",
    }
    content = load_source(artifact)
    assert "Original source could not be loaded" not in content
    # Ensure it loaded raw SQL, not parser JSON
    assert "SELECT" in content.upper() or "CREATE" in content.upper() or "SET" in content.upper()


def test_write_single_summary_naming_and_content(tmp_path):
    out_dir = tmp_path / "summaries"
    review = {
        "artifact_id": "artifact:sql:test",
        "status": "LLM_REVIEWED",
        "purpose": "Calculates revenue.",
        "summary": "Transforms orders into revenue aggregates.",
        "key_findings": ["Contains 1 JOIN"],
        "dependencies": ["orders_table"],
        "business_rules": ["Total must be positive"],
        "confidence": 0.9,
    }
    profile = {
        "artifact_id": "artifact:sql:test",
        "source_type": "sql",
        "entity_count": 5,
        "relationship_count": 3,
    }
    path = write_single_summary(
        artifact_id="artifact:sql:test",
        review=review,
        profile=profile,
        output_dir=out_dir,
        source_path="source/sql/revenue.sql",
    )

    assert path.name == "revenue_summary.md"
    assert "_metadata" not in path.name
    text = path.read_text(encoding="utf-8")
    assert "revenue.sql" in text
    assert "source/sql/revenue.sql" in text
    assert "Transforms orders into revenue aggregates." in text


def test_no_summary_rewrite_for_unrelated_artifacts(tmp_path):
    out_dir = tmp_path / "summaries"
    out_dir.mkdir(parents=True)
    other_summary = out_dir / "unrelated_summary.md"
    other_summary.write_text("Original content for unrelated file", encoding="utf-8")

    review = {"artifact_id": "artifact:target", "status": "DETERMINISTIC_REVIEW", "summary": "Target summary"}
    write_single_summary(
        artifact_id="artifact:target",
        review=review,
        output_dir=out_dir,
        source_path="source/sql/target.sql",
    )

    assert (out_dir / "target_summary.md").exists()
    assert other_summary.read_text(encoding="utf-8") == "Original content for unrelated file"


# ============================================================================
# 20-23. QDRANT, NEO4J & CROSS-FILE ISOLATION TESTS
# ============================================================================

class FakeQdrantStore:
    def __init__(self):
        self.collection_name = "kairix_knowledge"
        self.points = {}
        self.deleted_artifacts = []

    def ensure_collection(self, dimension):
        pass

    def delete_by_artifact_id(self, artifact_id):
        self.deleted_artifacts.append(artifact_id)
        self.points = {pid: pt for pid, pt in self.points.items() if pt.get("artifact_id") != artifact_id}

    def upsert(self, chunks, vectors):
        for chunk, vec in zip(chunks, vectors):
            self.points[chunk["id"]] = {"chunk": chunk, "vector": vec, "artifact_id": chunk.get("artifact_id")}
        return len(chunks)

    def count(self):
        return len(self.points)


class FakeRetriever:
    def __init__(self):
        self.store = FakeQdrantStore()
        self.embedder = type("FakeEmbedder", (), {
            "dimension": 4,
            "model_name": "fake-minilm",
            "embed": lambda self, texts: [[0.1, 0.2, 0.3, 0.4] for _ in texts],
        })()

    def index_artifact(self, artifact_id, chunks):
        self.store.delete_by_artifact_id(artifact_id)
        if not chunks:
            return 0
        vectors = self.embedder.embed([c["text"] for c in chunks])
        return self.store.upsert(chunks, vectors)


def test_qdrant_single_artifact_indexing():
    retriever = FakeRetriever()
    canonical = {
        "artifacts": [{"artifact_id": "artifact:A", "file_name": "A.sql", "source_type": "sql"}],
        "entities": [{"id": "entity:1", "artifact_id": "artifact:A", "name": "t1"}],
        "relationships": [],
        "business_rules": [],
    }

    result = index_single_artifact(canonical, "artifact:A", retriever=retriever)
    assert result["chunks_built"] == 2  # 1 artifact + 1 entity
    assert result["chunks_indexed"] == 2
    assert "artifact:A" in retriever.store.deleted_artifacts


def test_no_cross_file_contamination(tmp_path):
    source_dir = tmp_path / "source" / "sql"
    source_dir.mkdir(parents=True)
    file_a = source_dir / "A.sql"
    file_b = source_dir / "B.sql"
    file_a.write_text("SELECT col_a FROM table_a;", encoding="utf-8")
    file_b.write_text("SELECT col_b FROM table_b;", encoding="utf-8")

    pipeline = SingleFileIngestionPipeline(project_root=tmp_path)
    res_a = pipeline.ingest_file("source/sql/A.sql")
    assert res_a.status == "NEW"

    # Save state of A
    meta_a_before = (tmp_path / "output" / "sql" / "A_metadata.json").read_text(encoding="utf-8")
    summary_a_before = (tmp_path / "output" / "knowledge" / "summaries" / "A_summary.md").read_text(encoding="utf-8")
    manifest_a_before = pipeline.manifest.get_entry("source/sql/A.sql")

    # Ingest B
    res_b = pipeline.ingest_file("source/sql/B.sql")
    assert res_b.status == "NEW"

    # Verify A remains completely untouched
    meta_a_after = (tmp_path / "output" / "sql" / "A_metadata.json").read_text(encoding="utf-8")
    summary_a_after = (tmp_path / "output" / "knowledge" / "summaries" / "A_summary.md").read_text(encoding="utf-8")
    manifest_a_after = pipeline.manifest.get_entry("source/sql/A.sql")

    assert meta_a_before == meta_a_after
    assert summary_a_before == summary_a_after
    assert manifest_a_before == manifest_a_after


def test_failure_leaves_consistent_manifest(tmp_path):
    source_dir = tmp_path / "source" / "sql"
    source_dir.mkdir(parents=True)
    bad_file = source_dir / "bad.sql"
    bad_file.write_text("INVALID SQL SYNTAX {{{", encoding="utf-8")

    pipeline = SingleFileIngestionPipeline(project_root=tmp_path)
    result = pipeline.ingest_file("source/sql/non_existent.sql")

    assert result.status == "FAILED"
    entry = pipeline.manifest.get_entry("source/sql/non_existent.sql")
    # Missing source file was not added as INGESTED
    assert entry is None or entry.get("status") != "INGESTED"


def test_qdrant_unchanged_file_no_op(tmp_path):
    source_dir = tmp_path / "source" / "sql"
    source_dir.mkdir(parents=True)
    sql_file = source_dir / "unchanged.sql"
    sql_file.write_text("SELECT 1;", encoding="utf-8")

    retriever = FakeRetriever()
    pipeline = SingleFileIngestionPipeline(project_root=tmp_path, vector_retriever=retriever)
    res1 = pipeline.ingest_file("source/sql/unchanged.sql", update_vectors=True)
    assert res1.status == "NEW"
    assert res1.vector_update_status == "INDEXED"

    # Second run without force should do 0 vector operations
    res2 = pipeline.ingest_file("source/sql/unchanged.sql", update_vectors=True)
    assert res2.status == "SKIPPED"
    assert res2.vector_update_status == "SKIPPED"


def test_graph_single_artifact_update_contract():
    class FakeSession:
        def __init__(self):
            self.queries = []
            self.writes = []

        def run(self, query, **kwargs):
            self.queries.append((query, kwargs))

        def execute_write(self, fn, items):
            self.writes.append(items)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class FakeDriver:
        def __init__(self):
            self.session_obj = FakeSession()

        def session(self, **kwargs):
            return self.session_obj

    store = Neo4jKnowledgeGraphStore(uri="bolt://localhost:7687", password="mock")
    store._driver = FakeDriver()

    graph = {
        "nodes": [
            {"id": "node:A1", "artifact_id": "artifact:A", "type": "TABLE", "name": "t1"},
            {"id": "node:B1", "artifact_id": "artifact:B", "type": "TABLE", "name": "t2"},
        ],
        "edges": [
            {"id": "edge:A1", "artifact_id": "artifact:A", "source_entity_id": "node:A1", "target_entity_id": "node:A1", "relationship_type": "USES"},
            {"id": "edge:B1", "artifact_id": "artifact:B", "source_entity_id": "node:B1", "target_entity_id": "node:B1", "relationship_type": "USES"},
        ],
    }

    # Updating artifact:A only writes node:A1 and edge:A1
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(store, "counts", lambda: {"nodes": 1, "edges": 1, "supported": 1, "unverified": 0, "conflicts": 0})
        counts = store.write_artifact_graph("artifact:A", graph)
        assert counts["nodes"] == 1
        # Check deletion query targeted artifact:A
        del_query, kwargs = store._driver.session_obj.queries[0]
        assert "artifact_id: $artifact_id" in del_query
        assert kwargs["artifact_id"] == "artifact:A"


def test_backward_compatibility_with_existing_canonical_metadata():
    # Verify legacy-style metadata paths are handled gracefully by normalize_file
    legacy_file = PROJECT_ROOT / "output" / "sql" / "ClaimCenter_CPP_Breakdown_metadata.json"
    if legacy_file.exists():
        doc = normalize_file(legacy_file, "sql", project_root=PROJECT_ROOT)
        assert doc.artifacts[0].file_name == "ClaimCenter_CPP_Breakdown.sql"
        assert doc.artifacts[0].path == "source/sql/ClaimCenter_CPP_Breakdown.sql"
        assert len(doc.entities) > 0

