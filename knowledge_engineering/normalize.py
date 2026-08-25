"""Normalize parser-specific metadata into the canonical knowledge model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from .paths import get_project_root, to_project_relative
from .schema import Artifact, BusinessRule, Entity, Evidence, KnowledgeDocument, Relationship


class MetadataLoadError(ValueError):
    """Raised when a parser metadata file cannot be consumed as JSON."""


def stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(part or "") for part in parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def load_json(path: Path) -> Dict[str, Any]:
    """Load one metadata JSON file and fail with a useful file-level error."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MetadataLoadError(f"Cannot read metadata file: {path}: {exc}") from exc

    if not text.strip():
        raise MetadataLoadError(f"Metadata file is empty: {path}")

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MetadataLoadError(
            f"Metadata file contains invalid JSON: {path}: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise MetadataLoadError(
            f"Metadata root must be a JSON object: {path}"
        )

    return data


def infer_source_relative_path(
    metadata_path: Path,
    source_type: str,
    project_root: Path | None = None,
) -> str:
    """Infer the original source relative path from parser output metadata file."""
    stem = metadata_path.stem
    if stem.endswith("_metadata"):
        stem = stem[:-9]

    root = project_root or get_project_root()

    if source_type == "sql":
        candidate = root / "source" / "sql" / f"{stem}.sql"
        if candidate.exists():
            return to_project_relative(candidate, root)
        return f"source/sql/{stem}.sql"

    if source_type == "cobol":
        # Check .CBL, .cbl, .cob, .cpy
        for ext in (".CBL", ".cbl", ".cob", ".cpy"):
            candidate = root / "source" / "mainframe" / "cobol" / f"{stem}{ext}"
            if candidate.exists():
                return to_project_relative(candidate, root)
        return f"source/mainframe/cobol/{stem}.CBL"

    if source_type == "ssis":
        candidate = root / "source" / "ssis" / "packages" / f"{stem}.dtsx"
        if candidate.exists():
            return to_project_relative(candidate, root)
        return f"source/ssis/packages/{stem}.dtsx"

    return to_project_relative(metadata_path, root, allow_outside=True)


def normalize_source_metadata(
    data: Dict[str, Any],
    source_path: str | Path,
    source_type: str,
    project_root: Path | None = None,
) -> KnowledgeDocument:
    """Normalize extracted parser metadata dictionary against the authoritative source file path."""
    root = project_root or get_project_root()
    source_rel = to_project_relative(source_path, root, allow_outside=True)
    source_filename = Path(source_rel).name
    source_stem = Path(source_rel).stem

    artifact_id = stable_id("artifact", source_type, source_rel)

    artifact = Artifact(
        id=artifact_id,
        source_type=source_type,
        file_name=source_filename,
        path=source_rel,
        metadata={"metadata_version": data.get("metadata_version")},
    )

    entities: List[Entity] = []
    relationships: List[Relationship] = []
    evidence: List[Evidence] = []
    business_rules: List[BusinessRule] = []

    def add_entity(entity_type: str, name: str, properties: Dict[str, Any] | None = None) -> str:
        entity_id = stable_id("entity", source_type, source_rel, entity_type, name)
        entities.append(
            Entity(
                id=entity_id,
                entity_type=entity_type,
                name=name,
                artifact_id=artifact_id,
                properties=properties or {},
            )
        )
        return entity_id

    def add_relationship(
        source_id: str,
        relationship_type: str,
        target_id: str,
        properties: Dict[str, Any] | None = None,
    ) -> None:
        relationships.append(
            Relationship(
                id=stable_id("rel", source_id, relationship_type, target_id),
                source_entity_id=source_id,
                relationship_type=relationship_type,
                target_entity_id=target_id,
                artifact_id=artifact_id,
                properties=properties or {},
            )
        )

    artifact_entity_id = add_entity(
        "ARTIFACT",
        source_stem,
        {"source_type": source_type, "file_name": source_filename, "source_path": source_rel},
    )

    if source_type == "sql":
        for table in data.get("unique_tables", []):
            table_id = add_entity("TABLE", str(table))
            add_relationship(artifact_entity_id, "USES", table_id)

        for item in data.get("business_rules", []):
            rule_text = item.get("rule") if isinstance(item, dict) else str(item)
            rule_id = stable_id("rule", artifact_id, rule_text)
            evidence_id = stable_id("evidence", artifact_id, "business_rule", rule_text)
            evidence.append(
                Evidence(
                    id=evidence_id,
                    artifact_id=artifact_id,
                    source_type=source_type,
                    text=rule_text,
                )
            )
            business_rules.append(
                BusinessRule(
                    id=rule_id,
                    name="SQL business rule",
                    description=rule_text,
                    artifact_id=artifact_id,
                    evidence_ids=[evidence_id],
                    confidence=1.0,
                )
            )

    elif source_type == "cobol":
        program_name = data.get("program_id") or source_stem
        program_id = add_entity("PROGRAM", str(program_name))
        add_relationship(artifact_entity_id, "DEFINES", program_id)

        for record in data.get("records", []):
            name = record.get("record_name") if isinstance(record, dict) else str(record)
            record_id = add_entity("RECORD", str(name))
            add_relationship(program_id, "CONTAINS", record_id)

        for file_item in data.get("files", []):
            if isinstance(file_item, dict):
                name = file_item.get("name") or file_item.get("file")
            else:
                name = str(file_item)
            if name:
                file_id = add_entity(
                    "FILE",
                    str(name),
                    file_item if isinstance(file_item, dict) else {},
                )
                add_relationship(program_id, "USES", file_id)

        for variable in data.get("variables", []):
            if isinstance(variable, dict) and variable.get("name"):
                variable_id = add_entity("VARIABLE", variable["name"], variable)
                add_relationship(program_id, "USES", variable_id)

        for rel in data.get("relationships", []):
            if not isinstance(rel, dict):
                continue
            src = str(rel.get("source", ""))
            tgt = str(rel.get("target", ""))
            rel_type = str(rel.get("relationship") or rel.get("type") or "RELATED_TO")
            if src and tgt:
                src_id = add_entity("REFERENCE", src)
                tgt_id = add_entity("REFERENCE", tgt)
                add_relationship(src_id, rel_type, tgt_id)

    elif source_type == "ssis":
        package_name = source_stem
        package_id = add_entity("SSIS_PACKAGE", package_name)
        add_relationship(artifact_entity_id, "DEFINES", package_id)

        for task in data.get("tasks", []):
            if isinstance(task, dict):
                name = task.get("name") or task.get("task_name") or task.get("id")
            else:
                name = str(task)
            if name:
                task_id = add_entity(
                    "SSIS_TASK",
                    str(name),
                    task if isinstance(task, dict) else {},
                )
                add_relationship(package_id, "CONTAINS", task_id)

        for connection in data.get("connections", []):
            if isinstance(connection, dict):
                name = (
                    connection.get("name")
                    or connection.get("connection_name")
                    or connection.get("id")
                )
            else:
                name = str(connection)
            if name:
                connection_id = add_entity(
                    "CONNECTION",
                    str(name),
                    connection if isinstance(connection, dict) else {},
                )
                add_relationship(package_id, "USES_CONNECTION", connection_id)

        for rel in data.get("relationships", []):
            if not isinstance(rel, dict):
                continue
            src = str(rel.get("source", ""))
            tgt = str(rel.get("target", ""))
            rel_type = str(rel.get("relationship") or rel.get("type") or "RELATED_TO")
            if src and tgt:
                src_id = add_entity("REFERENCE", src)
                tgt_id = add_entity("REFERENCE", tgt)
                add_relationship(src_id, rel_type, tgt_id)

    return KnowledgeDocument(
        artifacts=[artifact],
        entities=entities,
        relationships=relationships,
        evidence=evidence,
        business_rules=business_rules,
    )


def normalize_file(
    path: Path,
    source_type: str,
    source_path: str | Path | None = None,
    project_root: Path | None = None,
) -> KnowledgeDocument:
    data = load_json(path)
    if source_path is None:
        source_rel = infer_source_relative_path(path, source_type, project_root)
    else:
        source_rel = to_project_relative(source_path, project_root, allow_outside=True)

    return normalize_source_metadata(data, source_rel, source_type, project_root)
