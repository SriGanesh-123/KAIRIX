"""Authoritative ingestion manifest for incremental single-file knowledge processing."""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Dict, Optional

from .paths import PROJECT_ROOT, get_project_root, to_project_relative

DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "output" / "knowledge" / "manifest.json"


def get_default_manifest_path(project_root: Path | None = None) -> Path:
    root = project_root or get_project_root()
    return root / "output" / "knowledge" / "manifest.json"


def load_manifest(path: Path | None = None, project_root: Path | None = None) -> Dict[str, Any]:
    manifest_path = path or get_default_manifest_path(project_root)
    if not manifest_path.exists():
        return {
            "manifest_version": "1.0",
            "last_updated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "files": {},
        }
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"manifest_version": "1.0", "files": {}}
        if "files" not in data or not isinstance(data["files"], dict):
            data["files"] = {}
        return data
    except (OSError, json.JSONDecodeError):
        return {
            "manifest_version": "1.0",
            "last_updated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "files": {},
        }


def save_manifest_atomic(
    manifest_data: Dict[str, Any],
    path: Path | None = None,
    project_root: Path | None = None,
) -> Path:
    manifest_path = path or get_default_manifest_path(project_root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = manifest_path.with_suffix(".json.tmp")

    payload = dict(manifest_data)
    payload["last_updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    raw_json = json.dumps(payload, indent=2, ensure_ascii=False)

    # Validate JSON serializability before writing
    json.loads(raw_json)

    temp_path.write_text(raw_json, encoding="utf-8")
    # Verify written content before replacing
    json.loads(temp_path.read_text(encoding="utf-8"))
    temp_path.replace(manifest_path)
    return manifest_path


class IngestionManifest:
    """Thread-safe and atomic manager for the knowledge ingestion manifest."""

    def __init__(
        self,
        manifest_path: Path | None = None,
        project_root: Path | None = None,
    ) -> None:
        self.project_root = project_root or get_project_root()
        self.manifest_path = manifest_path or get_default_manifest_path(self.project_root)
        self.data: Dict[str, Any] = load_manifest(self.manifest_path, self.project_root)

    def reload(self) -> None:
        self.data = load_manifest(self.manifest_path, self.project_root)

    def get_entry(self, source_path: str | Path) -> Optional[Dict[str, Any]]:
        rel_path = to_project_relative(source_path, self.project_root, allow_outside=True)
        return self.data.get("files", {}).get(rel_path)

    def check_status(self, source_path: str | Path, current_sha256: str) -> str:
        """Return 'NEW', 'UNCHANGED', or 'MODIFIED'."""
        entry = self.get_entry(source_path)
        if not entry:
            return "NEW"
        if entry.get("status") == "INGESTED" and entry.get("sha256") == current_sha256:
            return "UNCHANGED"
        return "MODIFIED"

    def record_ingested(
        self,
        source_path: str | Path,
        *,
        artifact_id: str,
        source_type: str,
        sha256: str,
        mtime: float,
        size_bytes: int,
        parser_output: str,
        summary_output: str,
        extra: Dict[str, Any] | None = None,
    ) -> None:
        rel_path = to_project_relative(source_path, self.project_root, allow_outside=True)
        files = self.data.setdefault("files", {})
        files[rel_path] = {
            "artifact_id": artifact_id,
            "source_type": source_type,
            "sha256": sha256,
            "mtime": mtime,
            "size_bytes": size_bytes,
            "status": "INGESTED",
            "parser_output": to_project_relative(parser_output, self.project_root, allow_outside=True),
            "summary_output": to_project_relative(summary_output, self.project_root, allow_outside=True),
            **(extra or {}),
        }
        self.save()

    def record_failed(
        self,
        source_path: str | Path,
        *,
        error: str,
        sha256: str | None = None,
    ) -> None:
        rel_path = to_project_relative(source_path, self.project_root, allow_outside=True)
        entry = self.get_entry(source_path) or {}
        entry["status"] = "FAILED"
        entry["last_error"] = error
        if sha256:
            entry["sha256"] = sha256
        self.data.setdefault("files", {})[rel_path] = entry
        self.save()

    def save(self) -> Path:
        return save_manifest_atomic(self.data, self.manifest_path, self.project_root)
