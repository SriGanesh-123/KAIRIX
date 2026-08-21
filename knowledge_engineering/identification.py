"""Artifact identification for the Knowledge Engineering Agent.

This stage normalizes the identity of each canonical artifact before parser
selection. It does not parse or mutate source files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def identify_artifacts(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create an auditable identification record for every canonical artifact."""
    result: list[dict[str, Any]] = []
    for artifact in artifacts:
        artifact_id = artifact.get("id")
        file_name = artifact.get("file_name")
        source_type = artifact.get("source_type")
        result.append(
            {
                "artifact_id": artifact_id,
                "file_name": file_name,
                "source_type": source_type,
                "artifact_kind": artifact.get("artifact_type") or source_type or "unknown",
                "extension": Path(file_name).suffix.lower() if file_name else None,
                "status": "IDENTIFIED" if artifact_id and file_name else "INCOMPLETE",
                "reason": "Canonical artifact identity is available"
                if artifact_id and file_name
                else "Artifact id or file name is missing",
            }
        )
    return result
