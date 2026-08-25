"""Parser registry and selection for the Knowledge Engineering Agent.

The registry deliberately does not reimplement the existing parsers. It provides
one technology-neutral selection contract over the parser implementations that
already live under ``parsers/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ParserSpec:
    name: str
    source_type: str
    extensions: tuple[str, ...]
    entrypoint: str
    output_dir: str
    available: bool = True


PARSER_REGISTRY: tuple[ParserSpec, ...] = (
    ParserSpec(
        name="COBOL parser",
        source_type="cobol",
        extensions=(".cbl", ".cob", ".cpy"),
        entrypoint="parsers.cobol.batch_parse_312_parity",
        output_dir="output/cobol",
    ),
    ParserSpec(
        name="SQL parser",
        source_type="sql",
        extensions=(".sql",),
        entrypoint="parsers.sql.parse",
        output_dir="output/sql",
    ),
    ParserSpec(
        name="SSIS parser",
        source_type="ssis",
        extensions=(".dtsx",),
        entrypoint="parsers.ssis.parse",
        output_dir="output/ssis",
    ),
)


def select_parser(*, source_type: str | None = None, file_name: str | None = None) -> ParserSpec:
    """Select the parser deterministically from canonical metadata or filename."""
    normalized_type = (source_type or "").strip().lower()
    normalized_name = (file_name or "").strip().lower()

    if normalized_type:
        for spec in PARSER_REGISTRY:
            if spec.source_type == normalized_type:
                return spec

    if normalized_name:
        suffix = Path(normalized_name).suffix.lower()
        matches = [spec for spec in PARSER_REGISTRY if suffix in spec.extensions]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous parser selection for file: {file_name}")

    raise ValueError(
        f"No parser registered for source_type={source_type!r}, file_name={file_name!r}"
    )


def select_parsers(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return an auditable parser-selection record for each artifact."""
    selections: list[dict[str, Any]] = []
    for artifact in artifacts:
        artifact_id = artifact.get("id")
        file_name = artifact.get("file_name")
        source_type = artifact.get("source_type")
        source_path = artifact.get("path")
        try:
            spec = select_parser(source_type=source_type, file_name=file_name or source_path)
            selections.append(
                {
                    "artifact_id": artifact_id,
                    "file_name": file_name,
                    "source_path": source_path,
                    "source_type": source_type,
                    "parser": spec.name,
                    "entrypoint": spec.entrypoint,
                    "status": "SELECTED",
                    "reason": "Matched canonical source_type" if source_type else "Matched file extension",
                }
            )
        except ValueError as exc:
            selections.append(
                {
                    "artifact_id": artifact_id,
                    "file_name": file_name,
                    "source_path": source_path,
                    "source_type": source_type,
                    "parser": None,
                    "entrypoint": None,
                    "status": "UNSUPPORTED",
                    "reason": str(exc),
                }
            )
    return selections
