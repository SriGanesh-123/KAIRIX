"""Authoritative project-relative path normalization and hashing utilities."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_project_root() -> Path:
    """Return the absolute root directory of the KAIRIX project."""
    return PROJECT_ROOT


def to_project_relative(
    path: str | Path,
    project_root: Path | None = None,
    *,
    allow_outside: bool = False,
) -> str:
    """Convert any path (absolute or relative) into an authoritative project-relative POSIX string.

    Examples:
        C:\\Users\\...\\KAIRIX\\source\\sql\\foo.sql -> source/sql/foo.sql
        source\\sql\\foo.sql -> source/sql/foo.sql
        source/sql/foo.sql -> source/sql/foo.sql

    Raises:
        ValueError: If the path is outside project_root and allow_outside is False.
    """
    root = (project_root or PROJECT_ROOT).resolve()
    raw_str = str(path).strip().replace("\\", "/")

    if not raw_str:
        raise ValueError("Cannot normalize an empty path")

    p = Path(raw_str)
    if not p.is_absolute():
        # Check if already a project-relative string that resolves inside root
        candidate = (root / p).resolve()
    else:
        candidate = p.resolve()

    try:
        rel = candidate.relative_to(root)
        posix_rel = rel.as_posix()
        # Guard against leading ./
        if posix_rel.startswith("./"):
            posix_rel = posix_rel[2:]
        return posix_rel
    except ValueError as exc:
        if allow_outside:
            return candidate.as_posix()
        raise ValueError(
            f"Path '{path}' resolves to '{candidate}', which is outside project root '{root}'"
        ) from exc


def resolve_project_path(
    path: str | Path,
    project_root: Path | None = None,
) -> Path:
    """Resolve a project-relative or absolute path to an authoritative absolute Path."""
    root = (project_root or PROJECT_ROOT).resolve()
    raw_str = str(path).strip().replace("\\", "/")
    p = Path(raw_str)
    if p.is_absolute():
        return p.resolve()
    return (root / p).resolve()


def compute_file_sha256(path: str | Path, project_root: Path | None = None) -> str:
    """Compute SHA-256 digest of a source file."""
    abs_path = resolve_project_path(path, project_root)
    if not abs_path.exists() or not abs_path.is_file():
        raise FileNotFoundError(f"Source file not found for hashing: {abs_path}")

    hasher = hashlib.sha256()
    with open(abs_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"


def get_source_type_from_path(path: str | Path) -> str:
    """Determine source type ('cobol', 'sql', 'ssis') from file extension."""
    suffix = Path(str(path)).suffix.lower()
    if suffix in (".cbl", ".cob", ".cpy"):
        return "cobol"
    if suffix == ".sql":
        return "sql"
    if suffix == ".dtsx":
        return "ssis"
    raise ValueError(f"Unsupported source file extension: {suffix} for path: {path}")
