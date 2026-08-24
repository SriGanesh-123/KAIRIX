"""Configuration-driven registry for predefined investigation output formats."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FormatDefinition:
    """Validated definition of a predefined investigation output format."""

    name: str
    description: str
    required_sections: tuple[str, ...]
    required_fields: tuple[str, ...]
    require_evidence: bool = True
    require_confidence: bool = True
    require_knowledge_gaps: bool = True
    require_trace: bool = True

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "FormatDefinition":
        name = str(value.get("name", "")).strip()
        if not name:
            raise ValueError("format definition requires name")
        description = str(value.get("description", "")).strip()

        def strings(key: str) -> tuple[str, ...]:
            raw = value.get(key, [])
            if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
                raise ValueError(f"{key} must be a list of strings")
            return tuple(item.strip() for item in raw if item.strip())

        return cls(
            name=name,
            description=description,
            required_sections=strings("required_sections"),
            required_fields=strings("required_fields"),
            require_evidence=bool(value.get("require_evidence", True)),
            require_confidence=bool(value.get("require_confidence", True)),
            require_knowledge_gaps=bool(value.get("require_knowledge_gaps", True)),
            require_trace=bool(value.get("require_trace", True)),
        )

    def validate(self, result: dict[str, Any]) -> list[str]:
        """Return format violations without modifying the investigation result."""
        errors: list[str] = []
        for field in self.required_fields:
            if field not in result:
                errors.append(f"Missing required field: {field}")
        if self.require_evidence and not result.get("evidence_ids"):
            errors.append("At least one evidence reference is required")
        if self.require_confidence and "confidence" not in result:
            errors.append("Confidence is required")
        if self.require_knowledge_gaps and "knowledge_gaps" not in result:
            errors.append("Knowledge gaps are required")
        if self.require_trace and not result.get("trace_references"):
            errors.append("Trace references are required")
        return errors


class FormatRegistry:
    """Load and resolve predefined formats from JSON configuration."""

    def __init__(self, definitions_dir: Path | None = None) -> None:
        self.definitions_dir = definitions_dir or Path(__file__).with_name("definitions")
        self._formats: dict[str, FormatDefinition] | None = None

    @staticmethod
    def _key(value: str) -> str:
        return "_".join(value.strip().lower().replace("-", "_").split())

    def _load(self) -> dict[str, FormatDefinition]:
        if self._formats is not None:
            return self._formats
        loaded: dict[str, FormatDefinition] = {}
        if self.definitions_dir.exists():
            for path in sorted(self.definitions_dir.glob("*.json")):
                data = json.loads(path.read_text(encoding="utf-8"))
                definition = FormatDefinition.from_mapping(data)
                loaded[self._key(path.stem)] = definition
                loaded[self._key(definition.name)] = definition
        self._formats = loaded
        return loaded

    def list_formats(self) -> list[FormatDefinition]:
        unique: dict[str, FormatDefinition] = {}
        for definition in self._load().values():
            unique[definition.name] = definition
        return sorted(unique.values(), key=lambda item: item.name.lower())

    def get(self, name: str) -> FormatDefinition:
        key = self._key(name)
        try:
            return self._load()[key]
        except KeyError as exc:
            available = ", ".join(item.name for item in self.list_formats())
            raise ValueError(f"Unknown investigation format {name!r}. Available: {available}") from exc

    def resolve(self, requested: str | None) -> FormatDefinition | None:
        if not requested or not requested.strip():
            return None
        return self.get(requested)
