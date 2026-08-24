from __future__ import annotations

import pytest

from knowledge_engineering.investigation_agent.formats import FormatRegistry


def _result() -> dict:
    return {
        "answer": "Supported answer",
        "evidence_ids": ["entity:1"],
        "confidence": 0.8,
        "knowledge_gaps": [],
        "trace_references": [{"step": 1, "query": "q", "evidence_ids": ["entity:1"]}],
    }


def test_registry_loads_predefined_formats() -> None:
    registry = FormatRegistry()
    names = {item.name for item in registry.list_formats()}
    assert names == {
        "Relationship Report",
        "Dependency Report",
        "Lineage Report",
        "Impact Analysis",
    }


def test_registry_resolves_name_and_filename_style_alias() -> None:
    registry = FormatRegistry()
    assert registry.get("Dependency Report").name == "Dependency Report"
    assert registry.get("dependency_report").name == "Dependency Report"


def test_format_validation_accepts_grounded_result() -> None:
    definition = FormatRegistry().get("Relationship Report")
    assert definition.validate(_result()) == []


def test_format_validation_reports_missing_evidence_and_trace() -> None:
    definition = FormatRegistry().get("Relationship Report")
    errors = definition.validate({"answer": "incomplete"})
    assert "At least one evidence reference is required" in errors
    assert "Trace references are required" in errors


def test_unknown_format_is_explicit() -> None:
    with pytest.raises(ValueError, match="Unknown investigation format"):
        FormatRegistry().get("unknown_report")
