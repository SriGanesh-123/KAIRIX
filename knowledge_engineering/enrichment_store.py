"""Merge and preserve knowledge-enrichment results across agent runs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_existing(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def merge_enrichment(previous: Dict[str, Any] | None, current: Dict[str, Any]) -> Dict[str, Any]:
    """Preserve successful LLM reviews while accepting current deterministic data."""
    if not previous:
        return current

    previous_reviews = {
        item.get("artifact_id"): item
        for item in previous.get("artifact_reviews", [])
        if item.get("artifact_id")
    }
    merged_reviews = []
    for item in current.get("artifact_reviews", []):
        artifact_id = item.get("artifact_id")
        old = previous_reviews.get(artifact_id)
        if old and old.get("status") == "LLM_REVIEWED":
            merged_reviews.append(old)
        else:
            merged_reviews.append(item)

    result = dict(current)
    result["artifact_reviews"] = merged_reviews
    result["summary"] = dict(current.get("summary", {}))
    result["summary"]["reviews"] = len(merged_reviews)
    result["summary"]["llm_reviews_completed"] = sum(
        item.get("status") == "LLM_REVIEWED" for item in merged_reviews
    )
    result["summary"]["llm_reviews_pending"] = sum(
        item.get("status") == "LLM_REVIEW_PENDING" for item in merged_reviews
    )
    return result


def write_merged(path: Path, current: Dict[str, Any]) -> Dict[str, Any]:
    previous = load_existing(path)
    merged = merge_enrichment(previous, current)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    return merged
