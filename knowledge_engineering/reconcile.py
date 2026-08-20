"""Reconcile canonical entities without changing source facts."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict


def normalize_name(value: str) -> str:
    value = re.sub(r"[^A-Z0-9]+", "_", str(value).upper()).strip("_")
    return value


def build_reconciliation(document: Dict[str, Any]) -> Dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for entity in document.get("entities", []):
        entity_type = str(entity.get("entity_type", "UNKNOWN"))
        name = normalize_name(entity.get("name", ""))
        if name:
            groups[(entity_type, name)].append(entity)

    aliases = []
    duplicates = []
    for (entity_type, name), items in groups.items():
        ids = [item["id"] for item in items]
        if len(ids) > 1:
            duplicates.append({"entity_type": entity_type, "canonical_name": name, "entity_ids": ids})
        aliases.append({"entity_type": entity_type, "normalized_name": name, "entity_ids": ids})

    reference_matches = _match_references(document)
    return {
        "normalized_entity_groups": aliases,
        "duplicate_groups": duplicates,
        "reference_matches": reference_matches,
        "summary": {
            "normalized_groups": len(aliases),
            "duplicate_groups": len(duplicates),
            "reference_matches": len(reference_matches),
        },
    }


def _match_references(document: Dict[str, Any]) -> list[dict[str, Any]]:
    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entity in document.get("entities", []):
        if entity.get("entity_type") != "REFERENCE":
            by_name[normalize_name(entity.get("name", ""))].append(entity)

    matches = []
    for reference in document.get("entities", []):
        if reference.get("entity_type") != "REFERENCE":
            continue
        key = normalize_name(reference.get("name", ""))
        candidates = by_name.get(key, [])
        if candidates:
            matches.append(
                {
                    "reference_entity_id": reference["id"],
                    "candidate_entity_ids": [item["id"] for item in candidates],
                    "normalized_name": key,
                    "confidence": 0.75 if len(candidates) == 1 else 0.55,
                }
            )
    return matches
