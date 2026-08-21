"""Generic cross-artifact candidate discovery from canonical reference evidence."""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

from .schema import relationship


def _entity_index(canonical: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(item.get("id")): item
        for item in canonical.get("entities", [])
        if item.get("id")
    }


def _artifact_id(entity: Dict[str, Any]) -> str | None:
    value = entity.get("artifact_id")
    if value:
        return str(value)
    provenance = entity.get("provenance")
    if isinstance(provenance, dict):
        artifact = provenance.get("artifact")
        if isinstance(artifact, dict) and artifact.get("artifact_id"):
            return str(artifact["artifact_id"])
    return None


def discover_reference_candidates(
    canonical: Dict[str, Any],
    existing_keys: Set[Tuple[str, str, str]],
) -> List[Dict[str, Any]]:
    """Turn explicit canonical reference matches into cross-artifact candidates.

    The function uses only IDs, artifact ownership and evidence already present
    in canonical metadata. It never guesses from artifact names, table names,
    column names, or project-specific vocabulary.
    """
    entities = _entity_index(canonical)
    candidates: List[Dict[str, Any]] = []
    seen = set(existing_keys)

    for match in canonical.get("reference_matches", []):
        if not isinstance(match, dict):
            continue
        source_id = match.get("reference_entity_id")
        target_ids = match.get("candidate_entity_ids", [])
        if not source_id or not isinstance(target_ids, list):
            continue

        source = entities.get(str(source_id))
        if not source:
            continue
        source_artifact = _artifact_id(source)
        if not source_artifact:
            continue

        confidence = float(match.get("confidence", 0.0) or 0.0)
        validation = "SUPPORTED" if len(target_ids) == 1 and confidence >= 0.75 else "UNVERIFIED"

        for target_id in target_ids:
            target = entities.get(str(target_id))
            if not target:
                continue
            target_artifact = _artifact_id(target)
            if not target_artifact or target_artifact == source_artifact:
                continue

            key = (str(source_id), "REFERENCES", str(target_id))
            if key in seen:
                continue

            evidence = {
                "reference_match": {
                    "reference_entity_id": source_id,
                    "candidate_entity_id": target_id,
                    "normalized_name": match.get("normalized_name"),
                    "confidence": confidence,
                },
                "source_artifact_id": source_artifact,
                "target_artifact_id": target_artifact,
            }

            candidates.append({
                **relationship(
                    str(source_id),
                    "REFERENCES",
                    str(target_id),
                    evidence=[evidence],
                    confidence=confidence,
                    discovery_method="canonical_reference_match",
                    validation_status=validation,
                ),
                "source_artifact_id": source_artifact,
                "target_artifact_id": target_artifact,
                "canonical_relationship_id": None,
            })
            seen.add(key)

    return candidates
