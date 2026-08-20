"""Validation for canonical Knowledge Engineering metadata."""

from __future__ import annotations

from typing import Iterable

from .schema import KnowledgeDocument


def validate_document(document: KnowledgeDocument) -> list[str]:
    errors: list[str] = []

    artifact_ids = {item.id for item in document.artifacts}
    entity_ids = {item.id for item in document.entities}
    evidence_ids = {item.id for item in document.evidence}

    for entity in document.entities:
        if entity.artifact_id not in artifact_ids:
            errors.append(
                f"Entity {entity.id} references missing artifact {entity.artifact_id}"
            )

        for evidence_id in entity.evidence_ids:
            if evidence_id not in evidence_ids:
                errors.append(
                    f"Entity {entity.id} references missing evidence {evidence_id}"
                )

    seen_relationship_ids: set[str] = set()

    for relationship in document.relationships:
        if relationship.id in seen_relationship_ids:
            errors.append(
                f"Duplicate relationship id: {relationship.id}"
            )
        seen_relationship_ids.add(relationship.id)

        if relationship.source_entity_id not in entity_ids:
            errors.append(
                f"Relationship {relationship.id} has missing source entity "
                f"{relationship.source_entity_id}"
            )

        if relationship.target_entity_id not in entity_ids:
            errors.append(
                f"Relationship {relationship.id} has missing target entity "
                f"{relationship.target_entity_id}"
            )

        if relationship.artifact_id and relationship.artifact_id not in artifact_ids:
            errors.append(
                f"Relationship {relationship.id} references missing artifact "
                f"{relationship.artifact_id}"
            )

        for evidence_id in relationship.evidence_ids:
            if evidence_id not in evidence_ids:
                errors.append(
                    f"Relationship {relationship.id} references missing evidence "
                    f"{evidence_id}"
                )

    for rule in document.business_rules:
        if rule.artifact_id not in artifact_ids:
            errors.append(
                f"Business rule {rule.id} references missing artifact "
                f"{rule.artifact_id}"
            )

        if not 0.0 <= rule.confidence <= 1.0:
            errors.append(
                f"Business rule {rule.id} has invalid confidence {rule.confidence}"
            )

        for evidence_id in rule.evidence_ids:
            if evidence_id not in evidence_ids:
                errors.append(
                    f"Business rule {rule.id} references missing evidence "
                    f"{evidence_id}"
                )

    return errors
