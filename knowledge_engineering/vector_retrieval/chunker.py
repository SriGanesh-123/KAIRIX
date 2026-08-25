"""Convert canonical KAIRIX knowledge into provenance-rich vector chunks."""

from __future__ import annotations

import json
from typing import Any, Dict, List
from uuid import NAMESPACE_URL, uuid5


class KnowledgeChunker:
    """Create deterministic, self-contained chunks from canonical metadata."""

    def build_chunks(self, canonical: Dict[str, Any]) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        artifacts = {item.get("artifact_id"): item for item in canonical.get("artifacts", [])}

        for artifact in canonical.get("artifacts", []):
            artifact_id = artifact.get("artifact_id")
            name = artifact.get("file_name") or artifact_id or "unknown-artifact"
            text = (
                f"Artifact: {name}. Source type: {artifact.get('source_type')}. "
                f"Kind: {artifact.get('artifact_kind')}. Status: {artifact.get('status')}. "
                f"Confidence: {artifact.get('reconciled_confidence')} ({artifact.get('confidence_level')})."
            )
            chunks.append(self._chunk("artifact", artifact_id, artifact_id, text, artifact))

        for entity in canonical.get("entities", []):
            entity_id = entity.get("id")
            artifact_id = entity.get("artifact_id")
            artifact = artifacts.get(artifact_id, {})
            text = self._entity_text(entity, artifact)
            chunks.append(self._chunk("entity", entity_id, artifact_id, text, entity))

        for relationship in canonical.get("relationships", []):
            relationship_id = relationship.get("id")
            artifact_id = relationship.get("artifact_id")
            source = relationship.get("source_entity_id", relationship.get("source"))
            target = relationship.get("target_entity_id", relationship.get("target"))
            rel_type = relationship.get("relationship_type", relationship.get("relationship"))
            text = (
                f"Relationship: {source} {rel_type} {target}. "
                f"Artifact: {artifact_id}. Validation: {relationship.get('validation_status')}. "
                f"Confidence: {relationship.get('confidence')}. "
                f"Discovery method: {relationship.get('discovery_method', 'canonical_metadata')}."
            )
            chunks.append(self._chunk("relationship", relationship_id, artifact_id, text, relationship))

        for index, rule in enumerate(canonical.get("business_rules", [])):
            artifact_id = rule.get("artifact_id")
            text = f"Business rule: {self._json_text(rule)}"
            chunks.append(self._chunk("business_rule", rule.get("id") or str(index), artifact_id, text, rule))

        for index, claim in enumerate(canonical.get("claims", [])):
            artifact_id = claim.get("artifact_id")
            text = (
                f"Claim for artifact {artifact_id}: field={claim.get('field')}; "
                f"claim={claim.get('claim')}; status={claim.get('status')}; "
                f"confidence={claim.get('claim_confidence')}; reason={claim.get('reason')}"
            )
            chunks.append(self._chunk("claim", claim.get("id") or str(index), artifact_id, text, claim))

        for index, item in enumerate(canonical.get("reference_matches", [])):
            artifact_id = item.get("artifact_id")
            text = f"Reference match: {self._json_text(item)}"
            chunks.append(self._chunk("reference_match", item.get("id") or str(index), artifact_id, text, item))

        for index, gap in enumerate(canonical.get("knowledge_gaps", [])):
            artifact_id = gap.get("artifact_id")
            text = f"Knowledge gap: type={gap.get('type')}; reason={gap.get('reason')}"
            chunks.append(self._chunk("knowledge_gap", gap.get("id") or str(index), artifact_id, text, gap))

        return chunks

    def build_artifact_chunks(self, canonical: Dict[str, Any], artifact_id: str) -> List[Dict[str, Any]]:
        """Create chunks scoped strictly to one artifact."""
        scoped_canonical = {
            "artifacts": [a for a in canonical.get("artifacts", []) if a.get("artifact_id") == artifact_id or a.get("id") == artifact_id],
            "entities": [e for e in canonical.get("entities", []) if e.get("artifact_id") == artifact_id],
            "relationships": [r for r in canonical.get("relationships", []) if r.get("artifact_id") == artifact_id or r.get("source_artifact_id") == artifact_id],
            "business_rules": [b for b in canonical.get("business_rules", []) if b.get("artifact_id") == artifact_id],
            "claims": [c for c in canonical.get("claims", []) if c.get("artifact_id") == artifact_id],
            "reference_matches": [m for m in canonical.get("reference_matches", []) if m.get("artifact_id") == artifact_id],
            "knowledge_gaps": [g for g in canonical.get("knowledge_gaps", []) if g.get("artifact_id") == artifact_id],
        }
        return self.build_chunks(scoped_canonical)

    @staticmethod
    def _entity_text(entity: Dict[str, Any], artifact: Dict[str, Any]) -> str:
        properties = entity.get("properties") or {}
        return (
            f"Entity: {entity.get('name')}. Type: {entity.get('entity_type', 'ENTITY')}. "
            f"Artifact: {artifact.get('file_name') or entity.get('artifact_id')}. "
            f"Artifact ID: {entity.get('artifact_id')}. "
            f"Properties: {KnowledgeChunker._json_text(properties)}. "
            f"Source confidence: {entity.get('source_confidence')}; "
            f"reconciled confidence: {entity.get('reconciled_confidence')}."
        )

    @staticmethod
    def _json_text(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)

    @staticmethod
    def _chunk(kind: str, source_id: Any, artifact_id: Any, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        stable_key = f"{kind}|{artifact_id or ''}|{source_id or ''}|{text}"
        chunk_id = str(uuid5(NAMESPACE_URL, f"kairix:{stable_key}"))
        return {
            "id": chunk_id,
            "text": text,
            "kind": kind,
            "source_id": source_id,
            "artifact_id": artifact_id,
            "metadata": KnowledgeChunker._safe_metadata(metadata),
        }

    @staticmethod
    def _safe_metadata(value: Dict[str, Any]) -> Dict[str, Any]:
        # Qdrant payloads must contain JSON-compatible values. Preserve the
        # original metadata as a JSON string rather than dropping provenance.
        return {"source_json": json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)}
