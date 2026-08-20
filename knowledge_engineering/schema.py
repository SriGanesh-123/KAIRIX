"""Canonical metadata schema for SQL, COBOL, and SSIS knowledge."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Artifact(BaseModel):
    id: str
    source_type: str
    file_name: str
    path: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Evidence(BaseModel):
    id: str
    artifact_id: str
    source_type: str
    location: Optional[str] = None
    text: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Entity(BaseModel):
    id: str
    entity_type: str
    name: str
    artifact_id: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    evidence_ids: List[str] = Field(default_factory=list)


class Relationship(BaseModel):
    id: str
    source_entity_id: str
    relationship_type: str
    target_entity_id: str
    artifact_id: Optional[str] = None
    evidence_ids: List[str] = Field(default_factory=list)
    properties: Dict[str, Any] = Field(default_factory=dict)


class BusinessRule(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    artifact_id: str
    evidence_ids: List[str] = Field(default_factory=list)
    confidence: float = 1.0
    properties: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeDocument(BaseModel):
    schema_version: str = "1.0"
    artifacts: List[Artifact] = Field(default_factory=list)
    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)
    business_rules: List[BusinessRule] = Field(default_factory=list)
