"""Neo4j persistence for the canonical knowledge graph.

All graph facts are supplied by the validated canonical/relationship-discovery
payload. Nested values are serialized generically so no domain-specific fields
are hardcoded into the persistence layer.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable

try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover
    GraphDatabase = None


def _neo4j_value(value: Any) -> Any:
    """Convert arbitrary JSON-compatible values to Neo4j property values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        if all(item is None or isinstance(item, (str, int, float, bool)) for item in value):
            return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _node_for_neo4j(node: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(node)
    payload["properties_json"] = json.dumps(node.get("properties", {}), ensure_ascii=False, sort_keys=True, default=str)
    payload.pop("properties", None)
    # MERGE requires a non-null stable node identifier.
    if not payload.get("id"):
        raise ValueError("Knowledge graph node is missing a non-empty id")
    return {key: _neo4j_value(value) for key, value in payload.items()}


def _edge_for_neo4j(edge: Dict[str, Any], index: int) -> Dict[str, Any]:
    payload = dict(edge)
    for key in ("evidence", "properties"):
        if key in payload:
            payload[f"{key}_json"] = json.dumps(payload[key], ensure_ascii=False, sort_keys=True, default=str)
            payload.pop(key, None)
    # Some relationship-discovery edges are derived objects without a
    # canonical relationship id. Give them a deterministic, content-derived
    # identifier rather than allowing a null Neo4j relationship key.
    edge_id = payload.get("id")
    if not edge_id:
        source = payload.get("source_entity_id") or payload.get("source")
        target = payload.get("target_entity_id") or payload.get("target")
        rel_type = payload.get("relationship_type") or payload.get("relationship")
        if not source or not target or not rel_type:
            raise ValueError(f"Knowledge graph relationship at index {index} is missing stable endpoint/type fields")
        stable = json.dumps([str(source), str(rel_type), str(target), payload.get("discovery_method", "")], ensure_ascii=False, separators=(",", ":"))
        import hashlib
        edge_id = f"derived-rel:{hashlib.sha256(stable.encode('utf-8')).hexdigest()[:24]}"
        payload["id"] = edge_id
    source_id = payload.get("source_entity_id") or payload.get("source")
    target_id = payload.get("target_entity_id") or payload.get("target")
    if not source_id or not target_id:
        raise ValueError(f"Knowledge graph relationship at index {index} is missing source/target entity ids")
    payload["source_entity_id"] = str(source_id)
    payload["target_entity_id"] = str(target_id)
    return {key: _neo4j_value(value) for key, value in payload.items()}


class Neo4jKnowledgeGraphStore:
    """Persist graph nodes and edges into Neo4j using idempotent MERGE operations."""

    def __init__(self, uri: str | None = None, username: str | None = None, password: str | None = None, database: str | None = None) -> None:
        self.uri = uri or os.getenv("NEO4J_URI")
        self.username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD")
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")
        self._driver = None

    @property
    def configured(self) -> bool:
        return bool(GraphDatabase and self.uri and self.password)

    def connect(self) -> None:
        if not self.configured:
            raise RuntimeError("Neo4j is not configured. Set NEO4J_URI and NEO4J_PASSWORD (and optionally NEO4J_USERNAME/NEO4J_DATABASE).")
        self._driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        self._driver.verify_connectivity()

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def initialize_constraints(self) -> None:
        self._require_connection()
        with self._driver.session(database=self.database) as session:
            session.run("CREATE CONSTRAINT kg_node_id IF NOT EXISTS FOR (n:KGNode) REQUIRE n.id IS UNIQUE")
            session.run("CREATE CONSTRAINT kg_edge_id IF NOT EXISTS FOR ()-[r:KG_RELATIONSHIP]-() REQUIRE r.id IS UNIQUE")

    def write_graph(self, graph: Dict[str, Any], *, clear_existing: bool = False) -> Dict[str, int]:
        self._require_connection()
        nodes = [_node_for_neo4j(node) for node in graph.get("nodes", [])]
        edges = [_edge_for_neo4j(edge, index) for index, edge in enumerate(graph.get("edges", []))]
        with self._driver.session(database=self.database) as session:
            if clear_existing:
                session.run("MATCH (n:KGNode) DETACH DELETE n")
            session.execute_write(self._write_nodes, nodes)
            session.execute_write(self._write_edges, edges)
        return self.counts()

    def write_artifact_graph(self, artifact_id: str, graph: Dict[str, Any]) -> Dict[str, int]:
        """Update nodes and relationships scoped strictly to one artifact."""
        self._require_connection()
        nodes = [
            _node_for_neo4j(node)
            for node in graph.get("nodes", [])
            if node.get("artifact_id") == artifact_id
        ]
        edges = [
            _edge_for_neo4j(edge, index)
            for index, edge in enumerate(graph.get("edges", []))
            if edge.get("artifact_id") == artifact_id
            or edge.get("source_artifact_id") == artifact_id
        ]
        with self._driver.session(database=self.database) as session:
            # Delete only relationships owned by this artifact
            session.run(
                "MATCH ()-[r:KG_RELATIONSHIP {artifact_id: $artifact_id}]->() DELETE r",
                artifact_id=str(artifact_id),
            )
            if nodes:
                session.execute_write(self._write_nodes, nodes)
            if edges:
                session.execute_write(self._write_edges, edges)
        return self.counts()

    @staticmethod
    def _write_nodes(tx: Any, nodes: Iterable[Dict[str, Any]]) -> None:
        query = """
        UNWIND $nodes AS node
        MERGE (n:KGNode {id: node.id})
        SET n.type = node.type,
            n.name = node.name,
            n.artifact_id = node.artifact_id,
            n.properties_json = node.properties_json,
            n.source_confidence = node.source_confidence,
            n.reconciled_confidence = node.reconciled_confidence
        """
        tx.run(query, nodes=list(nodes))

    @staticmethod
    def _write_edges(tx: Any, edges: Iterable[Dict[str, Any]]) -> None:
        query = """
        UNWIND $edges AS edge
        MATCH (s:KGNode {id: edge.source_entity_id})
        MATCH (t:KGNode {id: edge.target_entity_id})
        MERGE (s)-[r:KG_RELATIONSHIP {id: edge.id}]->(t)
        SET r.relationship_type = edge.relationship_type,
            r.artifact_id = edge.artifact_id,
            r.source_artifact_id = edge.source_artifact_id,
            r.target_artifact_id = edge.target_artifact_id,
            r.evidence_ids = edge.evidence_ids,
            r.evidence_json = edge.evidence_json,
            r.properties_json = edge.properties_json,
            r.confidence = edge.confidence,
            r.validation_status = edge.validation_status,
            r.discovery_method = edge.discovery_method
        """
        tx.run(query, edges=list(edges))

    def counts(self) -> Dict[str, int]:
        self._require_connection()
        with self._driver.session(database=self.database) as session:
            node_count = session.run("MATCH (n:KGNode) RETURN count(n) AS count").single()["count"]
            edge_count = session.run("MATCH ()-[r:KG_RELATIONSHIP]->() RETURN count(r) AS count").single()["count"]
            rows = session.run("MATCH ()-[r:KG_RELATIONSHIP]->() RETURN r.validation_status AS status, count(r) AS count")
            by_status = {row["status"]: row["count"] for row in rows}
        return {
            "nodes": node_count,
            "edges": edge_count,
            "supported": by_status.get("SUPPORTED", 0),
            "unverified": by_status.get("UNVERIFIED", 0),
            "conflicts": by_status.get("CONFLICT", 0),
        }

    def _require_connection(self) -> None:
        if self._driver is None:
            raise RuntimeError("Neo4j store is not connected. Call connect() first.")
