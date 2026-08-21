"""Read-only Neo4j query layer for knowledge graph retrieval."""

from __future__ import annotations

from typing import Any, Dict, List

from .neo4j_store import Neo4jKnowledgeGraphStore


class KnowledgeGraphQuery:
    """Execute parameterized, read-only graph queries without domain hardcoding."""

    def __init__(self, store: Neo4jKnowledgeGraphStore | None = None) -> None:
        self.store = store or Neo4jKnowledgeGraphStore()
        self._owns_store = store is None

    def __enter__(self) -> "KnowledgeGraphQuery":
        self.store.connect()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._owns_store:
            self.store.close()

    def neighbors(self, node_id: str, relationship_type: str | None = None, limit: int = 50) -> List[Dict[str, Any]]:
        self._validate_limit(limit)
        self._require_connected()
        query = """
        MATCH (n:KGNode {id: $node_id})-[r:KG_RELATIONSHIP]->(m:KGNode)
        WHERE $relationship_type IS NULL OR r.relationship_type = $relationship_type
        RETURN m.id AS id, m.name AS name, m.type AS type,
               r.id AS relationship_id, r.relationship_type AS relationship_type,
               r.validation_status AS validation_status,
               r.confidence AS confidence, m.artifact_id AS artifact_id
        LIMIT $limit
        """
        with self.store._driver.session(database=self.store.database) as session:
            return [dict(row) for row in session.run(query, node_id=node_id, relationship_type=relationship_type, limit=limit)]

    def search_nodes(self, text: str, limit: int = 20) -> List[Dict[str, Any]]:
        self._validate_limit(limit)
        self._require_connected()
        query = """
        MATCH (n:KGNode)
        WHERE toLower(coalesce(n.name, '')) CONTAINS toLower($text)
           OR toLower(coalesce(n.id, '')) CONTAINS toLower($text)
        RETURN n.id AS id, n.name AS name, n.type AS type,
               n.artifact_id AS artifact_id,
               n.reconciled_confidence AS confidence
        ORDER BY n.name
        LIMIT $limit
        """
        with self.store._driver.session(database=self.store.database) as session:
            return [dict(row) for row in session.run(query, text=text, limit=limit)]

    def cross_artifact_relationships(self, limit: int = 100) -> List[Dict[str, Any]]:
        self._validate_limit(limit)
        self._require_connected()
        query = """
        MATCH (a:KGNode)-[r:KG_RELATIONSHIP]->(b:KGNode)
        WHERE a.artifact_id IS NOT NULL AND b.artifact_id IS NOT NULL
          AND a.artifact_id <> b.artifact_id
        RETURN a.id AS source_id, a.name AS source_name,
               a.artifact_id AS source_artifact_id,
               r.id AS relationship_id, r.relationship_type AS relationship_type,
               r.validation_status AS validation_status,
               b.id AS target_id, b.name AS target_name,
               b.artifact_id AS target_artifact_id
        ORDER BY a.name, b.name
        LIMIT $limit
        """
        with self.store._driver.session(database=self.store.database) as session:
            return [dict(row) for row in session.run(query, limit=limit)]

    def traverse(self, node_id: str, hops: int = 2, limit: int = 100) -> List[Dict[str, Any]]:
        if not 1 <= hops <= 4:
            raise ValueError("hops must be between 1 and 4")
        self._validate_limit(limit)
        self._require_connected()
        query = f"""
        MATCH p=(n:KGNode {{id: $node_id}})-[:KG_RELATIONSHIP*1..{hops}]->(m:KGNode)
        RETURN [x IN nodes(p) | {{id: x.id, name: x.name, type: x.type, artifact_id: x.artifact_id}}] AS nodes,
               [x IN relationships(p) | {{id: x.id, relationship_type: x.relationship_type, validation_status: x.validation_status}}] AS relationships
        LIMIT $limit
        """
        with self.store._driver.session(database=self.store.database) as session:
            return [dict(row) for row in session.run(query, node_id=node_id, limit=limit)]

    def _require_connected(self) -> None:
        self.store._require_connection()

    @staticmethod
    def _validate_limit(limit: int) -> None:
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
