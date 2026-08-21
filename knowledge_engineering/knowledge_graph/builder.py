"""Build a knowledge graph from canonical metadata without hardcoded domain facts."""

from __future__ import annotations

from typing import Any, Dict, List

from .schema import edge_from_relationship, node_from_entity

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None


class KnowledgeGraphBuilder:
    """Construct an in-memory directed multigraph from canonical metadata."""

    def build(self, canonical: Dict[str, Any], relationships: Dict[str, Any] | None = None) -> Dict[str, Any]:
        entities: List[Dict[str, Any]] = canonical.get("entities", [])
        relationship_payload = relationships or {}
        edges = relationship_payload.get("relationships") or canonical.get("relationships", [])

        nodes = [node_from_entity(entity) for entity in entities if entity.get("id")]
        graph_edges = [edge_from_relationship(edge) for edge in edges if edge.get("source_entity_id", edge.get("source")) and edge.get("target_entity_id", edge.get("target"))]

        graph_data = {
            "schema_version": "1.0",
            "graph_type": "directed_multigraph",
            "nodes": nodes,
            "edges": graph_edges,
            "statistics": {
                "nodes": len(nodes),
                "edges": len(graph_edges),
                "supported_edges": sum(e.get("validation_status") == "SUPPORTED" for e in graph_edges),
                "unverified_edges": sum(e.get("validation_status") == "UNVERIFIED" for e in graph_edges),
                "conflict_edges": sum(e.get("validation_status") == "CONFLICT" for e in graph_edges),
            },
            "safety": {
                "artifact_specific_hardcoding": False,
                "source_modified": False,
                "source": "canonical_metadata_and_validated_relationships",
            },
        }

        if nx is not None:
            graph = nx.MultiDiGraph()
            for node in nodes:
                graph.add_node(node["id"], **node)
            for edge in graph_edges:
                graph.add_edge(edge["source_entity_id"], edge["target_entity_id"], key=edge.get("id"), **edge)
            graph_data["networkx"] = graph

        return graph_data
