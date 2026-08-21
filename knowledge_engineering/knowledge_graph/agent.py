"""Knowledge graph orchestration."""

from __future__ import annotations

from typing import Any, Dict

from .builder import KnowledgeGraphBuilder


class KnowledgeGraphAgent:
    """Build a provenance-preserving graph from canonical and validated data."""

    VERSION = "0.1.0"

    def __init__(self, builder: KnowledgeGraphBuilder | None = None) -> None:
        self.builder = builder or KnowledgeGraphBuilder()

    def run(self, canonical: Dict[str, Any], relationship_discovery: Dict[str, Any] | None = None) -> Dict[str, Any]:
        graph = self.builder.build(canonical, relationship_discovery)
        graph["agent"] = {
            "name": "knowledge_graph_agent",
            "version": self.VERSION,
            "mode": "in_memory_networkx" if graph.get("networkx") is not None else "portable_json",
        }
        graph.pop("networkx", None)
        return graph
