"""Hybrid semantic + graph retrieval for KAIRIX."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from .env import load_environment
from .knowledge_graph import KnowledgeGraphQuery
from .vector_retrieval import VectorRetriever


class HybridRetriever:
    """Combine Qdrant semantic evidence with Neo4j graph evidence."""

    _STOPWORDS = {
        "about", "after", "also", "does", "from", "have", "into", "that",
        "their", "there", "these", "this", "what", "when", "where", "which",
        "with", "would", "could", "should", "how", "why", "who", "are", "and",
        "the", "for", "between", "relationship", "relationships", "depend",
        "depends", "dependency", "dependencies",
    }

    def __init__(
        self,
        *,
        vector_retriever: VectorRetriever | None = None,
        graph_query: KnowledgeGraphQuery | None = None,
    ) -> None:
        load_environment()
        self.vector = vector_retriever or VectorRetriever()
        self.graph = graph_query or KnowledgeGraphQuery()
        self._owns_vector = vector_retriever is None
        self._owns_graph = graph_query is None

    def connect(self) -> None:
        self.vector.connect()
        self.graph.store.connect()

    def close(self) -> None:
        if self._owns_vector:
            self.vector.close()
        if self._owns_graph:
            self.graph.store.close()

    @classmethod
    def _graph_terms(cls, query: str) -> List[str]:
        """Extract a small set of useful graph lookup terms from a question."""
        terms = re.findall(r"[A-Za-z0-9_][A-Za-z0-9_.-]{2,}", query.lower())
        unique: List[str] = []
        for term in terms:
            if term in cls._STOPWORDS or term in unique:
                continue
            unique.append(term)
        return unique[:6]

    @classmethod
    def _semantic_queries(cls, query: str) -> List[str]:
        """Build bounded, deterministic query variants to improve semantic recall."""
        base = query.strip()
        variants: List[str] = [base]
        lowered = base.lower()

        if any(word in lowered for word in ("relationship", "relationships", "depend", "dependency")):
            terms = cls._graph_terms(base)
            if terms:
                variants.append(" ".join(terms) + " relationship dependency")

        if any(word in lowered for word in ("calculate", "calculated", "calculation", "formula", "computed")):
            terms = cls._graph_terms(base)
            if terms:
                variants.append(" ".join(terms) + " calculation formula business rule")

        unique: List[str] = []
        for variant in variants:
            normalized = " ".join(variant.split())
            if normalized and normalized not in unique:
                unique.append(normalized)
        return unique[:3]

    @staticmethod
    def _graph_text(node: Dict[str, Any], paths: List[Dict[str, Any]]) -> str:
        """Create compact evidence text for a graph-discovered node/path."""
        node_name = node.get("name") or node.get("id") or "unknown"
        node_type = node.get("type") or "UNKNOWN"
        lines = [f"Graph node: {node_name}. Type: {node_type}."]
        for path in paths[:5]:
            nodes = path.get("nodes", [])
            relationships = path.get("relationships", [])
            if not nodes or not relationships:
                continue
            node_names = [str(item.get("name") or item.get("id")) for item in nodes]
            rel_names = [str(item.get("relationship_type") or "RELATED_TO") for item in relationships]
            lines.append("Path: " + " -[".join([node_names[0]] + [f"{rel}]→ {name}" for rel, name in zip(rel_names, node_names[1:])]))
        return "\n".join(lines)

    def _graph_candidates(
        self,
        query: str,
        *,
        graph_hops: int,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Find graph nodes mentioned by the query and expose their paths as evidence."""
        candidates: Dict[str, Dict[str, Any]] = {}
        for term in self._graph_terms(query):
            try:
                nodes = self.graph.search_nodes(term, limit=max(5, limit))
            except Exception:
                continue
            for node in nodes:
                source_id = str(node.get("id") or "")
                if not source_id:
                    continue
                try:
                    paths = self.graph.traverse(source_id, hops=graph_hops, limit=20)
                except Exception:
                    paths = []
                if not paths:
                    continue
                candidates[source_id] = {
                    "id": f"graph-{source_id}",
                    "score": 0.55,
                    "text": self._graph_text(node, paths),
                    "kind": "graph_entity",
                    "source_id": source_id,
                    "artifact_id": node.get("artifact_id"),
                    "metadata": {
                        "name": node.get("name"),
                        "type": node.get("type"),
                        "retrieval": "graph_term_match",
                    },
                    "graph_evidence": paths,
                    "graph_evidence_count": len(paths),
                }
        return list(candidates.values())[: max(limit, 1)]

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        graph_hops: int = 1,
    ) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        if not 1 <= graph_hops <= 4:
            raise ValueError("graph_hops must be between 1 and 4")

        # Search the original question plus a small number of deterministic
        # intent-specific variants. This improves recall without requiring an
        # extra LLM call and keeps the search bounded.
        semantic_queries = self._semantic_queries(query)
        candidate_limit = min(100, max(limit * 3, limit))
        vector_batches: List[List[Dict[str, Any]]] = []
        for semantic_query in semantic_queries:
            try:
                vector_batches.append(
                    self.vector.search(semantic_query, limit=candidate_limit)
                )
            except Exception:
                continue

        results: Dict[str, Dict[str, Any]] = {}
        for vector_results in vector_batches:
            for result in vector_results:
                source_id = result.get("source_id")
                key = str(source_id or result.get("id") or "")
                if not key:
                    continue
                graph_evidence: List[Dict[str, Any]] = []
                try:
                    graph_evidence = self.graph.traverse(
                        str(source_id), hops=graph_hops, limit=20
                    ) if source_id else []
                except Exception:
                    graph_evidence = []

                vector_score = float(result.get("score", 0.0))
                existing = results.get(key)
                if existing is None or vector_score > float(existing.get("score", 0.0)):
                    results[key] = {
                        "id": result.get("id"),
                        "score": vector_score,
                        "text": result.get("text"),
                        "kind": result.get("kind"),
                        "source_id": source_id,
                        "artifact_id": result.get("artifact_id"),
                        "metadata": result.get("metadata", {}),
                        "graph_evidence": graph_evidence,
                        "graph_evidence_count": len(graph_evidence),
                    }
                elif graph_evidence and not existing.get("graph_evidence"):
                    existing["graph_evidence"] = graph_evidence
                    existing["graph_evidence_count"] = len(graph_evidence)

        for candidate in self._graph_candidates(query.strip(), graph_hops=graph_hops, limit=limit):
            key = str(candidate.get("source_id") or candidate.get("id") or "")
            if key and key not in results:
                results[key] = candidate

        ranked = list(results.values())
        ranked.sort(
            key=lambda item: (
                float(item.get("score", 0.0))
                + min(0.10, 0.02 * int(item.get("graph_evidence_count", 0) or 0)),
                int(item.get("graph_evidence_count", 0) or 0),
            ),
            reverse=True,
        )
        return ranked[:limit]
