"""Evidence-driven investigation loop for KAIRIX."""
from __future__ import annotations

import json
from typing import Any, Protocol

from .llm.generator import LLMGenerator, parse_generation


class InvestigationRetriever(Protocol):
    def search(self, query: str, *, limit: int = 5, graph_hops: int = 1) -> list[dict[str, Any]]: ...


class InvestigationAgent:
    """Escalate retrieval when the first evidence set is insufficient."""

    def __init__(self, *, retriever: InvestigationRetriever, generator: LLMGenerator) -> None:
        self.retriever = retriever
        self.generator = generator

    @staticmethod
    def _evidence_key(item: dict[str, Any]) -> str:
        return str(item.get("source_id") or item.get("id"))

    @classmethod
    def _merge_evidence(cls, batches: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for batch in batches:
            for item in batch:
                key = cls._evidence_key(item)
                if key and key not in merged:
                    merged[key] = item
        return list(merged.values())

    @staticmethod
    def _needs_investigation(evidence: list[dict[str, Any]]) -> bool:
        if not evidence:
            return True
        return not any(item.get("graph_evidence_count", 0) > 0 for item in evidence)

    @staticmethod
    def _build_follow_up_queries(query: str) -> list[str]:
        base = query.strip()
        return [
            f"{base} direct relationships dependencies",
            f"{base} cross-artifact relationships lineage",
        ]

    @staticmethod
    def _build_prompt(query: str, evidence: list[dict[str, Any]]) -> str:
        compact = [
            {
                "evidence_id": str(item.get("source_id") or item.get("id")),
                "score": item.get("score"),
                "kind": item.get("kind"),
                "artifact_id": item.get("artifact_id"),
                "text": item.get("text"),
                "metadata": item.get("metadata", {}),
                "graph_evidence": item.get("graph_evidence", []),
            }
            for item in evidence
        ]
        return (
            "You are the KAIRIX investigation answerer.\n"
            "Answer ONLY from the supplied investigation evidence.\n"
            "Do not invent relationships, dependencies, or business rules.\n"
            "If evidence remains insufficient, explicitly say that the relationship cannot be established.\n"
            "Return JSON with exactly: answer (string) and evidence_ids (array of strings).\n"
            "Only cite evidence_ids present in the supplied evidence.\n\n"
            f"USER QUESTION:\n{query}\n\n"
            f"INVESTIGATION EVIDENCE:\n{json.dumps(compact, ensure_ascii=False, indent=2)}"
        )

    def investigate(
        self,
        query: str,
        *,
        limit: int = 5,
        initial_graph_hops: int = 1,
        max_graph_hops: int = 3,
    ) -> dict[str, Any]:
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        if not 1 <= initial_graph_hops <= max_graph_hops <= 4:
            raise ValueError("graph hop range must be between 1 and 4")

        batches: list[list[dict[str, Any]]] = []
        steps: list[dict[str, Any]] = []
        base = query.strip()
        first = self.retriever.search(base, limit=limit, graph_hops=initial_graph_hops)
        batches.append(first)
        steps.append({"query": base, "graph_hops": initial_graph_hops, "retrieved_count": len(first)})

        if self._needs_investigation(first):
            follow_up_hops = min(max_graph_hops, initial_graph_hops + 1)
            for follow_up in self._build_follow_up_queries(base):
                evidence = self.retriever.search(follow_up, limit=limit, graph_hops=follow_up_hops)
                batches.append(evidence)
                steps.append({"query": follow_up, "graph_hops": follow_up_hops, "retrieved_count": len(evidence)})

        evidence = self._merge_evidence(batches)
        generated = parse_generation(self.generator.generate(self._build_prompt(base, evidence)))
        allowed = {self._evidence_key(item) for item in evidence}
        cited = [item for item in generated["evidence_ids"] if item in allowed]
        evidence_by_id = {self._evidence_key(item): item for item in evidence}
        return {
            "query": base,
            "answer": generated["answer"],
            "evidence_ids": cited,
            "evidence": [evidence_by_id[item] for item in cited],
            "retrieved_count": len(evidence),
            "investigation_triggered": len(steps) > 1,
            "steps": steps,
            "provider": getattr(self.generator, "provider", "unknown"),
            "model": getattr(self.generator, "model", "unknown"),
        }
