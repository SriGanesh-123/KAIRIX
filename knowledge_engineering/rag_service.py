"""Grounded RAG service combining hybrid retrieval with the configured LLM."""
from __future__ import annotations

import json
from typing import Any

from .env import load_environment
from .hybrid_retrieval import HybridRetriever
from .llm.config import LLMConfig
from .llm.generator import create_generator, parse_generation


class RAGService:
    """Answer user questions using only retrieved KAIRIX evidence."""

    def __init__(
        self,
        *,
        retriever: HybridRetriever | None = None,
        generator: Any | None = None,
    ) -> None:
        load_environment()
        self.retriever = retriever or HybridRetriever()
        self._owns_retriever = retriever is None
        self.generator = generator
        if self.generator is None:
            config = LLMConfig.from_env()
            if config is None:
                raise RuntimeError("LLM configuration missing. Set LLM_PROVIDER, LLM_MODEL and LLM_API_KEY.")
            self.generator = create_generator(config)

    def connect(self) -> None:
        self.retriever.connect()

    def close(self) -> None:
        if self._owns_retriever:
            self.retriever.close()

    @staticmethod
    def _build_prompt(query: str, evidence: list[dict[str, Any]]) -> str:
        compact: list[dict[str, Any]] = []
        for item in evidence:
            compact.append(
                {
                    "evidence_id": str(item.get("source_id") or item.get("id")),
                    "score": item.get("score"),
                    "kind": item.get("kind"),
                    "artifact_id": item.get("artifact_id"),
                    "text": item.get("text"),
                    "metadata": item.get("metadata", {}),
                    "graph_evidence": item.get("graph_evidence", []),
                }
            )
        return (
            "You are the KAIRIX grounded RAG answerer.\n"
            "Answer the user's question ONLY from the supplied evidence.\n"
            "Do not invent tables, relationships, business rules, or dependencies.\n"
            "Distinguish direct evidence from absence of evidence.\n"
            "If the supplied evidence does not establish the requested relationship, say so explicitly.\n"
            "When graph_evidence contains a path, use its relationship_type and node names to describe that path.\n"
            "Do not infer a direct relationship merely because two entities belong to the same artifact.\n"
            "Return JSON with exactly: answer (string) and evidence_ids (array of strings).\n"
            "Only include evidence_ids that appear in the supplied evidence.\n\n"
            f"USER QUESTION:\n{query.strip()}\n\n"
            f"RETRIEVED EVIDENCE:\n{json.dumps(compact, ensure_ascii=False, indent=2)}"
        )

    def ask(self, query: str, *, limit: int = 5, graph_hops: int = 1) -> dict[str, Any]:
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        evidence = self.retriever.search(query, limit=limit, graph_hops=graph_hops)
        prompt = self._build_prompt(query, evidence)
        generated = parse_generation(self.generator.generate(prompt))

        allowed = {str(item.get("source_id") or item.get("id")) for item in evidence}
        cited = [item for item in generated["evidence_ids"] if item in allowed]
        evidence_by_id = {
            str(item.get("source_id") or item.get("id")): item
            for item in evidence
        }
        return {
            "query": query.strip(),
            "answer": generated["answer"],
            "evidence_ids": cited,
            "evidence": [evidence_by_id[item] for item in cited],
            "retrieved_count": len(evidence),
            "provider": getattr(self.generator, "provider", "unknown"),
            "model": getattr(self.generator, "model", "unknown"),
        }
