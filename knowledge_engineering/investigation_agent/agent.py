"""Evidence-driven investigation agent for deeper RAG retrieval."""
from __future__ import annotations

import json
from typing import Any, Protocol

from ..llm.generator import LLMGenerator, parse_generation


class InvestigationRetriever(Protocol):
    """Retriever contract required by the investigation agent."""

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        graph_hops: int = 1,
    ) -> list[dict[str, Any]]:
        ...


class InvestigationPlanner(Protocol):
    """Planner contract for intent, sufficiency, and follow-up generation."""

    provider: str
    model: str

    def generate(self, prompt: str) -> str:
        ...


class InvestigationAgent:
    """Plan, assess, escalate, retrieve, and synthesize evidence.

    The agent is domain-neutral. Domain terminology, document types, output
    formats, and retrieval objectives come from the request and configured LLM,
    not from hard-coded artifact names.
    """

    def __init__(
        self,
        *,
        retriever: InvestigationRetriever,
        generator: LLMGenerator,
        planner: InvestigationPlanner | None = None,
        confidence_thresholds: tuple[float, float] = (0.45, 0.75),
    ) -> None:
        low, high = confidence_thresholds
        if not 0.0 <= low < high <= 1.0:
            raise ValueError("confidence thresholds must satisfy 0 <= low < high <= 1")
        self.retriever = retriever
        self.generator = generator
        self.planner = planner or generator
        self.confidence_thresholds = (low, high)

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
    def _has_graph_support(evidence: list[dict[str, Any]]) -> bool:
        return any(int(item.get("graph_evidence_count", 0) or 0) > 0 for item in evidence)

    @staticmethod
    def _normalize_request(request: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(request, str):
            question = request.strip()
            if not question:
                raise ValueError("query must not be empty")
            return {"question": question}
        if not isinstance(request, dict):
            raise TypeError("request must be a string or object")
        question = str(request.get("question") or request.get("query") or "").strip()
        if not question:
            raise ValueError("request must contain a non-empty question")
        normalized = dict(request)
        normalized["question"] = question
        return normalized

    @staticmethod
    def _build_plan_prompt(request: dict[str, Any]) -> str:
        return (
            "You are the KAIRIX investigation planning component.\n"
            "Understand the supplied user request and create a domain-neutral investigation plan.\n"
            "Do not invent facts about the underlying system.\n"
            "Preserve any document type, output format, constraints, or scope explicitly supplied.\n"
            "Generate retrieval queries that are useful for finding supporting evidence.\n"
            "Return JSON with exactly these keys:\n"
            "intent (string), objectives (array of strings), retrieval_queries (array of strings),\n"
            "evidence_requirements (array of strings), requested_output_format (string), constraints (array of strings).\n\n"
            f"USER REQUEST:\n{json.dumps(request, ensure_ascii=False, indent=2)}"
        )

    @staticmethod
    def _parse_plan(content: str, original_question: str) -> dict[str, Any]:
        value = json.loads(content)
        if not isinstance(value, dict):
            raise ValueError("investigation plan must be a JSON object")

        def strings(name: str) -> list[str]:
            raw = value.get(name, [])
            if raw is None:
                return []
            if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
                raise ValueError(f"investigation plan field {name!r} must be a list of strings")
            return [item.strip() for item in raw if item.strip()]

        intent = value.get("intent", "").strip() if isinstance(value.get("intent", ""), str) else ""
        output_format = value.get("requested_output_format", "")
        if not isinstance(output_format, str):
            output_format = str(output_format)
        queries = strings("retrieval_queries")
        if not queries:
            queries = [original_question]
        return {
            "intent": intent or "Investigate the supplied request using available evidence.",
            "objectives": strings("objectives"),
            "retrieval_queries": queries,
            "evidence_requirements": strings("evidence_requirements"),
            "requested_output_format": output_format.strip(),
            "constraints": strings("constraints"),
        }

    def _plan(self, request: dict[str, Any]) -> dict[str, Any]:
        try:
            return self._parse_plan(
                self.planner.generate(self._build_plan_prompt(request)),
                request["question"],
            )
        except Exception:
            return {
                "intent": "Investigate the supplied request using available evidence.",
                "objectives": [],
                "retrieval_queries": [request["question"]],
                "evidence_requirements": [],
                "requested_output_format": str(request.get("output_format", "")),
                "constraints": [],
            }

    @staticmethod
    def _build_sufficiency_prompt(
        query: str,
        plan: dict[str, Any],
        evidence: list[dict[str, Any]],
    ) -> str:
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
            "You are the evidence-sufficiency component of a domain-neutral investigation agent.\n"
            "Decide whether the supplied evidence actually answers the user's requested intent.\n"
            "Do not treat the mere presence of graph evidence as sufficient.\n"
            "For relationship or dependency questions, require evidence that establishes the requested connection, not merely that entities share an artifact.\n"
            "For other requests, judge against the stated objectives and evidence requirements.\n"
            "If important parts are missing, return sufficient=false and describe the knowledge gaps.\n"
            "Return JSON with exactly these keys: sufficient (boolean), knowledge_gaps (array of strings),\n"
            "follow_up_queries (array of strings). Follow-up queries must target missing evidence and remain domain-neutral.\n\n"
            f"USER QUESTION:\n{query}\n\n"
            f"INVESTIGATION PLAN:\n{json.dumps(plan, ensure_ascii=False, indent=2)}\n\n"
            f"RETRIEVED EVIDENCE:\n{json.dumps(compact, ensure_ascii=False, indent=2)}"
        )

    @staticmethod
    def _parse_sufficiency(content: str) -> dict[str, Any]:
        value = json.loads(content)
        if not isinstance(value, dict):
            raise ValueError("evidence sufficiency response must be a JSON object")
        sufficient = value.get("sufficient")
        if not isinstance(sufficient, bool):
            raise ValueError("sufficient must be a boolean")
        gaps = value.get("knowledge_gaps", [])
        if not isinstance(gaps, list):
            gaps = [str(gaps)]
        queries = value.get("follow_up_queries", [])
        if not isinstance(queries, list):
            queries = [str(queries)]
        return {
            "sufficient": sufficient,
            "knowledge_gaps": [str(item).strip() for item in gaps if str(item).strip()],
            "follow_up_queries": [str(item).strip() for item in queries if str(item).strip()],
        }

    def _assess_sufficiency(
        self,
        query: str,
        plan: dict[str, Any],
        evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:
        try:
            return self._parse_sufficiency(
                self.planner.generate(self._build_sufficiency_prompt(query, plan, evidence))
            )
        except Exception:
            # Conservative fallback: graph-backed evidence can stop escalation only
            # when it exists; otherwise continue investigation.
            return {
                "sufficient": bool(evidence) and self._has_graph_support(evidence),
                "knowledge_gaps": [] if evidence and self._has_graph_support(evidence) else [
                    "Initial retrieval did not provide sufficient graph-backed evidence."
                ],
                "follow_up_queries": [],
            }

    @staticmethod
    def _build_answer_prompt(
        query: str,
        evidence: list[dict[str, Any]],
        plan: dict[str, Any],
        knowledge_gaps: list[str],
    ) -> str:
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
            "Do not invent relationships, dependencies, business rules, or missing facts.\n"
            "If evidence remains insufficient, explicitly say that the requested fact cannot be established.\n"
            "Return JSON with exactly: answer (string), evidence_ids (array of strings), confidence (number 0.0-1.0),\n"
            "and knowledge_gaps (array of strings). Only cite evidence_ids present in the supplied evidence.\n\n"
            f"USER QUESTION:\n{query}\n\n"
            f"INVESTIGATION PLAN:\n{json.dumps(plan, ensure_ascii=False, indent=2)}\n\n"
            f"KNOWN KNOWLEDGE GAPS:\n{json.dumps(knowledge_gaps, ensure_ascii=False, indent=2)}\n\n"
            f"INVESTIGATION EVIDENCE:\n{json.dumps(compact, ensure_ascii=False, indent=2)}"
        )

    @classmethod
    def _parse_answer(cls, content: str, allowed: set[str]) -> dict[str, Any]:
        generated = parse_generation(content)
        value = json.loads(content)
        raw_confidence = value.get("confidence", 0.0)
        try:
            confidence = max(0.0, min(1.0, float(raw_confidence)))
        except (TypeError, ValueError):
            confidence = 0.0
        gaps = value.get("knowledge_gaps", [])
        if not isinstance(gaps, list):
            gaps = [str(gaps)]
        gaps = [str(item) for item in gaps]
        cited = [item for item in generated["evidence_ids"] if item in allowed]
        return {
            "answer": generated["answer"],
            "evidence_ids": cited,
            "confidence": confidence,
            "knowledge_gaps": gaps,
        }

    def _confidence_level(self, confidence: float) -> str:
        low, high = self.confidence_thresholds
        if confidence < low:
            return "LOW"
        if confidence < high:
            return "MEDIUM"
        return "HIGH"

    def investigate(
        self,
        query: str | dict[str, Any],
        *,
        limit: int = 5,
        initial_graph_hops: int = 1,
        max_graph_hops: int = 3,
    ) -> dict[str, Any]:
        request = self._normalize_request(query)
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        if not 1 <= initial_graph_hops <= max_graph_hops <= 4:
            raise ValueError("graph hop range must be between 1 and 4")

        plan = self._plan(request)
        base = request["question"]
        batches: list[list[dict[str, Any]]] = []
        steps: list[dict[str, Any]] = []

        first = self.retriever.search(base, limit=limit, graph_hops=initial_graph_hops)
        batches.append(first)
        steps.append(
            {
                "query": base,
                "graph_hops": initial_graph_hops,
                "retrieved_count": len(first),
                "evidence_ids": [self._evidence_key(item) for item in first],
            }
        )

        assessment = self._assess_sufficiency(base, plan, first)
        investigation_gaps = list(assessment["knowledge_gaps"])
        follow_up_queries = assessment["follow_up_queries"] or plan["retrieval_queries"]

        if not assessment["sufficient"]:
            follow_up_hops = min(max_graph_hops, initial_graph_hops + 1)
            seen_queries = {base}
            for follow_up in follow_up_queries:
                follow_up = follow_up.strip()
                if not follow_up or follow_up in seen_queries:
                    continue
                seen_queries.add(follow_up)
                evidence = self.retriever.search(
                    follow_up,
                    limit=limit,
                    graph_hops=follow_up_hops,
                )
                batches.append(evidence)
                steps.append(
                    {
                        "query": follow_up,
                        "graph_hops": follow_up_hops,
                        "retrieved_count": len(evidence),
                        "evidence_ids": [self._evidence_key(item) for item in evidence],
                    }
                )

        evidence = self._merge_evidence(batches)
        allowed = {self._evidence_key(item) for item in evidence}
        try:
            generated = self._parse_answer(
                self.generator.generate(
                    self._build_answer_prompt(base, evidence, plan, investigation_gaps)
                ),
                allowed,
            )
        except Exception:
            generated = {
                "answer": "The investigation could not produce a valid grounded answer from the supplied evidence.",
                "evidence_ids": [],
                "confidence": 0.0,
                "knowledge_gaps": ["The answer-generation step did not return a valid structured result."],
            }

        evidence_by_id = {self._evidence_key(item): item for item in evidence}
        combined_gaps = list(dict.fromkeys(investigation_gaps + generated["knowledge_gaps"]))
        if not generated["evidence_ids"]:
            combined_gaps.append("No retrieved evidence was cited by the answer generator.")

        trace_references = [
            {
                "step": index,
                "query": step["query"],
                "graph_hops": step["graph_hops"],
                "evidence_ids": step["evidence_ids"],
            }
            for index, step in enumerate(steps, start=1)
        ]

        return {
            "query": base,
            "request": request,
            "plan": plan,
            "answer": generated["answer"],
            "evidence_ids": generated["evidence_ids"],
            "evidence": [evidence_by_id[item] for item in generated["evidence_ids"]],
            "retrieved_count": len(evidence),
            "investigation_triggered": len(steps) > 1,
            "steps": steps,
            "trace_references": trace_references,
            "knowledge_gaps": combined_gaps,
            "confidence": generated["confidence"],
            "confidence_level": self._confidence_level(generated["confidence"]),
            "provider": getattr(self.generator, "provider", "unknown"),
            "model": getattr(self.generator, "model", "unknown"),
        }
