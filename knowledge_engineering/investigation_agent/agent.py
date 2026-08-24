"""Evidence-driven investigation agent for deeper RAG retrieval."""
from __future__ import annotations

import json
from typing import Any, Protocol

from ..llm.generator import LLMGenerator, generate_structured, parse_generation
from .contracts import (
    GroundedAnswer,
    InvestigationPlan,
    SufficiencyAssessment,
    VerificationResult,
    _normalize_string_list,
    extract_json_payload,
    parse_and_validate_answer,
    parse_and_validate_plan,
    parse_and_validate_sufficiency,
    parse_and_validate_verification,
)


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
    """Plan, assess, escalate, retrieve, verify, and synthesize evidence.

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
                if not key:
                    continue
                existing = merged.get(key)
                if existing is None:
                    merged[key] = item
                    continue
                current_score = float(existing.get("score", 0.0) or 0.0)
                candidate_score = float(item.get("score", 0.0) or 0.0)
                if candidate_score > current_score:
                    merged[key] = item
                elif candidate_score == current_score:
                    # Merge graph evidence if existing is missing it
                    if item.get("graph_evidence") and not existing.get("graph_evidence"):
                        existing["graph_evidence"] = item["graph_evidence"]
                        existing["graph_evidence_count"] = item.get("graph_evidence_count", 0)

        return sorted(
            merged.values(),
            key=lambda item: (
                float(item.get("score", 0.0) or 0.0),
                int(item.get("graph_evidence_count", 0) or 0),
            ),
            reverse=True,
        )

    @staticmethod
    def _has_graph_support(evidence: list[dict[str, Any]]) -> bool:
        return any(int(item.get("graph_evidence_count", 0) or 0) > 0 for item in evidence)

    @classmethod
    def _compact_evidence(cls, evidence: list[dict[str, Any]], max_items: int = 15) -> list[dict[str, Any]]:
        """Compact evidence representations to prevent token blowouts in prompt contexts."""
        compact: list[dict[str, Any]] = []
        for item in evidence[:max_items]:
            source_id = cls._evidence_key(item)
            text = str(item.get("text") or "").strip()
            if len(text) > 400:
                text = text[:400] + "..."

            # Summarize graph paths into compact readable strings
            graph_paths: list[str] = []
            for path in item.get("graph_evidence", [])[:4]:
                nodes = path.get("nodes", [])
                rels = path.get("relationships", [])
                if nodes and rels:
                    node_names = [str(n.get("name") or n.get("id")) for n in nodes]
                    rel_types = [str(r.get("relationship_type") or "RELATED_TO") for r in rels]
                    parts = [node_names[0]]
                    for i in range(min(len(rel_types), len(node_names) - 1)):
                        parts.append(f"-[{rel_types[i]}]→ {node_names[i+1]}")
                    graph_paths.append(" ".join(parts))

            entry: dict[str, Any] = {
                "evidence_id": source_id,
                "score": item.get("score"),
                "kind": item.get("kind"),
                "artifact_id": item.get("artifact_id"),
                "text": text,
            }
            if graph_paths:
                entry["graph_paths"] = graph_paths
            elif item.get("graph_evidence"):
                entry["graph_evidence"] = item.get("graph_evidence")

            compact.append(entry)
        return compact

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
            "Generate a small set of complementary retrieval queries that cover the request from distinct evidence perspectives.\n"
            "Prefer precise entity, relationship, dependency, rule, lineage, or calculation terms when the request implies them, but derive all terms from the request.\n"
            "Return JSON with exactly these keys:\n"
            "intent (string), objectives (array of strings), retrieval_queries (array of strings),\n"
            "evidence_requirements (array of strings), requested_output_format (string), constraints (array of strings).\n\n"
            f"USER REQUEST:\n{json.dumps(request, ensure_ascii=False, indent=2)}"
        )

    def _plan(self, request: dict[str, Any]) -> InvestigationPlan:
        prompt = self._build_plan_prompt(request)
        try:
            return generate_structured(
                self.planner,
                prompt,
                lambda content: parse_and_validate_plan(content, request["question"]),
                max_repair_retries=1,
            )
        except Exception:
            return InvestigationPlan(
                intent="Investigate the supplied request using available evidence.",
                objectives=[],
                retrieval_queries=[request["question"]],
                evidence_requirements=[],
                requested_output_format=str(request.get("output_format", "")),
                constraints=[],
            )

    @classmethod
    def _build_sufficiency_prompt(
        cls,
        query: str,
        plan: dict[str, Any],
        evidence: list[dict[str, Any]],
    ) -> str:
        compact = cls._compact_evidence(evidence)
        return (
            "You are the evidence-sufficiency component of a domain-neutral investigation agent.\n"
            "Decide whether the supplied evidence actually answers the user's requested intent.\n"
            "Do not treat the mere presence of graph evidence as sufficient.\n"
            "Require evidence that establishes the requested fact, connection, dependency, lineage, calculation, or other objective rather than merely related context.\n"
            "For other requests, judge against the stated objectives and evidence requirements.\n"
            "If important parts are missing, return sufficient=false and describe the knowledge gaps.\n"
            "Return JSON with exactly these keys: sufficient (boolean), knowledge_gaps (array of strings),\n"
            "follow_up_queries (array of strings). Follow-up queries must target missing evidence and remain domain-neutral.\n\n"
            f"USER QUESTION:\n{query}\n\n"
            f"INVESTIGATION PLAN:\n{json.dumps(plan, ensure_ascii=False, indent=2)}\n\n"
            f"RETRIEVED EVIDENCE:\n{json.dumps(compact, ensure_ascii=False, indent=2)}"
        )

    def _assess_sufficiency(
        self,
        query: str,
        plan: InvestigationPlan,
        evidence: list[dict[str, Any]],
    ) -> SufficiencyAssessment:
        if not evidence:
            return SufficiencyAssessment(
                sufficient=False,
                knowledge_gaps=["No initial evidence was retrieved."],
                follow_up_queries=list(plan.retrieval_queries),
            )
        prompt = self._build_sufficiency_prompt(query, plan.to_dict(), evidence)
        try:
            return generate_structured(
                self.planner,
                prompt,
                parse_and_validate_sufficiency,
                max_repair_retries=1,
            )
        except Exception:
            has_graph = self._has_graph_support(evidence)
            return SufficiencyAssessment(
                sufficient=bool(evidence) and has_graph,
                knowledge_gaps=[] if evidence and has_graph else [
                    "Initial retrieval did not provide sufficient graph-backed evidence."
                ],
                follow_up_queries=list(plan.retrieval_queries) if not has_graph else [],
            )

    @classmethod
    def _build_answer_prompt(
        cls,
        query: str,
        evidence: list[dict[str, Any]],
        plan: dict[str, Any],
        knowledge_gaps: list[str],
    ) -> str:
        compact = cls._compact_evidence(evidence)
        return (
            "You are the KAIRIX investigation answerer.\n"
            "Produce a precise, useful, evidence-grounded answer to the user's request.\n"
            "Answer ONLY from the supplied investigation evidence.\n"
            "Do not invent relationships, dependencies, business rules, formulas, fields, schemas, lineage, or missing facts.\n"
            "Start with the direct answer, then explain the supporting reasoning in a concise step-by-step form when the evidence permits.\n"
            "Name relevant artifacts, entities, rules, or other source concepts only when they are present in the evidence.\n"
            "Distinguish directly supported facts from cautious inference. Never present an inference as an established fact.\n"
            "For every important factual claim, cite one or more supporting evidence IDs in the evidence_ids output.\n"
            "If evidence is incomplete or conflicting, explicitly state what is established, what is uncertain, and what is missing.\n"
            "Do not repeat the same knowledge gap in different wording. Deduplicate gaps and keep them specific.\n"
            "If the evidence cannot establish the requested fact, say so clearly rather than filling the gap from general knowledge.\n"
            "Return JSON with exactly: answer (string), evidence_ids (array of strings), confidence (number 0.0-1.0),\n"
            "and knowledge_gaps (array of strings). Only cite evidence_ids present in the supplied evidence.\n\n"
            f"USER QUESTION:\n{query}\n\n"
            f"INVESTIGATION PLAN:\n{json.dumps(plan, ensure_ascii=False, indent=2)}\n\n"
            f"KNOWN KNOWLEDGE GAPS:\n{json.dumps(knowledge_gaps, ensure_ascii=False, indent=2)}\n\n"
            f"INVESTIGATION EVIDENCE:\n{json.dumps(compact, ensure_ascii=False, indent=2)}"
        )

    def _generate_answer(
        self,
        query: str,
        evidence: list[dict[str, Any]],
        plan: InvestigationPlan,
        knowledge_gaps: list[str],
        allowed: set[str],
    ) -> GroundedAnswer:
        if not evidence:
            return GroundedAnswer(
                answer="The investigation could not find any evidence matching the query in the knowledge base.",
                evidence_ids=[],
                confidence=0.0,
                knowledge_gaps=["No evidence was retrieved for the requested query."],
            )
        prompt = self._build_answer_prompt(query, evidence, plan.to_dict(), knowledge_gaps)
        try:
            return generate_structured(
                self.generator,
                prompt,
                lambda content: parse_and_validate_answer(content, allowed),
                max_repair_retries=1,
            )
        except Exception as exc:
            err_msg = str(exc)
            diagnostic = "The answer-generation step did not return a valid structured result."
            if "Rate limit" in err_msg or "rate_limit_exceeded" in err_msg:
                diagnostic = "Provider rate limit reached during answer generation."
            elif "Request too large" in err_msg:
                diagnostic = "Prompt context exceeded model request size limits."

            return GroundedAnswer(
                answer="The investigation could not produce a valid grounded answer from the supplied evidence.",
                evidence_ids=[],
                confidence=0.0,
                knowledge_gaps=[diagnostic],
            )

    @classmethod
    def _build_verification_prompt(
        cls,
        query: str,
        draft: dict[str, Any],
        evidence: list[dict[str, Any]],
    ) -> str:
        compact = cls._compact_evidence(evidence)
        return (
            "You are the final evidence verifier for a domain-neutral investigation agent.\n"
            "Review the draft answer against the supplied evidence only.\n"
            "Remove or rewrite unsupported claims. Preserve useful supported detail.\n"
            "Ensure every cited evidence ID exists and actually supports the answer.\n"
            "Do not add facts that are absent from the evidence.\n"
            "Lower confidence when the evidence is incomplete or only indirectly supports the answer.\n"
            "Deduplicate knowledge gaps.\n"
            "Return JSON with exactly: answer (string), evidence_ids (array of strings), confidence (number 0.0-1.0),\n"
            "knowledge_gaps (array of strings), verified (boolean).\n\n"
            f"USER QUESTION:\n{query}\n\n"
            f"DRAFT ANSWER:\n{json.dumps(draft, ensure_ascii=False, indent=2)}\n\n"
            f"SUPPLIED EVIDENCE:\n{json.dumps(compact, ensure_ascii=False, indent=2)}"
        )

    def _verify_answer(
        self,
        query: str,
        draft: GroundedAnswer,
        evidence: list[dict[str, Any]],
        allowed: set[str],
    ) -> VerificationResult:
        if not evidence or not draft.answer or (draft.confidence == 0.0 and not draft.evidence_ids):
            return VerificationResult(
                verified=False,
                answer=draft.answer,
                evidence_ids=draft.evidence_ids,
                confidence=draft.confidence,
                knowledge_gaps=draft.knowledge_gaps,
            )
        prompt = self._build_verification_prompt(query, draft.to_dict(), evidence)
        try:
            return generate_structured(
                self.planner,
                prompt,
                lambda content: parse_and_validate_verification(content, draft, allowed),
                max_repair_retries=1,
            )
        except Exception:
            return VerificationResult(
                verified=False,
                answer=draft.answer,
                evidence_ids=draft.evidence_ids,
                confidence=draft.confidence,
                knowledge_gaps=draft.knowledge_gaps,
            )

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
        investigation_gaps = list(assessment.knowledge_gaps)
        follow_up_queries = assessment.follow_up_queries or plan.retrieval_queries

        if not assessment.sufficient:
            follow_up_hops = min(max_graph_hops, initial_graph_hops + 1)
            seen_queries = {base}
            for follow_up in follow_up_queries:
                follow_up = follow_up.strip()
                if not follow_up or follow_up in seen_queries:
                    continue
                seen_queries.add(follow_up)
                try:
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
                except Exception:
                    continue

        evidence = self._merge_evidence(batches)
        allowed = {self._evidence_key(item) for item in evidence}

        generated = self._generate_answer(base, evidence, plan, investigation_gaps, allowed)
        verified = self._verify_answer(base, generated, evidence, allowed)

        evidence_by_id = {self._evidence_key(item): item for item in evidence}
        combined_gaps = _normalize_string_list(investigation_gaps + verified.knowledge_gaps)
        if not verified.evidence_ids:
            combined_gaps.append("No retrieved evidence was cited by the answer generator or verifier.")

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
            "plan": plan.to_dict(),
            "answer": verified.answer,
            "evidence_ids": verified.evidence_ids,
            "evidence": [evidence_by_id[item] for item in verified.evidence_ids if item in evidence_by_id],
            "retrieved_count": len(evidence),
            "investigation_triggered": len(steps) > 1,
            "steps": steps,
            "trace_references": trace_references,
            "knowledge_gaps": combined_gaps,
            "confidence": verified.confidence,
            "confidence_level": self._confidence_level(verified.confidence),
            "answer_verified": verified.verified,
            "provider": getattr(self.generator, "provider", "unknown"),
            "model": getattr(self.generator, "model", "unknown"),
        }
