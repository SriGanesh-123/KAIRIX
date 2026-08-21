"""Run the Knowledge Engineering Agent against canonical metadata."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .agent import KnowledgeEngineeringAgent, load_canonical
from .enrichment_store import write_merged
from .llm.config import LLMConfig
from .llm.factory import create_reviewer
from .summary import write_artifact_summaries

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PATH = PROJECT_ROOT / "output" / "knowledge" / "canonical_metadata.json"
ENRICHMENT_PATH = PROJECT_ROOT / "output" / "knowledge" / "knowledge_enrichment.json"
SUMMARY_DIR = PROJECT_ROOT / "output" / "knowledge" / "summaries"


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Knowledge Engineering Agent")
    parser.add_argument("--deterministic", action="store_true", help="Skip LLM review and run deterministic baseline only.")
    parser.add_argument("--execute-parsers", action="store_true", help="Execute selected existing parser modules.")
    parser.add_argument("--limit", type=int, default=0, help="Review only the first N artifacts (0 = all).")
    args = parser.parse_args()
    if not CANONICAL_PATH.exists():
        raise SystemExit(f"Canonical metadata not found: {CANONICAL_PATH}")
    _load_dotenv(PROJECT_ROOT / ".env")
    canonical = load_canonical(CANONICAL_PATH)
    if args.limit < 0:
        raise SystemExit("--limit must be >= 0")
    if args.limit:
        canonical = dict(canonical)
        canonical["artifacts"] = canonical.get("artifacts", [])[: args.limit]
        allowed = {item["id"] for item in canonical["artifacts"]}
        for key in ("entities", "relationships", "evidence", "business_rules"):
            canonical[key] = [item for item in canonical.get(key, []) if item.get("artifact_id") in allowed]

    reviewer = None
    llm_label = "deterministic"
    if not args.deterministic:
        config = LLMConfig.from_env()
        if config is None:
            raise SystemExit("LLM configuration missing. Set LLM_PROVIDER, LLM_MODEL and LLM_API_KEY, or use --deterministic.")
        reviewer = create_reviewer(config)
        llm_label = f"{config.provider}/{config.model}"
        print(f"LLM selected: {llm_label}", flush=True)

    result = KnowledgeEngineeringAgent(reviewer=reviewer, execute_parsers=args.execute_parsers, project_root=PROJECT_ROOT).run(canonical)
    merged = write_merged(ENRICHMENT_PATH, result)
    summary_paths = write_artifact_summaries(merged, SUMMARY_DIR)
    summary = merged["summary"]
    print("=" * 72)
    print("KNOWLEDGE ENGINEERING AGENT")
    print("=" * 72)
    print(f"Mode                  : {merged['agent']['mode']}")
    print(f"LLM                   : {llm_label}")
    print(f"Artifact profiles     : {summary['profiles']}")
    print(f"Artifact reviews      : {summary['reviews']}")
    print(f"Parser executions     : {summary['parser_executions']}")
    print(f"Parser successes      : {summary['parser_executions_successful']}")
    print(f"Parser failures       : {summary['parser_executions_failed']}")
    print(f"Knowledge gaps        : {summary['knowledge_gaps']}")
    print(f"Deeper analysis       : {summary['deeper_analysis_required']}")
    print(f"Artifact summaries    : {len(summary_paths)}")
    print(f"Summary directory     : {SUMMARY_DIR}")
    print(f"Output                : {ENRICHMENT_PATH}")


if __name__ == "__main__":
    main()
