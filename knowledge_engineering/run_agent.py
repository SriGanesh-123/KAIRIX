"""Run the Knowledge Engineering Agent against canonical metadata."""

from __future__ import annotations

from pathlib import Path

from .agent import KnowledgeEngineeringAgent, load_canonical, write_enrichment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PATH = PROJECT_ROOT / "output" / "knowledge" / "canonical_metadata.json"
ENRICHMENT_PATH = PROJECT_ROOT / "output" / "knowledge" / "knowledge_enrichment.json"


def main() -> None:
    if not CANONICAL_PATH.exists():
        raise SystemExit(f"Canonical metadata not found: {CANONICAL_PATH}")

    canonical = load_canonical(CANONICAL_PATH)
    result = KnowledgeEngineeringAgent().run(canonical)
    write_enrichment(ENRICHMENT_PATH, result)

    summary = result["summary"]
    print("=" * 72)
    print("KNOWLEDGE ENGINEERING AGENT")
    print("=" * 72)
    print(f"Mode                  : {result['agent']['mode']}")
    print(f"Artifact profiles     : {summary['profiles']}")
    print(f"Artifact reviews      : {summary['reviews']}")
    print(f"Knowledge gaps        : {summary['knowledge_gaps']}")
    print(f"Deeper analysis       : {summary['deeper_analysis_required']}")
    print(f"Output                : {ENRICHMENT_PATH}")


if __name__ == "__main__":
    main()
