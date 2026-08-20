"""Run the Knowledge Engineering Agent against canonical metadata."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .agent import KnowledgeEngineeringAgent, load_canonical, write_enrichment
from .gemini_reviewer import GeminiArtifactReviewer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PATH = PROJECT_ROOT / "output" / "knowledge" / "canonical_metadata.json"
ENRICHMENT_PATH = PROJECT_ROOT / "output" / "knowledge" / "knowledge_enrichment.json"


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader so python-dotenv is not required."""
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
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Skip Gemini and run the deterministic baseline only.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Review only the first N artifacts with Gemini (0 = all). Useful for testing free-tier limits.",
    )
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
            canonical[key] = [
                item for item in canonical.get(key, []) if item.get("artifact_id") in allowed
            ]

    reviewer = None
    if not args.deterministic:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if api_key:
            reviewer = GeminiArtifactReviewer(api_key=api_key)
        else:
            print("GEMINI_API_KEY not found; using deterministic baseline.")

    result = KnowledgeEngineeringAgent(reviewer=reviewer).run(canonical)
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
