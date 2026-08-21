"""Retry only artifact reviews that are marked LLM_REVIEW_PENDING.

This command deliberately does not rerun already successful LLM reviews. It reads
an existing knowledge_enrichment.json, finds pending artifact IDs, loads the
corresponding canonical artifacts/profiles, and replaces only those pending
reviews after a successful Gemini call.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .gemini_reviewer import GeminiArtifactReviewer
from .profile import build_artifact_profiles


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PATH = PROJECT_ROOT / "output" / "knowledge" / "canonical_metadata.json"
ENRICHMENT_PATH = PROJECT_ROOT / "output" / "knowledge" / "knowledge_enrichment.json"


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


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def retry_pending(*, max_artifacts: int = 0) -> int:
    _load_dotenv(PROJECT_ROOT / ".env")

    if not CANONICAL_PATH.exists():
        raise SystemExit(f"Canonical metadata not found: {CANONICAL_PATH}")
    if not ENRICHMENT_PATH.exists():
        raise SystemExit(f"Knowledge enrichment not found: {ENRICHMENT_PATH}")

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not found in environment/.env")

    canonical = _load(CANONICAL_PATH)
    enrichment = _load(ENRICHMENT_PATH)
    reviews = enrichment.get("artifact_reviews", [])
    pending = [item for item in reviews if item.get("status") == "LLM_REVIEW_PENDING"]

    if max_artifacts:
        pending = pending[:max_artifacts]

    print(f"Pending LLM reviews found: {len(pending)}")
    if not pending:
        print("Nothing to retry. Existing enrichment contains no LLM_REVIEW_PENDING artifacts.")
        return 0

    pending_ids = {item.get("artifact_id") for item in pending}
    artifacts = [item for item in canonical.get("artifacts", []) if item.get("id") in pending_ids]
    profiles = build_artifact_profiles({**canonical, "artifacts": artifacts})
    artifacts_by_id = {item["id"]: item for item in artifacts}
    profiles_by_id = {item["artifact_id"]: item for item in profiles}

    reviewer = GeminiArtifactReviewer(api_key=api_key)
    replaced = 0

    for old_review in pending:
        artifact_id = old_review.get("artifact_id")
        artifact = artifacts_by_id.get(artifact_id)
        profile = profiles_by_id.get(artifact_id)
        if not artifact or not profile:
            print(f"SKIP {artifact_id}: canonical artifact/profile not found")
            continue

        try:
            review = reviewer.review(artifact, profile)
        except Exception as exc:
            print(f"PENDING {artifact.get('file_name', artifact_id)}: {exc}")
            continue

        new_review = {
            "artifact_id": artifact_id,
            "mode": "llm",
            "status": "LLM_REVIEWED",
            "artifact_identification": old_review.get("artifact_identification", {}),
            "parser_selection": old_review.get("parser_selection", {}),
            "parser_execution": old_review.get("parser_execution", {}),
            **review,
        }

        for index, existing in enumerate(reviews):
            if existing.get("artifact_id") == artifact_id:
                reviews[index] = new_review
                replaced += 1
                break

        print(f"REVIEWED: {artifact.get('file_name', artifact_id)}")

    enrichment["artifact_reviews"] = reviews
    enrichment.setdefault("summary", {})["llm_reviews_completed"] = sum(
        item.get("status") == "LLM_REVIEWED" for item in reviews
    )
    enrichment["summary"]["llm_reviews_pending"] = sum(
        item.get("status") == "LLM_REVIEW_PENDING" for item in reviews
    )

    # Remove only the LLM-pending gap entries that have now been resolved.
    resolved_ids = {
        item.get("artifact_id")
        for item in reviews
        if item.get("status") == "LLM_REVIEWED"
    }
    enrichment["knowledge_gaps"] = [
        gap
        for gap in enrichment.get("knowledge_gaps", [])
        if not (
            gap.get("type") == "LLM_REVIEW_PENDING"
            and gap.get("artifact_id") in resolved_ids
        )
    ]

    _write(ENRICHMENT_PATH, enrichment)
    print(f"Pending reviews replaced: {replaced}")
    print(f"Remaining pending: {enrichment['summary']['llm_reviews_pending']}")
    print(f"Output: {ENRICHMENT_PATH}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Retry only pending Gemini artifact reviews")
    parser.add_argument("--limit", type=int, default=0, help="Retry at most N pending reviews (0 = all pending)")
    args = parser.parse_args()
    if args.limit < 0:
        raise SystemExit("--limit must be >= 0")
    raise SystemExit(retry_pending(max_artifacts=args.limit))


if __name__ == "__main__":
    main()
