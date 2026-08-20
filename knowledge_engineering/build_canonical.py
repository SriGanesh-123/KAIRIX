"""Build and validate canonical metadata from parser outputs."""

from __future__ import annotations

import json
from pathlib import Path

from .normalize import normalize_file
from .schema import KnowledgeDocument
from .validate import validate_document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "output"
KNOWLEDGE_OUTPUT = OUTPUT_ROOT / "knowledge" / "canonical_metadata.json"


def main() -> None:
    documents = []

    source_dirs = [
        (OUTPUT_ROOT / "sql", "sql"),
        (OUTPUT_ROOT / "cobol", "cobol"),
        (OUTPUT_ROOT / "ssis", "ssis"),
    ]

    for directory, source_type in source_dirs:
        if not directory.exists():
            continue

        for path in sorted(directory.glob("*.json")):
            if path.name == "semantic_data.json":
                continue

            documents.append(
                normalize_file(path, source_type)
            )

    merged = KnowledgeDocument()

    for document in documents:
        merged.artifacts.extend(document.artifacts)
        merged.entities.extend(document.entities)
        merged.relationships.extend(document.relationships)
        merged.evidence.extend(document.evidence)
        merged.business_rules.extend(document.business_rules)

    merged.artifacts = list({item.id: item for item in merged.artifacts}.values())
    merged.entities = list({item.id: item for item in merged.entities}.values())
    merged.relationships = list({item.id: item for item in merged.relationships}.values())
    merged.evidence = list({item.id: item for item in merged.evidence}.values())
    merged.business_rules = list({item.id: item for item in merged.business_rules}.values())

    errors = validate_document(merged)

    KNOWLEDGE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    payload = merged.model_dump()
    payload["validation"] = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "counts": {
            "artifacts": len(merged.artifacts),
            "entities": len(merged.entities),
            "relationships": len(merged.relationships),
            "evidence": len(merged.evidence),
            "business_rules": len(merged.business_rules),
        },
    }

    KNOWLEDGE_OUTPUT.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 72)
    print("CANONICAL KNOWLEDGE BUILD")
    print("=" * 72)
    print(f"Output       : {KNOWLEDGE_OUTPUT}")
    print(f"Artifacts    : {len(merged.artifacts)}")
    print(f"Entities     : {len(merged.entities)}")
    print(f"Relationships: {len(merged.relationships)}")
    print(f"Evidence     : {len(merged.evidence)}")
    print(f"Business rules: {len(merged.business_rules)}")
    print(f"Validation   : {'PASS' if not errors else 'FAIL'}")

    if errors:
        print("Errors:")
        for error in errors:
            print(f"  - {error}")


if __name__ == "__main__":
    main()
