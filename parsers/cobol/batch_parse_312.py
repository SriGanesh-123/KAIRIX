"""Python 3.12-safe COBOL batch runner.

Uses the existing parse_cobol_file implementation directly in the same
interpreter instead of spawning a subprocess for every COBOL program.
"""

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = PROJECT_ROOT / "source" / "mainframe" / "cobol"
OUTPUT_DIR = PROJECT_ROOT / "output" / "cobol"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parse  # noqa: E402


def write_metadata(file_path: Path, metadata: dict) -> Path:
    output_file = OUTPUT_DIR / f"{file_path.stem}_metadata.json"
    temporary_file = output_file.with_suffix(".json.tmp")

    payload = json.dumps(
        metadata,
        indent=4,
        ensure_ascii=False,
    )

    json.loads(payload)
    temporary_file.write_text(payload, encoding="utf-8")
    json.loads(temporary_file.read_text(encoding="utf-8"))
    temporary_file.replace(output_file)
    return output_file


def main() -> None:
    cobol_files = sorted(BASE_DIR.glob("*.CBL"))

    print("=" * 80)
    print("COBOL BATCH PARSER - PYTHON 3.12 DIRECT MODE")
    print("=" * 80)
    print(f"COBOL files found: {len(cobol_files)}")

    successful = 0
    failed = 0

    for file_path in cobol_files:
        print("=" * 80)
        print(f"PARSING: {file_path.name}")
        print("=" * 80)

        try:
            metadata = parse.parse_cobol_file(file_path)
            output_file = write_metadata(file_path, metadata)

            print(f"SUCCESS: {file_path.name}")
            print(f"Output: {output_file}")
            print(f"Size: {output_file.stat().st_size}")
            print(f"Records: {len(metadata.get('records', []))}")
            print(f"Files: {len(metadata.get('files', []))}")
            print(f"Variables: {len(metadata.get('variables', []))}")
            print(f"Relationships: {len(metadata.get('relationships', []))}")
            successful += 1

        except Exception as error:
            failed += 1
            print(f"FAILED: {file_path.name}")
            print(f"{type(error).__name__}: {error}")

    semantic_output = OUTPUT_DIR / "semantic_data.json"
    combined_metadata = {"programs": []}

    for file_path in cobol_files:
        metadata_file = OUTPUT_DIR / f"{file_path.stem}_metadata.json"
        if not metadata_file.exists() or metadata_file.stat().st_size == 0:
            continue
        try:
            combined_metadata["programs"].append(
                json.loads(metadata_file.read_text(encoding="utf-8"))
            )
        except Exception as error:
            print(f"WARNING reading {metadata_file.name}: {error}")

    semantic_output.write_text(
        json.dumps(combined_metadata, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 80)
    print("BATCH PARSING COMPLETED")
    print("=" * 80)
    print(f"Programs found : {len(cobol_files)}")
    print(f"Programs parsed: {successful}")
    print(f"Programs failed: {failed}")
    print(f"Combined metadata: {semantic_output}")


if __name__ == "__main__":
    main()
