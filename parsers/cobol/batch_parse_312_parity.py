"""Python 3.12 COBOL runner with 3.14 extraction parity.

Uses a crash-safe Tree-sitter parse while delegating semantic extraction to
Python 3.12-compatible helpers that mirror the known-good 3.14 behavior.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = PROJECT_ROOT / "source" / "mainframe" / "cobol"
OUTPUT_DIR = PROJECT_ROOT / "output" / "cobol"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compat_helpers_312 as extract  # noqa: E402
from tree_sitter_language_pack import get_parser  # noqa: E402

PARSER = get_parser("cobol")


def safe_tree_metadata(root):
    node_count = 0
    error_count = 0
    type_counts = {}
    stack = [root]
    while stack:
        node = stack.pop()
        node_count += 1
        node_type = node.type
        type_counts[node_type] = type_counts.get(node_type, 0) + 1
        if node_type == "ERROR":
            error_count += 1
        stack.extend(reversed(node.children))
    return {
        "root_type": root.type,
        "has_error": root.has_error,
        "node_count": node_count,
        "error_count": error_count,
        "node_types": type_counts,
    }


def parse_one(file_path: Path) -> dict:
    print()
    print("=" * 80)
    print(f"PARSING: {file_path.name}")
    print("=" * 80)

    source = file_path.read_bytes()
    text = source.decode("utf-8", errors="replace")

    print("  [1/8] Tree-sitter parse...", flush=True)
    tree = PARSER.parse(source)
    root = tree.root_node

    print("  [2/8] Tree metadata...", flush=True)
    tree_metadata = safe_tree_metadata(root)

    print("  [3/8] Source extraction...", flush=True)
    program_id = extract.extract_program_id(text)
    divisions = extract.extract_divisions(text)
    sections = extract.extract_sections(text)
    paragraphs = extract.extract_paragraphs(text)

    print("  [4/8] Data extraction...", flush=True)
    files = extract.extract_files(text)
    variables = extract.extract_variables(text)
    records = extract.extract_records(variables)
    copybooks = extract.extract_copybooks(text)

    print("  [5/8] Operations...", flush=True)
    operations = extract.extract_operations(text)
    performs = extract.extract_performs(text)
    calls = extract.extract_calls(text)

    print("  [6/8] I/O and database...", flush=True)
    sql_statements = extract.extract_sql(text)
    database_tables = extract.extract_database_tables(sql_statements)
    file_operations = extract.extract_file_operations(text)
    moves = extract.extract_moves(text)
    conditions = extract.extract_conditions(text)
    cics_statements = extract.extract_cics(text)

    metadata = {
        "file": file_path.name,
        "program_id": program_id,
        "parser": {
            "name": "Tree-sitter",
            "language": "COBOL",
            "grammar": "tree-sitter-language-pack",
        },
        "root": root.type,
        "has_errors": root.has_error,
        "divisions": divisions,
        "sections": sections,
        "paragraphs": paragraphs,
        "files": files,
        "variables": variables,
        "records": records,
        "copybooks": copybooks,
        "operations": operations,
        "performs": performs,
        "calls": calls,
        "sql_statements": sql_statements,
        "cics_statements": cics_statements,
        "database_tables": database_tables,
        "database_columns": [],
        "file_operations": file_operations,
        "moves": moves,
        "conditions": conditions,
        "tree_sitter": tree_metadata,
        "relationships": [],
        "parse_errors": [],
    }

    print("  [7/8] Building metadata...", flush=True)
    print("  [8/8] Relationships...", flush=True)
    metadata["relationships"] = extract.extract_relationships(file_path.name, metadata)

    if tree_metadata["error_count"]:
        metadata["parse_errors"] = [{"type": "TREE_SITTER_ERROR", "count": tree_metadata["error_count"]}]

    json.dumps(metadata, ensure_ascii=False)
    print("  PARSE COMPLETE", flush=True)
    return metadata


def write_atomic(file_path: Path, metadata: dict, output_dir: Path | None = None) -> Path:
    out_dir = output_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{file_path.stem}_metadata.json"
    temp = output.with_suffix(".json.tmp")
    payload = json.dumps(metadata, indent=4, ensure_ascii=False)
    json.loads(payload)
    temp.write_text(payload, encoding="utf-8")
    json.loads(temp.read_text(encoding="utf-8"))
    temp.replace(output)
    return output


def parse_single_file(source_path: str | Path, output_dir: Path | None = None) -> dict:
    """Parse exactly one COBOL source file and persist its metadata."""
    p = Path(source_path)
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"COBOL source file not found: {p}")

    metadata = parse_one(p)
    write_atomic(p, metadata, output_dir=output_dir)
    return metadata


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="COBOL Parser (Single-File & Batch)")
    parser.add_argument("--file", type=str, default="", help="Path to a single COBOL file to parse")
    args = parser.parse_args()

    if args.file:
        file_path = Path(args.file)
        try:
            metadata = parse_single_file(file_path)
            print(f"SUCCESS: {file_path.name}")
            print(f"  Records: {len(metadata.get('records', []))}")
            print(f"  Files: {len(metadata.get('files', []))}")
            print(f"  Variables: {len(metadata.get('variables', []))}")
            print(f"  Relationships: {len(metadata.get('relationships', []))}")
        except Exception as error:
            print(f"FAILED: {file_path.name}")
            print(f"{type(error).__name__}: {error}")
            sys.exit(1)
        return

    files = sorted(BASE_DIR.glob("*.CBL"))
    print("=" * 80)
    print("COBOL BATCH PARSER - PYTHON 3.12 / 3.14 PARITY MODE")
    print("=" * 80)
    print(f"COBOL files found: {len(files)}")

    parsed = []
    failed = 0

    for file_path in files:
        try:
            metadata = parse_one(file_path)
            output = write_atomic(file_path, metadata)
            parsed.append(metadata)
            print(f"SUCCESS: {file_path.name}")
            print(f"  Output: {output}")
            print(f"  Size: {output.stat().st_size}")
            print(f"  Records: {len(metadata['records'])}")
            print(f"  Files: {len(metadata['files'])}")
            print(f"  Variables: {len(metadata['variables'])}")
            print(f"  Relationships: {len(metadata['relationships'])}")
        except Exception as error:
            failed += 1
            print(f"FAILED: {file_path.name}")
            print(f"{type(error).__name__}: {error}")

    semantic_output = OUTPUT_DIR / "semantic_data.json"
    semantic_output.write_text(
        json.dumps({"programs": parsed}, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 80)
    print("BATCH PARSING COMPLETED")
    print("=" * 80)
    print(f"Programs found : {len(files)}")
    print(f"Programs parsed: {len(parsed)}")
    print(f"Programs failed: {failed}")
    print(f"Combined metadata: {semantic_output}")


if __name__ == "__main__":
    main()
