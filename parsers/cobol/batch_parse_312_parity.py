"""Python 3.12 COBOL runner with 3.14 extraction parity.

Uses a crash-safe Tree-sitter parse while semantic extraction is handled by
explicit Python 3.12-compatible compatibility helpers.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = PROJECT_ROOT / "source" / "mainframe" / "cobol"
OUTPUT_DIR = PROJECT_ROOT / "output" / "cobol"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COBOL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(COBOL_DIR))

import compat_helpers_312 as compat  # noqa: E402
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
    divisions = compat.extract_divisions(text)
    sections = compat.extract_sections(text)
    paragraphs = compat.extract_paragraphs(text)
    files = compat.extract_files(text)
    variables = compat.extract_variables(text)
    records = compat.extract_records(variables)
    copybooks = compat.extract_copybooks(text)

    print("  [4/8] Data extraction...", flush=True)
    print("  [5/8] Operations...", flush=True)
    operations = compat.extract_operations(text)
    performs = compat.extract_performs(text)
    calls = compat.extract_calls(text)

    print("  [6/8] I/O and database...", flush=True)
    sql_statements = compat.extract_sql(text)
    cics_statements = compat.extract_cics(text)
    database_tables = compat.extract_database_tables(sql_statements)
    file_operations = compat.extract_file_operations(text)
    moves = compat.extract_moves(text)
    conditions = compat.extract_conditions(text)

    metadata = {
        "file": file_path.name,
        "program_id": compat.extract_program_id(text),
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
    metadata["relationships"] = compat.extract_relationships(file_path.name, metadata)

    if tree_metadata["error_count"]:
        metadata["parse_errors"] = [{
            "type": "TREE_SITTER_ERROR",
            "count": tree_metadata["error_count"],
        }]

    json.dumps(metadata, ensure_ascii=False)
    print("  PARSE COMPLETE", flush=True)
    return metadata


def write_atomic(file_path: Path, metadata: dict) -> Path:
    output = OUTPUT_DIR / f"{file_path.stem}_metadata.json"
    temp = output.with_suffix(".json.tmp")
    payload = json.dumps(metadata, indent=4, ensure_ascii=False)
    json.loads(payload)
    temp.write_text(payload, encoding="utf-8")
    json.loads(temp.read_text(encoding="utf-8"))
    temp.replace(output)
    return output


def main() -> None:
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
        print()

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
