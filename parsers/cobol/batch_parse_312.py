"""Python 3.12-safe COBOL batch parser.

Keeps the existing deterministic source extraction helpers, but avoids the
native Tree-sitter node APIs that were crashing on EARNPREM/KPICALC under
Python 3.12. Tree-sitter is used only for parsing and safe node-type counts.
"""

import json
import re
import sys
from pathlib import Path

from tree_sitter_language_pack import get_parser

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = PROJECT_ROOT / "source" / "mainframe" / "cobol"
OUTPUT_DIR = PROJECT_ROOT / "output" / "cobol"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parse  # noqa: E402

PARSER = get_parser("cobol")


def clean_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def unique(values):
    result = []
    seen = set()
    for value in values:
        key = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def line_number(source: str, position: int) -> int:
    return source[:position].count("\n") + 1


def extract_tree_metadata(root):
    """Only use stable Tree-sitter APIs: type, children and has_error."""
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
        for child in reversed(node.children):
            stack.append(child)

    return {
        "root_type": root.type,
        "has_error": root.has_error,
        "node_count": node_count,
        "error_count": error_count,
        "node_types": type_counts,
    }


def extract_program_id(source):
    match = re.search(r"\bPROGRAM-ID\.\s*([A-Z0-9_-]+)", source, re.IGNORECASE)
    return match.group(1).upper() if match else None


def extract_operations(source):
    patterns = {
        "perform": r"\bPERFORM\b[^.]*\.",
        "read": r"\bREAD\s+[A-Z0-9-]+[^.]*\.",
        "write": r"\bWRITE\s+[A-Z0-9-]+[^.]*\.",
        "move": r"\bMOVE\b[^.]*\.",
        "open": r"\bOPEN\s+(?:INPUT|OUTPUT|I-O|EXTEND)?\s*[A-Z0-9-]+[^.]*\.",
        "close": r"\bCLOSE\s+[A-Z0-9-]+[^.]*\.",
        "display": r"\bDISPLAY\b[^.]*\.",
        "if": r"\bIF\b[^.]*\.",
        "goto": r"\bGO\s+TO\s+[A-Z0-9-]+[^.]*\.",
        "add": r"\bADD\b[^.]*\.",
    }
    return {
        key: unique([clean_spaces(x) for x in re.findall(pattern, source, re.IGNORECASE)])
        for key, pattern in patterns.items()
    }


def extract_variables_312(source: str, records: list, existing: list) -> list:
    """Recover the variable inventory without Tree-sitter position/text APIs.

    The 3.14 parser counted COBOL data definitions broadly: 01 groups,
    elementary 02/05/10/15/20/49/77 items, 88 condition names, and the
    SELECT file names. Reconstruct the same inventory from source text.
    """
    variables = []
    seen = set()

    def add(item):
        name = item.get("name")
        if not name:
            return
        key = (str(item.get("level", "")), name.upper())
        if key not in seen:
            seen.add(key)
            variables.append(item)

    # 01-level data definitions, including FD records and working-storage groups.
    level_01 = re.compile(
        r"(?m)^\s*01\s+(?P<name>[A-Z0-9-]+)\s*\.\s*$",
        re.IGNORECASE,
    )
    for m in level_01.finditer(source):
        add({
            "name": m.group("name").upper(),
            "level": 1,
            "start_line": line_number(source, m.start()),
        })

    # Elementary and subordinate COBOL data items.
    elementary = re.compile(
        r"(?m)^\s*(?P<level>02|05|10|15|20|49|77)\s+"
        r"(?P<name>[A-Z0-9-]+)"
        r"(?:\s+PIC(?:TURE)?\s+"
        r"(?P<picture>[A-Z0-9()VXS9+\-.,]+))?"
        r"(?:\s+VALUE\s+(?P<value>[^.]+))?\s*\.\s*$",
        re.IGNORECASE,
    )
    for m in elementary.finditer(source):
        item = {
            "name": m.group("name").upper(),
            "level": int(m.group("level")),
            "start_line": line_number(source, m.start()),
        }
        if m.group("picture"):
            item["picture"] = m.group("picture").upper()
        if m.group("value"):
            item["value"] = clean_spaces(m.group("value"))
        add(item)

    # 88 condition names are variables in the parser's semantic inventory.
    condition = re.compile(
        r"(?m)^\s*88\s+(?P<name>[A-Z0-9-]+)\s+VALUE\s+(?P<value>[^.]+)\.\s*$",
        re.IGNORECASE,
    )
    for m in condition.finditer(source):
        add({
            "name": m.group("name").upper(),
            "level": 88,
            "value": clean_spaces(m.group("value")),
            "start_line": line_number(source, m.start()),
        })

    # SELECT file identifiers are represented in the 3.14 variable inventory.
    select_pattern = re.compile(
        r"(?mi)^\s*SELECT\s+(?P<name>[A-Z0-9-]+)\b"
    )
    for m in select_pattern.finditer(source):
        add({
            "name": m.group("name").upper(),
            "level": "SELECT",
            "start_line": line_number(source, m.start()),
        })

    # Preserve any variables already recovered by fallback_extract().
    for item in existing:
        add(item)

    return variables


def parse_cobol_file_312(file_path: Path) -> dict:
    print()
    print("=" * 80)
    print(f"PARSING: {file_path.name}")
    print("=" * 80)

    source = file_path.read_bytes()
    source_text = source.decode("utf-8", errors="replace")

    print("  [1/8] Tree-sitter parse...", flush=True)
    tree = PARSER.parse(source)
    root = tree.root_node

    print("  [2/8] Tree metadata...", flush=True)
    tree_metadata = extract_tree_metadata(root)

    print("  [3/8] Source extraction...", flush=True)
    metadata = {
        "file": file_path.name,
        "program_id": extract_program_id(source_text),
        "parser": {
            "name": "Tree-sitter",
            "language": "COBOL",
            "grammar": "tree-sitter-language-pack",
        },
        "root": root.type,
        "has_errors": root.has_error,
        "divisions": [],
        "paragraphs": [],
        "records": [],
        "files": [],
        "variables": [],
        "copybooks": [],
        "operations": extract_operations(source_text),
        "performs": [],
        "calls": [],
        "sql_statements": [],
        "cics_statements": [],
        "database_tables": [],
        "database_columns": [],
        "file_operations": [],
        "moves": [],
        "conditions": [],
        "tree_sitter": tree_metadata,
        "relationships": [],
        "parse_errors": [],
    }

    metadata = parse.fallback_extract(source_text, metadata)

    print("  [4/8] Data extraction...", flush=True)
    metadata["records"] = parse.extract_records(source_text)
    metadata["variables"] = extract_variables_312(
        source_text,
        metadata["records"],
        metadata["variables"],
    )

    print("  [5/8] Operations...", flush=True)
    metadata["performs"] = unique(
        re.findall(r"\bPERFORM\s+([A-Z0-9-]+)", source_text, re.IGNORECASE)
    )
    metadata["calls"] = unique(
        re.findall(r"\bCALL\s+['\"]?([A-Z0-9_-]+)", source_text, re.IGNORECASE)
    )

    print("  [6/8] I/O and database...", flush=True)
    metadata["sql_statements"] = unique(
        clean_spaces(x)
        for x in re.findall(r"EXEC\s+SQL(.*?)END-EXEC", source_text, re.IGNORECASE | re.DOTALL)
    )
    metadata["cics_statements"] = unique(
        clean_spaces(x)
        for x in re.findall(r"EXEC\s+CICS(.*?)END-EXEC", source_text, re.IGNORECASE | re.DOTALL)
    )
    metadata["database_tables"] = unique(
        x.upper()
        for sql in metadata["sql_statements"]
        for x in re.findall(r"\b(?:FROM|JOIN|UPDATE|INTO)\s+([A-Z0-9_.-]+)", sql, re.IGNORECASE)
    )
    metadata["file_operations"] = [
        {"operation": operation, "file": target.upper()}
        for operation, target in re.findall(
            r"\b(READ|WRITE|REWRITE|DELETE|CLOSE)\s+([A-Z0-9-]+)",
            source_text,
            re.IGNORECASE,
        )
    ]
    metadata["file_operations"] += [
        {"operation": "OPEN", "file": target.upper()}
        for target in re.findall(
            r"\bOPEN\s+(?:INPUT|OUTPUT|I-O|EXTEND)?\s*([A-Z0-9-]+)",
            source_text,
            re.IGNORECASE,
        )
    ]
    metadata["moves"] = [
        {"source": clean_spaces(src), "target": target.upper()}
        for src, target in re.findall(
            r"\bMOVE\s+(.+?)\s+TO\s+([A-Z0-9-]+)",
            source_text,
            re.IGNORECASE | re.DOTALL,
        )
        if len(clean_spaces(src)) <= 200
    ]
    metadata["conditions"] = unique(
        clean_spaces(x)
        for x in re.findall(
            r"\bIF\s+(.+?)(?=\b(?:MOVE|DISPLAY|PERFORM|READ|WRITE|COMPUTE|ADD|SUBTRACT|MULTIPLY|DIVIDE|END-IF)\b|\.)",
            source_text,
            re.IGNORECASE | re.DOTALL,
        )
        if 0 < len(clean_spaces(x)) <= 300
    )

    print("  [7/8] Building metadata...", flush=True)
    print("  [8/8] Relationships...", flush=True)
    metadata["relationships"] = parse.extract_relationships(file_path.name, metadata)

    if tree_metadata["error_count"]:
        metadata["parse_errors"] = [{
            "type": "TREE_SITTER_ERROR",
            "count": tree_metadata["error_count"],
        }]

    print("  PARSE COMPLETE", flush=True)
    return metadata


def write_metadata(file_path: Path, metadata: dict) -> Path:
    output_file = OUTPUT_DIR / f"{file_path.stem}_metadata.json"
    temporary_file = output_file.with_suffix(".json.tmp")
    payload = json.dumps(metadata, indent=4, ensure_ascii=False)
    json.loads(payload)
    temporary_file.write_text(payload, encoding="utf-8")
    json.loads(temporary_file.read_text(encoding="utf-8"))
    temporary_file.replace(output_file)
    return output_file


def main() -> None:
    cobol_files = sorted(BASE_DIR.glob("*.CBL"))

    print("=" * 80)
    print("COBOL BATCH PARSER - PYTHON 3.12 SAFE MODE")
    print("=" * 80)
    print(f"COBOL files found: {len(cobol_files)}")
    for file_path in cobol_files:
        print(f"  - {file_path.name}")

    successful = 0
    failed = 0
    parsed = []

    for file_path in cobol_files:
        try:
            metadata = parse_cobol_file_312(file_path)
            output_file = write_metadata(file_path, metadata)
            parsed.append(metadata)
            successful += 1
            print(f"SUCCESS: {file_path.name}")
            print(f"  Output: {output_file}")
            print(f"  Size: {output_file.stat().st_size}")
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
    semantic_payload = {"programs": parsed}
    semantic_output.write_text(
        json.dumps(semantic_payload, indent=4, ensure_ascii=False),
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
