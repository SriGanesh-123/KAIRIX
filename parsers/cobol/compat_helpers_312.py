"""Safe semantic extraction helpers for Python 3.12 COBOL parsing."""

import re


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def line_number(source: str, pos: int) -> int:
    return source[:pos].count("\n") + 1


def extract_program_id(source: str):
    match = re.search(r"\bPROGRAM-ID\.\s*([A-Z0-9_-]+)", source, re.I)
    return match.group(1).upper() if match else None


def extract_divisions(source):
    result = []
    patterns = (("identification_division", r"\bIDENTIFICATION\s+DIVISION\s*\."),("environment_division", r"\bENVIRONMENT\s+DIVISION\s*\."),("data_division", r"\bDATA\s+DIVISION\s*\."),("procedure_division", r"\bPROCEDURE\s+DIVISION\s*\."))
    for kind, pattern in patterns:
        match = re.search(pattern, source, re.I)
        if match:
            result.append({"type": kind, "text": match.group(0), "start_line": line_number(source, match.start())})
    return result


def extract_sections(source):
    result = []
    pattern = re.compile(r"(?m)^\s*([A-Z][A-Z0-9-]*(?:\s+[A-Z0-9-]+)*)\s+SECTION\s*\.\s*$", re.I)
    for match in pattern.finditer(source):
        result.append({"text": clean(match.group(1)) + " SECTION.", "start_line": line_number(source, match.start())})
    return result


def extract_paragraphs(source):
    result = []
    ignored = {"IDENTIFICATION", "ENVIRONMENT", "DATA", "PROCEDURE", "FILE-CONTROL", "FILE", "WORKING-STORAGE", "LOCAL-STORAGE", "LINKAGE", "INPUT-OUTPUT"}
    pattern = re.compile(r"(?m)^\s{0,11}([A-Z][A-Z0-9-]*|[0-9][0-9A-Z-]*)\.\s*$")
    seen = set()
    for match in pattern.finditer(source):
        name = match.group(1).upper()
        if name in ignored or name in seen:
            continue
        seen.add(name)
        result.append({"text": name + ".", "start_line": line_number(source, match.start())})
    return result


def extract_files(source):
    # The canonical parser represents logical COBOL files by SELECT entries.
    # FD entries describe record layouts and must not be counted as separate files.
    result = []
    select_pattern = re.compile(r"\bSELECT\s+([A-Z0-9-]+)(.*?\.)", re.I | re.S)
    for match in select_pattern.finditer(source):
        text = "SELECT " + match.group(1) + match.group(2)
        item = {"type": "SELECT", "name": match.group(1).upper(), "start_line": line_number(source, match.start())}
        assign = re.search(r"\bASSIGN\s+TO\s+([A-Z0-9-]+)", text, re.I)
        if assign:
            item["assign_to"] = assign.group(1).upper()
        result.append(item)
    return result


def extract_variables(source):
    result = []
    seen = set()
    pattern = re.compile(r"(?m)^\s*(01|02|05|10|15|20|49|77)\s+([A-Z0-9-]+)(.*)$", re.I)
    for match in pattern.finditer(source):
        level = int(match.group(1)); name = match.group(2).upper(); key = (level, name)
        if key in seen: continue
        seen.add(key)
        item = {"name": name, "level": level, "start_line": line_number(source, match.start())}
        tail = match.group(3)
        picture = re.search(r"\bPIC(?:TURE)?\s+([A-Z0-9()VXS9+\-.,]+)", tail, re.I)
        if picture: item["picture"] = picture.group(1).upper()
        value = re.search(r"\bVALUE\s+(.+?)\s*\.\s*$", tail, re.I)
        if value: item["value"] = clean(value.group(1))
        result.append(item)
    for match in re.finditer(r"(?m)^\s*88\s+([A-Z0-9-]+)\s+VALUE\s+(.+?)\s*\.\s*$", source, re.I):
        result.append({"name": match.group(1).upper(), "level": 88, "value": clean(match.group(2)), "start_line": line_number(source, match.start())})
    return result


def extract_records(variables):
    records = []
    for record in [v for v in variables if v.get("level") == 1]:
        fields = []
        for var in variables:
            if var.get("level") in (2, 5):
                fields.append(var.copy())
        records.append({"record_name": record["name"], "level": 1, "start_line": record.get("start_line"), "fields": fields})
    return records


def extract_copybooks(source):
    return [{"name": m.group(1).upper(), "start_line": line_number(source, m.start())} for m in re.finditer(r"(?m)^\s*COPY\s+([A-Z0-9-]+)\s*\.\s*$", source, re.I)]


def _unique(values):
    seen = set(); out = []
    for value in values:
        key = value if isinstance(value, str) else repr(sorted(value.items()))
        if key not in seen: seen.add(key); out.append(value)
    return out


def extract_operations(source):
    patterns = {"perform": r"\bPERFORM\b[^.]*\.", "read": r"\bREAD\s+[A-Z0-9-]+[^.]*\.", "write": r"\bWRITE\s+[A-Z0-9-]+[^.]*\.", "move": r"\bMOVE\b[^.]*\.", "open": r"\bOPEN\s+(?:INPUT|OUTPUT|I-O|EXTEND)?\s*[A-Z0-9-]+[^.]*\.", "close": r"\bCLOSE\s+[A-Z0-9-]+[^.]*\.", "display": r"\bDISPLAY\b[^.]*\.", "if": r"\bIF\b[^.]*\.", "goto": r"\bGO\s+TO\s+[A-Z0-9-]+[^.]*\.", "add": r"\bADD\b[^.]*\."}
    return {key: _unique([clean(x) for x in re.findall(pattern, source, re.I)]) for key, pattern in patterns.items()}


def extract_performs(source):
    return _unique([m.group(1).upper() for m in re.finditer(r"\bPERFORM\s+([A-Z0-9-]+)", source, re.I)])


def extract_calls(source):
    return _unique([m.group(1).upper() for m in re.finditer(r"\bCALL\s+['\"]?([A-Z0-9_-]+)", source, re.I)])


def extract_sql(source):
    return _unique([clean(x) for x in re.findall(r"EXEC\s+SQL(.*?)END-EXEC", source, re.I | re.S)])


def extract_cics(source):
    return _unique([clean(x) for x in re.findall(r"EXEC\s+CICS(.*?)END-EXEC", source, re.I | re.S)])


def extract_database_tables(sql):
    return _unique([m.upper() for text in sql for m in re.findall(r"\b(?:FROM|JOIN|UPDATE|INTO)\s+([A-Z0-9_.-]+)", text, re.I)])


def extract_file_operations(source):
    ops = [{"operation": op.upper(), "file": target.upper()} for op, target in re.findall(r"\b(READ|WRITE|REWRITE|DELETE|CLOSE)\s+([A-Z0-9-]+)", source, re.I)]
    ops += [{"operation": "OPEN", "file": target.upper()} for target in re.findall(r"\bOPEN\s+(?:INPUT|OUTPUT|I-O|EXTEND)?\s*([A-Z0-9-]+)", source, re.I)]
    return ops


def extract_moves(source):
    result = []
    for src, target in re.findall(r"\bMOVE\s+(.+?)\s+TO\s+([A-Z0-9-]+)", source, re.I | re.S):
        src = clean(src)
        if len(src) <= 200: result.append({"source": src, "target": target.upper()})
    return result


def extract_conditions(source):
    values = re.findall(r"\bIF\s+(.+?)(?=\b(?:MOVE|DISPLAY|PERFORM|READ|WRITE|COMPUTE|ADD|SUBTRACT|MULTIPLY|DIVIDE|END-IF)\b|\.)", source, re.I | re.S)
    return _unique([clean(v) for v in values if 0 < len(clean(v)) <= 300])


def extract_relationships(file_name, metadata):
    """Build the compact relationship set used by the canonical 3.14 output.

    Do not emit structural field/paragraph edges here: those are represented
    inside records/paragraphs and were the source of the 3.12 relationship
    explosion (hundreds of synthetic edges per program).
    """
    relationships = []
    def add(kind, target):
        if target:
            relationships.append({"source": file_name, "relationship": kind, "target": target})

    for op in metadata["operations"].get("read", []):
        m = re.search(r"\bREAD\s+([A-Z0-9-]+)", op, re.I)
        if m: add("READS", m.group(1).upper())
    for op in metadata["operations"].get("write", []):
        m = re.search(r"\bWRITE\s+([A-Z0-9-]+)", op, re.I)
        if m: add("WRITES", m.group(1).upper())
    for item in metadata.get("copybooks", []): add("USES_COPYBOOK", item.get("name"))
    for record in metadata.get("records", []): add("CONTAINS_RECORD", record.get("record_name"))
    for target in metadata.get("performs", []): add("PERFORMS", target)
    for target in metadata.get("calls", []): add("CALLS", target)

    # Deduplicate exact semantic edges while preserving order.
    seen = set(); result = []
    for rel in relationships:
        key = (rel["source"], rel["relationship"], rel["target"])
        if key not in seen:
            seen.add(key); result.append(rel)
    return result
