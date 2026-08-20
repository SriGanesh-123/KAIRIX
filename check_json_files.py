import json
from pathlib import Path

for root in ["output/sql", "output/cobol", "output/ssis"]:
    print(f"\n=== {root} ===")

    for path in sorted(Path(root).glob("*.json")):
        try:
            text = path.read_text(encoding="utf-8")

            if not text.strip():
                print(f"EMPTY   : {path} ({path.stat().st_size} bytes)")
                continue

            json.loads(text)
            print(f"VALID   : {path.name} ({path.stat().st_size} bytes)")

        except Exception as e:
            print(f"INVALID : {path} ({path.stat().st_size} bytes)")
            print(f"          {type(e).__name__}: {e}")
