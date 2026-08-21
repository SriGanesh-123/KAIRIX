"""Execution adapter for existing parser entrypoints.

The Knowledge Engineering Agent owns orchestration, while parser logic remains
inside the existing technology-specific parser modules. Execution is explicit
because some legacy parsers are batch-oriented and use their own configured
input/output directories.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any


class ParserExecutionError(RuntimeError):
    """Raised when an existing parser cannot be executed."""


def execute_parser(selection: dict[str, Any], project_root: Path, timeout: int = 300) -> dict[str, Any]:
    """Execute one selected parser module as a Python module.

    Existing parsers are deliberately treated as black boxes. This keeps their
    validated extraction logic unchanged while giving the agent a common
    execution/audit contract.
    """
    entrypoint = selection.get("entrypoint")
    if not entrypoint:
        return {
            "status": "NOT_EXECUTED",
            "reason": "No parser entrypoint was selected.",
        }

    started = time.perf_counter()
    command = [sys.executable, "-m", entrypoint]
    try:
        completed = subprocess.run(
            command,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "TIMEOUT",
            "command": command,
            "duration_seconds": round(time.perf_counter() - started, 3),
            "stdout_tail": (exc.stdout or "")[-4000:],
            "stderr_tail": (exc.stderr or "")[-4000:],
            "reason": f"Parser exceeded {timeout} seconds.",
        }
    except Exception as exc:
        return {
            "status": "EXECUTION_ERROR",
            "command": command,
            "duration_seconds": round(time.perf_counter() - started, 3),
            "reason": str(exc),
        }

    status = "EXECUTED" if completed.returncode == 0 else "FAILED"
    return {
        "status": status,
        "return_code": completed.returncode,
        "command": command,
        "duration_seconds": round(time.perf_counter() - started, 3),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "output_dir": selection.get("output_dir"),
        "reason": "Parser completed successfully." if status == "EXECUTED" else "Parser process returned a non-zero exit code.",
    }
