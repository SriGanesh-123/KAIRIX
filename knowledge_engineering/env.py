"""Application environment loading."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_environment() -> None:
    """Load project .env into the process environment."""
    load_dotenv(PROJECT_ROOT / ".env")