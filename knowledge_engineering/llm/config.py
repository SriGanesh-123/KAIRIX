"""Central LLM configuration shared by all agents."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    api_key: str
    max_retries: int = 1

    @classmethod
    def from_env(cls) -> "LLMConfig | None":
        provider = os.getenv("LLM_PROVIDER", "").strip().lower()
        model = os.getenv("LLM_MODEL", "").strip()
        api_key = os.getenv("LLM_API_KEY", "").strip()
        if not api_key and provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY", "").strip()
        elif not api_key and provider == "groq":
            api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not provider or not model or not api_key:
            return None
        return cls(provider=provider, model=model, api_key=api_key)
