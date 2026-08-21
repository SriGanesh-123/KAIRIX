"""Create exactly the provider selected by the developer."""
from __future__ import annotations

from .config import LLMConfig
from .gemini import GeminiReviewer
from .groq import GroqReviewer
from .reviewer import LLMReviewer


def create_reviewer(config: LLMConfig) -> LLMReviewer:
    provider = config.provider.lower()
    if provider == "gemini":
        return GeminiReviewer(config.api_key, config.model, config.max_retries)
    if provider == "groq":
        return GroqReviewer(config.api_key, config.model, config.max_retries)
    raise ValueError(
        f"Unsupported LLM_PROVIDER={config.provider!r}. Supported providers: gemini, groq"
    )
