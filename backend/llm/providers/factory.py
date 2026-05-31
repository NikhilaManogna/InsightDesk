from __future__ import annotations

from backend.llm.providers.base import BaseLLMProvider, LLMProviderError
from backend.llm.providers.gemini_provider import GeminiProvider
from backend.llm.providers.groq_provider import GroqProvider
from backend.llm.providers.openrouter_provider import OpenRouterProvider
from backend.utils.config import Settings


def build_llm_provider(settings: Settings) -> BaseLLMProvider:
    provider = settings.llm_provider.lower().strip()
    timeout = settings.llm_timeout_seconds
    retries = settings.llm_max_retries

    if provider == "groq":
        return GroqProvider(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            timeout_seconds=timeout,
            max_retries=retries,
        )
    if provider == "openrouter":
        return OpenRouterProvider(
            api_key=settings.openrouter_api_key,
            model=settings.openrouter_model,
            timeout_seconds=timeout,
            max_retries=retries,
        )
    if provider == "gemini":
        return GeminiProvider(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            timeout_seconds=timeout,
            max_retries=retries,
        )

    raise LLMProviderError(
        f"Unsupported LLM_PROVIDER `{settings.llm_provider}`. "
        "Use groq, openrouter, or gemini."
    )
