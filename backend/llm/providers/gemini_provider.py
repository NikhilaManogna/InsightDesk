from __future__ import annotations

import time

from backend.llm.providers.base import (
    BaseLLMProvider,
    LLMInvalidResponseError,
    LLMProviderError,
    LLMRateLimitError,
    LLMRequest,
    LLMTimeoutError,
)
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class GeminiProvider(BaseLLMProvider):
    name = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_seconds: int,
        max_retries: int = 2,
    ) -> None:
        if not api_key:
            raise LLMProviderError("GEMINI_API_KEY is not configured.")
        try:
            from google import genai
        except ImportError as exc:
            raise LLMProviderError(
                "Gemini provider requires optional package `google-genai`."
            ) from exc

        self.model = model
        self.max_retries = max_retries
        self.client = genai.Client(api_key=api_key)
        self.timeout_seconds = timeout_seconds

    def complete(self, request: LLMRequest) -> str:
        prompt = f"{request.system_prompt}\n\n{request.user_prompt}"
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                logger.info("llm_request provider=gemini model=%s attempt=%s", self.model, attempt + 1)
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                )
                text = getattr(response, "text", "") or ""
                if not text.strip():
                    raise LLMInvalidResponseError("Gemini returned an empty response.")
                return text.strip()
            except Exception as exc:
                message = str(exc).lower()
                if "429" in message or "quota" in message or "rate" in message:
                    raise LLMRateLimitError("Gemini rate limit or quota was exceeded.") from exc
                if "timeout" in message:
                    raise LLMTimeoutError("Gemini request timed out.") from exc
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(0.6 * (attempt + 1))

        raise LLMProviderError(f"Gemini request failed: {last_error}") from last_error
