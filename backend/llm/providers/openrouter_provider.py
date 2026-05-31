from __future__ import annotations

import time

import requests
from requests import Timeout

from backend.llm.providers.base import (
    BaseLLMProvider,
    LLMProviderError,
    LLMRateLimitError,
    LLMRequest,
    LLMTimeoutError,
)
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class OpenRouterProvider(BaseLLMProvider):
    name = "openrouter"

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_seconds: int,
        max_retries: int = 2,
    ) -> None:
        if not api_key:
            raise LLMProviderError("OPENROUTER_API_KEY is not configured.")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    def complete(self, request: LLMRequest) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "InsightDesk",
        }
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                logger.info(
                    "llm_request provider=openrouter model=%s attempt=%s",
                    self.model,
                    attempt + 1,
                )
                response = requests.post(
                    self.url,
                    headers=headers,
                    json=body,
                    timeout=self.timeout_seconds,
                )
                if response.status_code == 429:
                    raise LLMRateLimitError("OpenRouter rate limit or quota was exceeded.")
                response.raise_for_status()
                return self._read_openai_compatible_text(response.json())
            except LLMRateLimitError:
                raise
            except Timeout as exc:
                raise LLMTimeoutError("OpenRouter request timed out.") from exc
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(0.6 * (attempt + 1))

        raise LLMProviderError(f"OpenRouter request failed: {last_error}") from last_error
