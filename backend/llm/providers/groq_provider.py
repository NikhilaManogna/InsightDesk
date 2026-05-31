from __future__ import annotations

import time

from groq import APIConnectionError, APIStatusError, APITimeoutError, Groq, RateLimitError

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


class GroqProvider(BaseLLMProvider):
    name = "groq"

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_seconds: int,
        max_retries: int = 2,
    ) -> None:
        if not api_key:
            raise LLMProviderError("GROQ_API_KEY is not configured.")
        self.model = model
        self.max_retries = max_retries
        self.client = Groq(api_key=api_key, timeout=timeout_seconds)

    def complete(self, request: LLMRequest) -> str:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                logger.info("llm_request provider=groq model=%s attempt=%s", self.model, attempt + 1)
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": request.system_prompt},
                        {"role": "user", "content": request.user_prompt},
                    ],
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                )
                content = response.choices[0].message.content if response.choices else None
                if not content or not content.strip():
                    raise LLMInvalidResponseError("Groq returned an empty response.")
                return content.strip()
            except RateLimitError as exc:
                logger.warning("llm_rate_limit provider=groq model=%s", self.model)
                raise LLMRateLimitError("Groq rate limit or quota was exceeded.") from exc
            except APITimeoutError as exc:
                logger.warning("llm_timeout provider=groq model=%s", self.model)
                raise LLMTimeoutError("Groq request timed out.") from exc
            except (APIConnectionError, APIStatusError, LLMInvalidResponseError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(0.6 * (attempt + 1))

        raise LLMProviderError(f"Groq request failed: {last_error}") from last_error
