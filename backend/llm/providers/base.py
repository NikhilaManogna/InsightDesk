from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class LLMProviderError(RuntimeError):
    """Base error for provider failures."""


class LLMRateLimitError(LLMProviderError):
    """Raised when the provider reports a rate or quota limit."""


class LLMTimeoutError(LLMProviderError):
    """Raised when the provider request times out."""


class LLMInvalidResponseError(LLMProviderError):
    """Raised when a provider returns an unusable response."""


@dataclass(frozen=True)
class LLMRequest:
    system_prompt: str
    user_prompt: str
    temperature: float = 0.1
    max_tokens: int = 1200


class BaseLLMProvider(ABC):
    name: str

    @abstractmethod
    def complete(self, request: LLMRequest) -> str:
        """Return model text for a single chat-style request."""

    def _read_openai_compatible_text(self, payload: dict[str, Any]) -> str:
        choices = payload.get("choices") or []
        if not choices:
            raise LLMInvalidResponseError(f"{self.name} returned no choices.")
        message = choices[0].get("message") or {}
        text = message.get("content")
        if not isinstance(text, str) or not text.strip():
            raise LLMInvalidResponseError(f"{self.name} returned an empty message.")
        return text.strip()
