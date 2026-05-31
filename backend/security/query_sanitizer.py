from __future__ import annotations

import re

from backend.security.validation_models import DangerousQueryError, ValidationCode, ValidationIssue
from backend.utils.logger import get_logger

logger = get_logger(__name__)


WRITE_INTENT = {
    "delete",
    "drop",
    "truncate",
    "remove",
    "erase",
    "insert",
    "update",
    "alter",
    "create",
    "replace",
    "merge",
    "execute",
}

PROMPT_INJECTION_PATTERNS = (
    "ignore previous",
    "ignore the rules",
    "system prompt",
    "developer message",
    "bypass validation",
    "run raw sql",
)


class QuerySanitizer:
    """Screens user intent before it reaches SQL generation."""

    def sanitize_question(self, question: str) -> str:
        cleaned = re.sub(r"\s+", " ", question).strip()
        lowered = cleaned.lower()

        for pattern in PROMPT_INJECTION_PATTERNS:
            if pattern in lowered:
                logger.warning("blocked_prompt_injection pattern=%s", pattern)
                raise DangerousQueryError(
                    "This request looks like an attempt to bypass the SQL safety rules.",
                    pattern=pattern,
                )

        words = set(re.findall(r"[a-zA-Z_]+", lowered))
        matched = sorted(words & WRITE_INTENT)
        if matched:
            logger.warning("blocked_write_intent terms=%s", ",".join(matched))
            raise DangerousQueryError(
                "Only read-only analytical questions are allowed. Destructive or data-changing requests are blocked.",
                terms=matched,
            )

        return cleaned

    def validate_raw_sql_text(self, sql: str) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if "--" in sql or "/*" in sql or "*/" in sql:
            issues.append(
                ValidationIssue(
                    ValidationCode.DANGEROUS_QUERY,
                    "SQL comments are not allowed.",
                )
            )
        if "$$" in sql or re.search(r"\bBEGIN\b|\bCOMMIT\b|\bROLLBACK\b", sql, re.IGNORECASE):
            issues.append(
                ValidationIssue(
                    ValidationCode.DANGEROUS_QUERY,
                    "Procedural SQL and transaction statements are not allowed.",
                )
            )
        return issues
