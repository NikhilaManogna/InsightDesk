from __future__ import annotations

import re


AMBIGUOUS_TERMS = {
    "best": "Do you mean by revenue, order volume, profit, retention, or another metric?",
    "top": "Do you mean top by revenue, count, profit, or recent activity?",
    "highest growth": "Should growth be measured by revenue, order count, users, or percentage change?",
    "growth": "Should growth be measured by revenue, order count, users, or percentage change?",
    "performance": "Which metric should define performance: revenue, count, profit, or conversion?",
}


class AmbiguousQuestionError(ValueError):
    """Raised when a business term needs a metric before SQL is safe."""


def detect_ambiguity(question: str) -> str | None:
    normalized = re.sub(r"\s+", " ", question.lower()).strip()
    metric_words = {
        "revenue",
        "sales",
        "profit",
        "margin",
        "count",
        "volume",
        "orders",
        "quantity",
        "average",
        "total",
    }
    if any(word in normalized for word in metric_words):
        return None

    for term, clarification in AMBIGUOUS_TERMS.items():
        if term in normalized:
            return clarification
    return None
