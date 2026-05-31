from __future__ import annotations

import json


DEFAULT_ALIASES = {
    "revenue": "total_revenue revenue amount order_amount sales",
    "sales": "orders transactions revenue",
    "customers": "customers users customer_id user_id",
    "products": "products product sku item",
}


def parse_business_aliases(raw: str | None) -> dict[str, str]:
    if not raw:
        return DEFAULT_ALIASES.copy()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return {str(key): str(value) for key, value in parsed.items()}
    except json.JSONDecodeError:
        pass

    aliases = DEFAULT_ALIASES.copy()
    for item in raw.split(";"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        aliases[key.strip()] = value.strip()
    return aliases


def expand_question(question: str, aliases: dict[str, str]) -> str:
    lowered = question.lower()
    additions = [target for alias, target in aliases.items() if alias.lower() in lowered]
    return f"{question} {' '.join(additions)}".strip() if additions else question
