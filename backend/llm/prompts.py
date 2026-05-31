from __future__ import annotations

from typing import Any

from backend.db.schema_metadata import Relationship, SchemaMap


SQL_SYSTEM_PROMPT = """You are a senior analytics engineer.
Generate one safe SQL query for the user's question.

Rules:
- Return SQL only. No markdown.
- Use only SELECT or WITH queries.
- Do not use INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, COPY, MERGE, GRANT, REVOKE, CALL, EXECUTE, or stored procedures.
- Use table and column names exactly as provided in the schema.
- Prefer PostgreSQL-compatible syntax when the dialect is postgres.
- Use DATE_TRUNC('day'|'week'|'month', timestamp_column) for PostgreSQL time buckets.
- Prefer readable CTEs for multi-step logic.
- Add conservative filters only when the question asks for them.
- If the question is ambiguous, do not guess. Ask for clarification outside SQL generation.
"""


SQL_REPAIR_PROMPT = """The previous SQL response was not usable.
Return a corrected SQL query only.

Rules:
- Return SQL only. No markdown.
- The query must start with SELECT or WITH.
- Do not include explanations.
- Preserve PostgreSQL syntax when dialect is postgres.
"""


FEW_SHOT_SQL_EXAMPLES = """Reusable patterns:

Aggregation:
SELECT region, SUM(revenue) AS total_revenue
FROM orders
GROUP BY region
ORDER BY total_revenue DESC
LIMIT 1000

Join:
SELECT c.customer_name, SUM(o.revenue) AS total_revenue
FROM orders o
JOIN customers c ON o.customer_id = c.id
GROUP BY c.customer_name
ORDER BY total_revenue DESC
LIMIT 1000

PostgreSQL time series:
SELECT DATE_TRUNC('month', created_at) AS month, SUM(revenue) AS total_revenue
FROM orders
GROUP BY month
ORDER BY month
LIMIT 1000

Ranking:
SELECT product, SUM(revenue) AS total_revenue
FROM orders
GROUP BY product
ORDER BY total_revenue DESC
LIMIT 10
"""


INSIGHT_PROMPT = """You are a business analyst reviewing query results.
Write concise, practical insights for a decision maker.

Rules:
- Mention trends, outliers, concentration, or gaps when visible.
- Avoid pretending certainty if the result is small.
- Do not invent numbers that are not in the sample.
- Keep it under 180 words.
"""


def format_schema_for_prompt(schema: SchemaMap) -> str:
    lines: list[str] = []
    for table_name, columns in sorted(schema.items()):
        rendered_columns = ", ".join(f"{col} {dtype}" for col, dtype in columns.items())
        lines.append(f"- {table_name}: {rendered_columns}")
    return "\n".join(lines)


def format_relationships_for_prompt(relationships: tuple[Relationship, ...]) -> str:
    if not relationships:
        return "- No explicit relationships detected."
    return "\n".join(f"- {relationship.render()}" for relationship in relationships)


def build_sql_prompt(
    question: str,
    schema: SchemaMap,
    dialect: str,
    relationships: tuple[Relationship, ...] = (),
) -> str:
    return (
        f"SQL dialect: {dialect}\n\n"
        f"Schema:\n{format_schema_for_prompt(schema)}\n\n"
        f"Relationships:\n{format_relationships_for_prompt(relationships)}\n\n"
        f"Examples:\n{FEW_SHOT_SQL_EXAMPLES}\n\n"
        f"User question:\n{question.strip()}\n"
    )


def build_sql_repair_prompt(
    question: str,
    schema: SchemaMap,
    dialect: str,
    bad_sql: str,
    errors: list[str] | None = None,
    relationships: tuple[Relationship, ...] = (),
) -> str:
    rendered_errors = "\n".join(f"- {error}" for error in errors or ["The SQL was invalid."])
    return (
        f"SQL dialect: {dialect}\n\n"
        f"Schema:\n{format_schema_for_prompt(schema)}\n\n"
        f"Relationships:\n{format_relationships_for_prompt(relationships)}\n\n"
        f"Original question:\n{question.strip()}\n\n"
        f"Invalid SQL:\n{bad_sql}\n\n"
        f"Problems:\n{rendered_errors}\n"
    )


def build_insight_prompt(question: str, sql: str, rows: list[dict[str, Any]]) -> str:
    preview = rows[:25]
    return (
        f"{INSIGHT_PROMPT}\n\n"
        f"Question: {question}\n\n"
        f"SQL:\n{sql}\n\n"
        f"Result sample:\n{preview}\n"
    )
