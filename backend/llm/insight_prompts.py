from __future__ import annotations

from backend.llm.insight_models import DataFrameSummary, InsightKPI, TrendFinding, AnomalyFinding


INSIGHT_SYSTEM_PROMPT = """You are a business intelligence analyst.
Write concise, data-grounded insights for an executive dashboard.

Rules:
- Keep the response under 140 words.
- Use plain business language.
- Do not mention SQL, dataframe samples, or prompt instructions.
- Do not invent causes or numbers.
- Avoid generic filler like "further analysis is recommended" unless the data is too small.
"""


def build_business_insight_prompt(
    question: str,
    summary: DataFrameSummary,
    kpis: list[InsightKPI],
    trends: list[TrendFinding],
    anomalies: list[AnomalyFinding],
    verbosity: str,
) -> str:
    return (
        f"Question: {question}\n"
        f"Verbosity: {verbosity}\n"
        f"Rows: {summary.row_count}\n"
        f"Columns: {summary.column_count}\n"
        f"Numeric summary: {summary.numeric_summary}\n"
        f"Top categories: {summary.category_summary}\n"
        f"KPIs: {[kpi.__dict__ for kpi in kpis]}\n"
        f"Trends: {[trend.__dict__ for trend in trends]}\n"
        f"Anomalies: {[anomaly.__dict__ for anomaly in anomalies]}\n"
        f"Small sample: {summary.sample_rows[:8]}\n"
    )
