from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class InsightKPI:
    label: str
    value: str
    detail: str | None = None


@dataclass(frozen=True)
class TrendFinding:
    column: str
    direction: str
    change_percent: float | None
    description: str


@dataclass(frozen=True)
class AnomalyFinding:
    column: str
    value: str
    description: str


@dataclass(frozen=True)
class DataFrameSummary:
    row_count: int
    column_count: int
    numeric_summary: dict[str, dict[str, float]]
    category_summary: dict[str, list[tuple[str, int]]]
    sample_rows: list[dict[str, object]]


@dataclass
class InsightBundle:
    summary: str
    key_findings: list[str] = field(default_factory=list)
    kpis: list[InsightKPI] = field(default_factory=list)
    trends: list[TrendFinding] = field(default_factory=list)
    anomalies: list[AnomalyFinding] = field(default_factory=list)
    fallback_used: bool = False

    def as_text(self) -> str:
        parts = [self.summary]
        if self.key_findings:
            parts.extend(self.key_findings[:3])
        if self.trends:
            parts.append(self.trends[0].description)
        if self.anomalies:
            parts.append(self.anomalies[0].description)
        return " ".join(part for part in parts if part).strip()
