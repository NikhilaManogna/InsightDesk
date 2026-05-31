from __future__ import annotations

import pandas as pd

from backend.llm.insight_models import InsightKPI


class InsightKPIExtractor:
    def __init__(self, max_kpis: int = 4) -> None:
        self.max_kpis = max_kpis

    def extract(self, frame: pd.DataFrame) -> list[InsightKPI]:
        kpis = [InsightKPI("Records", f"{len(frame):,}")]
        for column in frame.select_dtypes(include="number").columns:
            series = frame[column].dropna()
            if series.empty:
                continue
            label = self._label(column)
            if self._is_total_metric(column):
                kpis.append(InsightKPI(f"Total {label}", f"{series.sum():,.2f}"))
            else:
                kpis.append(InsightKPI(f"Avg {label}", f"{series.mean():,.2f}"))
            if len(kpis) >= self.max_kpis:
                break
        return kpis

    @staticmethod
    def _is_total_metric(column: str) -> bool:
        name = column.lower()
        return any(token in name for token in ("revenue", "sales", "amount", "total", "count", "qty"))

    @staticmethod
    def _label(column: str) -> str:
        return column.replace("_", " ").title()
