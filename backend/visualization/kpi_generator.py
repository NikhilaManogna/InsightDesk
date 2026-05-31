from __future__ import annotations

import pandas as pd

from backend.visualization.chart_models import DataFrameProfile, KPI
from backend.visualization.visualization_config import VisualizationConfig


class KPIGenerator:
    def __init__(self, config: VisualizationConfig) -> None:
        self.config = config

    def build(self, frame: pd.DataFrame, profile: DataFrameProfile) -> list[KPI]:
        kpis = [KPI("Records", f"{profile.row_count:,}")]
        for column in profile.numeric_columns[: self.config.max_kpis - 1]:
            series = frame[column].dropna()
            if series.empty:
                continue
            if self._is_count_like(column):
                value = f"{series.sum():,.0f}"
                base_label = self._label(column)
                label = base_label if base_label.lower().startswith("total ") else f"Total {base_label}"
            else:
                value = f"{series.mean():,.2f}"
                label = f"Avg {self._label(column)}"
            kpis.append(KPI(label, value, f"Based on non-null values in `{column}`."))
        return kpis[: self.config.max_kpis]

    @staticmethod
    def _is_count_like(column: str) -> bool:
        name = column.lower()
        return any(token in name for token in ("count", "qty", "quantity", "total", "revenue", "sales", "amount"))

    @staticmethod
    def _label(column: str) -> str:
        return column.replace("_", " ").title()
