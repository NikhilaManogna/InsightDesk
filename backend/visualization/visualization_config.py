from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VisualizationConfig:
    max_chart_rows: int = 5000
    sample_threshold: int = 10000
    category_limit: int = 25
    histogram_bins: int = 30
    max_kpis: int = 4
