from __future__ import annotations

import pandas as pd

from backend.llm.anomaly_detector import AnomalyDetector
from backend.llm.dataframe_summarizer import DataFrameSummarizer
from backend.llm.kpi_extractor import InsightKPIExtractor
from backend.llm.trend_detector import TrendDetector


def test_dataframe_summarizer_reduces_rows() -> None:
    frame = pd.DataFrame({"region": ["North", "South", "North"], "revenue": [10, 20, 30]})
    summary = DataFrameSummarizer(max_rows=2).summarize(frame)
    assert summary.row_count == 3
    assert len(summary.sample_rows) == 2
    assert summary.numeric_summary["revenue"]["sum"] == 60


def test_kpi_extractor_finds_totals() -> None:
    frame = pd.DataFrame({"total_revenue": [100, 200, 50]})
    kpis = InsightKPIExtractor().extract(frame)
    assert kpis[0].label == "Records"
    assert any("Revenue" in kpi.label for kpi in kpis)


def test_trend_detector_identifies_upward_trend() -> None:
    frame = pd.DataFrame(
        {
            "day": pd.date_range("2026-01-01", periods=4, freq="D"),
            "revenue": [10, 15, 25, 40],
        }
    )
    trends = TrendDetector().detect(frame)
    assert trends
    assert trends[0].direction == "upward"


def test_anomaly_detector_flags_outlier() -> None:
    frame = pd.DataFrame({"revenue": [10, 11, 12, 13, 100]})
    anomalies = AnomalyDetector(sensitivity=1.5).detect(frame)
    assert anomalies
    assert anomalies[0].column == "revenue"


def test_anomaly_detector_handles_flat_data() -> None:
    frame = pd.DataFrame({"revenue": [10, 10, 10, 10]})
    assert AnomalyDetector().detect(frame) == []
