from __future__ import annotations

import pandas as pd

from backend.visualization.chart_generator import ChartGenerator
from backend.visualization.chart_selector import ChartSelector
from backend.visualization.dataframe_analyzer import DataFrameAnalyzer
from backend.visualization.visualization_config import VisualizationConfig


def test_detects_time_series_columns() -> None:
    frame = pd.DataFrame(
        {
            "created_at": pd.date_range("2026-01-01", periods=4, freq="D"),
            "revenue": [10, 20, 15, 30],
        }
    )
    profile = DataFrameAnalyzer().analyze(frame)
    assert profile.datetime_columns == ("created_at",)
    assert profile.numeric_columns == ("revenue",)


def test_recommends_line_for_time_series() -> None:
    frame = pd.DataFrame(
        {
            "month": pd.date_range("2026-01-01", periods=6, freq="MS"),
            "total_revenue": [10, 20, 18, 30, 32, 40],
        }
    )
    spec = ChartSelector().select(frame)
    assert spec is not None
    assert spec.kind == "line"


def test_recommends_bar_for_category_metric() -> None:
    frame = pd.DataFrame({"region": ["North", "South"], "revenue": [100, 50]})
    spec = ChartSelector().select(frame)
    assert spec is not None
    assert spec.kind == "bar"


def test_recommends_histogram_for_numeric_distribution() -> None:
    frame = pd.DataFrame({"order_value": list(range(20))})
    spec = ChartSelector().select(frame)
    assert spec is not None
    assert spec.kind == "histogram"


def test_unsupported_shape_returns_table_bundle() -> None:
    frame = pd.DataFrame({"notes": ["a", "b", "c"]})
    bundle = ChartGenerator().build_bundle(frame)
    assert bundle.primary is None
    assert bundle.table is frame
    assert bundle.notes


def test_large_dataset_is_reduced_for_charting() -> None:
    frame = pd.DataFrame(
        {
            "region": ["North", "South"] * 600,
            "revenue": range(1200),
        }
    )
    config = VisualizationConfig(max_chart_rows=100, sample_threshold=500)
    bundle = ChartGenerator(config).build_bundle(frame)
    assert bundle.primary is not None
    assert bundle.metadata["rendered_rows"] == 100
    assert bundle.notes
