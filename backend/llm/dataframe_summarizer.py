from __future__ import annotations

import pandas as pd

from backend.llm.insight_models import DataFrameSummary


class DataFrameSummarizer:
    def __init__(self, max_rows: int) -> None:
        self.max_rows = max_rows

    def summarize(self, frame: pd.DataFrame) -> DataFrameSummary:
        numeric_summary: dict[str, dict[str, float]] = {}
        for column in frame.select_dtypes(include="number").columns:
            series = frame[column].dropna()
            if series.empty:
                continue
            numeric_summary[column] = {
                "sum": float(series.sum()),
                "mean": float(series.mean()),
                "min": float(series.min()),
                "max": float(series.max()),
                "std": float(series.std()) if len(series) > 1 else 0.0,
            }

        category_summary: dict[str, list[tuple[str, int]]] = {}
        for column in frame.select_dtypes(exclude="number").columns[:4]:
            counts = frame[column].astype(str).value_counts(dropna=True).head(5)
            category_summary[column] = [(str(index), int(value)) for index, value in counts.items()]

        sample = (
            frame.head(self.max_rows)
            .where(pd.notnull(frame.head(self.max_rows)), None)
            .to_dict(orient="records")
        )
        return DataFrameSummary(
            row_count=len(frame),
            column_count=len(frame.columns),
            numeric_summary=numeric_summary,
            category_summary=category_summary,
            sample_rows=sample,
        )
