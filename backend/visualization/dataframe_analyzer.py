from __future__ import annotations

import pandas as pd

from backend.visualization.chart_models import ColumnProfile, DataFrameProfile


class DataFrameAnalyzer:
    """Profiles a query result without doing expensive transformations."""

    def analyze(self, frame: pd.DataFrame) -> DataFrameProfile:
        numeric: list[str] = []
        categorical: list[str] = []
        datetimes: list[str] = []
        booleans: list[str] = []
        percentages: list[str] = []
        profiles: list[ColumnProfile] = []

        for column in frame.columns:
            series = frame[column]
            role = "categorical"
            if pd.api.types.is_bool_dtype(series):
                booleans.append(column)
                role = "boolean"
            elif pd.api.types.is_datetime64_any_dtype(series):
                datetimes.append(column)
                role = "datetime"
            elif pd.api.types.is_numeric_dtype(series):
                numeric.append(column)
                role = "numeric"
                if self._looks_like_percentage(column, series):
                    percentages.append(column)
            else:
                parsed = self._maybe_datetime(series)
                if parsed is not None:
                    datetimes.append(column)
                    role = "datetime"
                else:
                    categorical.append(column)

            profiles.append(
                ColumnProfile(
                    name=column,
                    dtype=str(series.dtype),
                    role=role,
                    non_null=int(series.notna().sum()),
                    unique=int(series.nunique(dropna=True)),
                )
            )

        return DataFrameProfile(
            row_count=len(frame),
            column_count=len(frame.columns),
            numeric_columns=tuple(numeric),
            categorical_columns=tuple(categorical),
            datetime_columns=tuple(datetimes),
            boolean_columns=tuple(booleans),
            percentage_columns=tuple(percentages),
            columns=tuple(profiles),
        )

    @staticmethod
    def _looks_like_percentage(column: str, series: pd.Series) -> bool:
        name = column.lower()
        if any(token in name for token in ("percent", "percentage", "rate", "ratio", "pct")):
            return True
        clean = series.dropna()
        return not clean.empty and clean.between(0, 1).all() and clean.nunique() > 2

    @staticmethod
    def _maybe_datetime(series: pd.Series) -> pd.Series | None:
        if series.empty or series.dtype != "object":
            return None
        sample = series.dropna().head(50)
        if sample.empty:
            return None
        parsed = pd.to_datetime(sample, errors="coerce")
        if parsed.notna().mean() >= 0.8:
            return parsed
        return None
