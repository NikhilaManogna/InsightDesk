from __future__ import annotations

import pandas as pd

from backend.llm.insight_models import TrendFinding


class TrendDetector:
    def detect(self, frame: pd.DataFrame) -> list[TrendFinding]:
        datetime_cols = [
            col for col in frame.columns if pd.api.types.is_datetime64_any_dtype(frame[col])
        ]
        if not datetime_cols:
            for col in frame.select_dtypes(include="object").columns:
                parsed = pd.to_datetime(frame[col], errors="coerce")
                if parsed.notna().mean() >= 0.8:
                    frame = frame.copy()
                    frame[col] = parsed
                    datetime_cols.append(col)
                    break

        numeric_cols = list(frame.select_dtypes(include="number").columns)
        if not datetime_cols or not numeric_cols or len(frame) < 3:
            return []

        time_col = datetime_cols[0]
        metric = numeric_cols[0]
        ordered = frame[[time_col, metric]].dropna().sort_values(time_col)
        if len(ordered) < 3:
            return []

        first = float(ordered[metric].iloc[0])
        last = float(ordered[metric].iloc[-1])
        if first == 0:
            change = None
        else:
            change = ((last - first) / abs(first)) * 100
        direction = "upward" if last > first else "downward" if last < first else "flat"
        if change is None:
            detail = f"{metric} moved from {first:,.2f} to {last:,.2f}."
        else:
            detail = f"{metric} shows a {direction} trend of {change:,.1f}% across the result window."
        return [TrendFinding(metric, direction, change, detail)]
