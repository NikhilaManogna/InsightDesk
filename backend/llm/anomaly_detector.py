from __future__ import annotations

import pandas as pd

from backend.llm.insight_models import AnomalyFinding


class AnomalyDetector:
    def __init__(self, sensitivity: float = 2.0) -> None:
        self.sensitivity = sensitivity

    def detect(self, frame: pd.DataFrame) -> list[AnomalyFinding]:
        findings: list[AnomalyFinding] = []
        for column in frame.select_dtypes(include="number").columns:
            series = frame[column].dropna()
            if len(series) < 4:
                continue
            std = series.std()
            if std == 0:
                continue
            mean = series.mean()
            z_scores = ((series - mean).abs() / std).sort_values(ascending=False)
            top_index = z_scores.index[0]
            if z_scores.iloc[0] >= self.sensitivity:
                value = series.loc[top_index]
                findings.append(
                    AnomalyFinding(
                        column=column,
                        value=f"{value:,.2f}",
                        description=(
                            f"{column} has an unusual value of {value:,.2f}, "
                            f"well away from the average of {mean:,.2f}."
                        ),
                    )
                )
        return findings[:3]
