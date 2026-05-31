from __future__ import annotations

from backend.visualization.chart_models import ChartSpec, DataFrameProfile


class ChartRules:
    def recommend(self, profile: DataFrameProfile) -> list[ChartSpec]:
        if profile.is_empty:
            return []

        specs: list[ChartSpec] = []
        numeric = list(profile.numeric_columns)
        categorical = list(profile.categorical_columns)
        datetimes = list(profile.datetime_columns)

        if datetimes and numeric:
            specs.append(
                ChartSpec(
                    "line",
                    95,
                    f"{self._label(numeric[0])} over time",
                    x=datetimes[0],
                    y=numeric[0],
                    reason="time-series trend",
                )
            )
            if profile.row_count >= 4:
                specs.append(
                    ChartSpec(
                        "area",
                        72,
                        f"Cumulative {self._label(numeric[0])}",
                        x=datetimes[0],
                        y=numeric[0],
                        reason="cumulative time-series view",
                    )
                )

        if categorical and numeric:
            category_cardinality = self._cardinality(profile, categorical[0])
            if category_cardinality <= 30:
                specs.append(
                    ChartSpec(
                        "bar",
                        88,
                        f"{self._label(numeric[0])} by {self._label(categorical[0])}",
                        x=categorical[0],
                        y=numeric[0],
                        reason="grouped comparison",
                    )
                )
            if len(categorical) >= 2 and category_cardinality <= 20:
                specs.append(
                    ChartSpec(
                        "stacked_bar",
                        75,
                        f"{self._label(numeric[0])} by {self._label(categorical[0])} and {self._label(categorical[1])}",
                        x=categorical[0],
                        y=numeric[0],
                        color=categorical[1],
                        reason="segmented comparison",
                    )
                )
            if category_cardinality <= 8 and self._looks_like_breakdown(profile, numeric[0]):
                specs.append(
                    ChartSpec(
                        "donut",
                        68,
                        f"{self._label(numeric[0])} share by {self._label(categorical[0])}",
                        x=categorical[0],
                        y=numeric[0],
                        reason="small percentage breakdown",
                    )
                )

        if len(numeric) >= 2 and profile.row_count >= 5:
            specs.append(
                ChartSpec(
                    "scatter",
                    70,
                    f"{self._label(numeric[1])} vs {self._label(numeric[0])}",
                    x=numeric[0],
                    y=numeric[1],
                    color=categorical[0] if categorical and self._cardinality(profile, categorical[0]) <= 12 else None,
                    reason="numeric relationship",
                )
            )

        if len(numeric) == 1 and not categorical and not datetimes and profile.row_count >= 8:
            specs.append(
                ChartSpec(
                    "histogram",
                    76,
                    f"Distribution of {self._label(numeric[0])}",
                    x=numeric[0],
                    reason="single numeric distribution",
                )
            )

        return sorted(specs, key=lambda spec: spec.score, reverse=True)

    @staticmethod
    def _cardinality(profile: DataFrameProfile, column: str) -> int:
        for item in profile.columns:
            if item.name == column:
                return item.unique
        return 0

    @staticmethod
    def _looks_like_breakdown(profile: DataFrameProfile, column: str) -> bool:
        return column in profile.percentage_columns or profile.row_count <= 12

    @staticmethod
    def _label(column: str) -> str:
        return column.replace("_", " ").title()
