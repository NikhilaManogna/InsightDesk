from __future__ import annotations

import pandas as pd

from backend.visualization.chart_models import ChartSpec
from backend.visualization.chart_rules import ChartRules
from backend.visualization.dataframe_analyzer import DataFrameAnalyzer
from backend.visualization.visualization_config import VisualizationConfig
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class ChartSelector:
    def __init__(self, config: VisualizationConfig | None = None) -> None:
        self.config = config or VisualizationConfig()
        self.analyzer = DataFrameAnalyzer()
        self.rules = ChartRules()

    def select(self, frame: pd.DataFrame) -> ChartSpec | None:
        recommendations = self.recommend(frame)
        return recommendations[0] if recommendations else None

    def recommend(self, frame: pd.DataFrame) -> list[ChartSpec]:
        profile = self.analyzer.analyze(frame)
        specs = self.rules.recommend(profile)
        if specs:
            logger.info(
                "chart_recommendation selected=%s score=%s rows=%s columns=%s",
                specs[0].kind,
                specs[0].score,
                profile.row_count,
                profile.column_count,
            )
        else:
            logger.info(
                "chart_recommendation unsupported rows=%s columns=%s",
                profile.row_count,
                profile.column_count,
            )
        return specs
