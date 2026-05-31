from __future__ import annotations

from time import perf_counter

import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure

from backend.visualization.chart_formatter import ChartFormatter
from backend.visualization.chart_models import ChartSpec, VisualizationBundle
from backend.visualization.chart_selector import ChartSelector
from backend.visualization.dataframe_analyzer import DataFrameAnalyzer
from backend.visualization.kpi_generator import KPIGenerator
from backend.visualization.visualization_config import VisualizationConfig
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class ChartGenerator:
    def __init__(self, config: VisualizationConfig | None = None) -> None:
        self.config = config or VisualizationConfig()
        self.selector = ChartSelector(self.config)
        self.analyzer = DataFrameAnalyzer()
        self.kpis = KPIGenerator(self.config)
        self.formatter = ChartFormatter()

    def build(self, frame: pd.DataFrame) -> Figure | None:
        return self.build_bundle(frame).primary

    def build_bundle(self, frame: pd.DataFrame) -> VisualizationBundle:
        started = perf_counter()
        profile = self.analyzer.analyze(frame)
        notes: list[str] = []
        chart_frame = self._prepare_frame(frame, notes)
        recommendations = self.selector.recommend(chart_frame)
        kpis = self.kpis.build(frame, profile)

        if not recommendations:
            return VisualizationBundle(
                primary=None,
                kpis=kpis,
                table=frame,
                notes=notes or ["No reliable chart recommendation for this result shape."],
                metadata={"row_count": profile.row_count, "column_count": profile.column_count},
            )

        figures: list[Figure] = []
        for spec in recommendations[:2]:
            try:
                figures.append(self._render(chart_frame, spec))
            except Exception:
                logger.exception("chart_render_failed kind=%s", spec.kind)

        elapsed_ms = int((perf_counter() - started) * 1000)
        logger.info(
            "visualization_built primary=%s supporting=%s rows=%s latency_ms=%s",
            recommendations[0].kind,
            max(0, len(figures) - 1),
            len(chart_frame),
            elapsed_ms,
        )
        return VisualizationBundle(
            primary=figures[0] if figures else None,
            supporting=figures[1:],
            kpis=kpis,
            table=frame,
            selected=recommendations[0],
            notes=notes,
            metadata={"row_count": profile.row_count, "rendered_rows": len(chart_frame)},
        )

    def _prepare_frame(self, frame: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
        if len(frame) <= self.config.max_chart_rows:
            return self._coerce_dates(frame.copy())
        if len(frame) <= self.config.sample_threshold:
            notes.append(
                f"Chart rendered the first {self.config.max_chart_rows:,} of {len(frame):,} rows."
            )
            logger.info(
                "visualization_trimmed original_rows=%s rendered_rows=%s",
                len(frame),
                self.config.max_chart_rows,
            )
            return self._coerce_dates(frame.head(self.config.max_chart_rows).copy())
        sampled = frame.sample(
            n=self.config.max_chart_rows,
            random_state=7,
        ).sort_index()
        notes.append(
            f"Chart sampled {self.config.max_chart_rows:,} of {len(frame):,} rows for browser performance."
        )
        logger.info(
            "visualization_sampled original_rows=%s rendered_rows=%s",
            len(frame),
            len(sampled),
        )
        return self._coerce_dates(sampled)

    @staticmethod
    def _coerce_dates(frame: pd.DataFrame) -> pd.DataFrame:
        for column in frame.columns:
            if frame[column].dtype == "object":
                sample = frame[column].dropna().head(50)
                if sample.empty:
                    continue
                parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().mean() >= 0.8:
                    frame[column] = pd.to_datetime(frame[column], errors="coerce")
        return frame

    def _render(self, frame: pd.DataFrame, spec: ChartSpec) -> Figure:
        if spec.kind == "line":
            data = frame.sort_values(spec.x)
            figure = px.line(data, x=spec.x, y=spec.y, title=spec.title, markers=True)
        elif spec.kind == "area":
            data = frame.sort_values(spec.x).copy()
            data[spec.y] = data[spec.y].cumsum()
            figure = px.area(data, x=spec.x, y=spec.y, title=spec.title)
        elif spec.kind == "bar":
            data = self._top_categories(frame, spec)
            figure = px.bar(data, x=spec.x, y=spec.y, title=spec.title)
        elif spec.kind == "stacked_bar":
            data = self._top_categories(frame, spec)
            figure = px.bar(data, x=spec.x, y=spec.y, color=spec.color, title=spec.title)
        elif spec.kind == "donut":
            data = self._top_categories(frame, spec)
            figure = px.pie(data, names=spec.x, values=spec.y, hole=0.45, title=spec.title)
        elif spec.kind == "histogram":
            figure = px.histogram(
                frame,
                x=spec.x,
                nbins=self.config.histogram_bins,
                title=spec.title,
            )
        else:
            figure = px.scatter(
                frame,
                x=spec.x,
                y=spec.y,
                color=spec.color,
                title=spec.title,
            )
        return self.formatter.apply(figure)

    def _top_categories(self, frame: pd.DataFrame, spec: ChartSpec) -> pd.DataFrame:
        if not spec.x or not spec.y:
            return frame
        grouped = frame
        if spec.x in frame.columns and spec.y in frame.columns:
            group_cols = [spec.x] + ([spec.color] if spec.color else [])
            grouped = (
                frame.groupby(group_cols, dropna=False, as_index=False)[spec.y]
                .sum()
                .sort_values(spec.y, ascending=False)
                .head(self.config.category_limit)
            )
        return grouped
