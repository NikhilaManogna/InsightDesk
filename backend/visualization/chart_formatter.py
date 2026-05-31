from __future__ import annotations

from plotly.graph_objects import Figure


class ChartFormatter:
    def apply(self, figure: Figure) -> Figure:
        figure.update_layout(
            template="plotly_white",
            margin=dict(l=24, r=24, t=56, b=32),
            hovermode="x unified",
            legend_title_text="",
            font=dict(size=13),
        )
        figure.update_xaxes(showgrid=False, title_standoff=10)
        figure.update_yaxes(gridcolor="rgba(0,0,0,0.08)", title_standoff=10)
        return figure
