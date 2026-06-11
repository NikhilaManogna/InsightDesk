from __future__ import annotations

from plotly.graph_objects import Figure


class ChartFormatter:
    def apply(self, figure: Figure) -> Figure:
        figure.update_layout(
            template="plotly_dark",
            margin=dict(l=32, r=24, t=64, b=42),
            hovermode="x unified",
            legend_title_text="",
            font=dict(size=13, color="#dbeafe"),
            title=dict(font=dict(size=18, color="#f8fafc"), x=0.01, xanchor="left"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#111827",
            colorway=[
                "#4f8cff",
                "#23c7a7",
                "#f59e0b",
                "#e879f9",
                "#38bdf8",
                "#f87171",
            ],
            hoverlabel=dict(bgcolor="#0f172a", bordercolor="rgba(148,163,184,0.35)"),
        )
        figure.update_xaxes(
            showgrid=False,
            title_standoff=12,
            zeroline=False,
            linecolor="rgba(148,163,184,0.22)",
        )
        figure.update_yaxes(
            gridcolor="rgba(148,163,184,0.14)",
            title_standoff=12,
            zeroline=False,
        )
        return figure
