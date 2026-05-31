from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from plotly.graph_objects import Figure


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    dtype: str
    role: str
    non_null: int
    unique: int


@dataclass(frozen=True)
class DataFrameProfile:
    row_count: int
    column_count: int
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    datetime_columns: tuple[str, ...]
    boolean_columns: tuple[str, ...]
    percentage_columns: tuple[str, ...]
    columns: tuple[ColumnProfile, ...]

    @property
    def is_empty(self) -> bool:
        return self.row_count == 0 or self.column_count == 0


@dataclass(frozen=True)
class ChartSpec:
    kind: str
    score: int
    title: str
    x: str | None = None
    y: str | None = None
    color: str | None = None
    reason: str = ""


@dataclass(frozen=True)
class KPI:
    label: str
    value: str
    help_text: str | None = None


@dataclass
class VisualizationBundle:
    primary: Figure | None
    supporting: list[Figure] = field(default_factory=list)
    kpis: list[KPI] = field(default_factory=list)
    table: pd.DataFrame | None = None
    selected: ChartSpec | None = None
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
