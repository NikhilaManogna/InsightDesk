from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SchemaDocument:
    id: str
    table: str
    text: str
    kind: str = "table"


@dataclass(frozen=True)
class RetrievalResult:
    document: SchemaDocument
    score: float
