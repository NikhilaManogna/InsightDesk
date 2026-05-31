from __future__ import annotations

from dataclasses import dataclass, field


SchemaMap = dict[str, dict[str, str]]


@dataclass(frozen=True)
class Relationship:
    source_table: str
    source_column: str
    target_table: str
    target_column: str

    def render(self) -> str:
        return (
            f"{self.source_table}.{self.source_column} -> "
            f"{self.target_table}.{self.target_column}"
        )


@dataclass(frozen=True)
class TableMetadata:
    name: str
    columns: dict[str, str]
    primary_keys: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SchemaMetadata:
    tables: dict[str, TableMetadata]
    relationships: tuple[Relationship, ...] = field(default_factory=tuple)

    def as_schema_map(self) -> SchemaMap:
        return {name: table.columns for name, table in self.tables.items()}

    @classmethod
    def from_schema_map(cls, schema: SchemaMap) -> "SchemaMetadata":
        return cls(
            tables={
                name: TableMetadata(name=name, columns=columns)
                for name, columns in schema.items()
            }
        )
