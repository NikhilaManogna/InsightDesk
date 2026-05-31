from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from backend.db.schema_metadata import Relationship, SchemaMap, SchemaMetadata, TableMetadata
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class SchemaLoader:
    def __init__(self, engine: Engine, dialect: str) -> None:
        self.engine = engine
        self.dialect = dialect

    def load(self) -> SchemaMap:
        return self.load_metadata().as_schema_map()

    def load_metadata(self) -> SchemaMetadata:
        if self.dialect == "duckdb":
            return self._load_duckdb_metadata()
        return self._load_inspector_metadata()

    def _load_inspector_metadata(self) -> SchemaMetadata:
        inspector = inspect(self.engine)
        tables: dict[str, TableMetadata] = {}
        relationships: list[Relationship] = []
        for table_name in inspector.get_table_names():
            columns = inspector.get_columns(table_name)
            primary_key = inspector.get_pk_constraint(table_name).get("constrained_columns") or []
            tables[table_name] = TableMetadata(
                name=table_name,
                columns={col["name"]: str(col["type"]) for col in columns},
                primary_keys=tuple(primary_key),
            )
            for fk in inspector.get_foreign_keys(table_name):
                referred_table = fk.get("referred_table")
                constrained = fk.get("constrained_columns") or []
                referred = fk.get("referred_columns") or []
                if not referred_table:
                    continue
                for source_col, target_col in zip(constrained, referred):
                    relationships.append(
                        Relationship(table_name, source_col, referred_table, target_col)
                    )
        logger.info(
            "schema_loaded dialect=%s tables=%s relationships=%s",
            self.dialect,
            len(tables),
            len(relationships),
        )
        return SchemaMetadata(tables=tables, relationships=tuple(relationships))

    def _load_duckdb_metadata(self) -> SchemaMetadata:
        query = """
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
        ORDER BY table_name, ordinal_position
        """
        schema: SchemaMap = {}
        with self.engine.connect() as conn:
            for row in conn.execute(text(query)).mappings():
                schema.setdefault(row["table_name"], {})[row["column_name"]] = row["data_type"]
        relationships = self._infer_relationships(schema)
        logger.info("schema_loaded dialect=duckdb tables=%s relationships=%s", len(schema), len(relationships))
        return SchemaMetadata(
            tables={
                name: TableMetadata(
                    name=name,
                    columns=columns,
                    primary_keys=tuple(self._guess_primary_keys(name, columns)),
                )
                for name, columns in schema.items()
            },
            relationships=tuple(relationships),
        )

    def _infer_relationships(self, schema: SchemaMap) -> list[Relationship]:
        relationships: list[Relationship] = []
        for table_name, columns in schema.items():
            for column in columns:
                if not column.lower().endswith("_id"):
                    continue
                target: tuple[str, str] | None = None
                prefix = column[:-3]
                for candidate_table, candidate_columns in schema.items():
                    if candidate_table == table_name:
                        continue
                    names = {candidate_table.lower(), candidate_table.lower().rstrip("s")}
                    if prefix.lower() in names:
                        target_key = "id" if "id" in {c.lower() for c in candidate_columns} else None
                        if target_key:
                            target = (candidate_table, self._actual_column(candidate_columns, target_key))
                            break
                if target and target[0] != table_name:
                    relationships.append(Relationship(table_name, column, target[0], target[1]))
        return relationships

    @staticmethod
    def _guess_primary_keys(table_name: str, columns: dict[str, str]) -> list[str]:
        lower = {column.lower(): column for column in columns}
        candidates = ["id", f"{table_name.rstrip('s')}_id", f"{table_name}_id"]
        return [lower[name] for name in candidates if name in lower]

    @staticmethod
    def _actual_column(columns: dict[str, str], column: str) -> str:
        for name in columns:
            if name.lower() == column.lower():
                return name
        return column
