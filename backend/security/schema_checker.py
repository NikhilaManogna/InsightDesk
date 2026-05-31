from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp

from backend.db.schema_metadata import SchemaMap, SchemaMetadata
from backend.security.validation_models import (
    InvalidColumnError,
    InvalidTableError,
    ValidationCode,
    ValidationIssue,
)


@dataclass(frozen=True)
class SchemaIssue:
    message: str


class SchemaChecker:
    def __init__(self, schema: SchemaMap | SchemaMetadata) -> None:
        self.metadata = (
            schema
            if isinstance(schema, SchemaMetadata)
            else SchemaMetadata.from_schema_map(schema)
        )
        self.schema = self.metadata.as_schema_map()
        self.tables = {name.lower(): name for name in self.schema}

    def validate(self, sql: str, dialect: str) -> list[ValidationIssue]:
        try:
            parsed = sqlglot.parse_one(sql, read=dialect)
        except Exception as exc:
            return [ValidationIssue(ValidationCode.MALFORMED_SQL, f"SQL parse failed: {exc}")]

        issues: list[ValidationIssue] = []
        used_tables = self._tables(parsed)
        for table in used_tables:
            if table.lower() not in self.tables:
                issues.append(InvalidTableError(f"Unknown table `{table}`.", table=table).issue)

        aliases = self._table_aliases(parsed)
        physical_tables = [self.tables[t.lower()] for t in used_tables if t.lower() in self.tables]
        for column in parsed.find_all(exp.Column):
            name = column.name
            qualifier = column.table
            if not name or name == "*":
                continue
            if qualifier:
                table_name = aliases.get(qualifier.lower())
                if not table_name:
                    issues.append(
                        InvalidColumnError(
                            f"Unknown table alias `{qualifier}`.",
                            alias=qualifier,
                            column=name,
                        ).issue
                    )
                elif name.lower() not in {c.lower() for c in self.schema[table_name]}:
                    issues.append(
                        InvalidColumnError(
                            f"Unknown column `{qualifier}.{name}`.",
                            table=table_name,
                            column=name,
                        ).issue
                    )

        if len(set(physical_tables)) == 1:
            valid_columns = {c.lower() for c in self.schema[physical_tables[0]]}
            for column in parsed.find_all(exp.Column):
                if column.table:
                    continue
                name = column.name
                if name and name.lower() not in valid_columns:
                    issues.append(
                        InvalidColumnError(
                            f"Unknown column `{name}` for `{physical_tables[0]}`.",
                            table=physical_tables[0],
                            column=name,
                        ).issue
                    )
        issues.extend(self._validate_joins(parsed, aliases))
        issues.extend(self._validate_group_by(parsed))
        return issues

    @staticmethod
    def _tables(parsed: exp.Expression) -> set[str]:
        cte_names = {cte.alias_or_name for cte in parsed.find_all(exp.CTE)}
        tables = {table.name for table in parsed.find_all(exp.Table)}
        return {name for name in tables if name not in cte_names}

    def _table_aliases(self, parsed: exp.Expression) -> dict[str, str]:
        cte_names = {cte.alias_or_name for cte in parsed.find_all(exp.CTE)}
        aliases: dict[str, str] = {}
        for table in parsed.find_all(exp.Table):
            table_name = table.name
            if table_name in cte_names or table_name.lower() not in self.tables:
                continue
            physical_name = self.tables[table_name.lower()]
            aliases[table_name.lower()] = physical_name
            alias = table.alias
            if alias:
                aliases[alias.lower()] = physical_name
        return aliases

    def _validate_joins(
        self,
        parsed: exp.Expression,
        aliases: dict[str, str],
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        relationship_pairs = {
            (
                rel.source_table.lower(),
                rel.source_column.lower(),
                rel.target_table.lower(),
                rel.target_column.lower(),
            )
            for rel in self.metadata.relationships
        }
        relationship_pairs |= {
            (target_table, target_column, source_table, source_column)
            for source_table, source_column, target_table, target_column in relationship_pairs
        }

        for join in parsed.find_all(exp.Join):
            on_expr = join.args.get("on")
            if on_expr is None:
                issues.append(
                    ValidationIssue(
                        ValidationCode.INVALID_JOIN,
                        "JOIN must include an ON condition.",
                    )
                )
                continue

            comparisons = list(on_expr.find_all(exp.EQ))
            if not comparisons:
                issues.append(
                    ValidationIssue(
                        ValidationCode.INVALID_JOIN,
                        "JOIN condition must compare columns with equality.",
                    )
                )
                continue

            valid_relationship = False
            for comparison in comparisons:
                left = comparison.left
                right = comparison.right
                if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
                    continue
                left_table = aliases.get((left.table or "").lower())
                right_table = aliases.get((right.table or "").lower())
                if not left_table or not right_table:
                    continue
                pair = (
                    left_table.lower(),
                    left.name.lower(),
                    right_table.lower(),
                    right.name.lower(),
                )
                if not relationship_pairs or pair in relationship_pairs:
                    valid_relationship = True
                    break

            if not valid_relationship:
                issues.append(
                    ValidationIssue(
                        ValidationCode.INVALID_JOIN,
                        "JOIN does not match a known table relationship.",
                    )
                )
        return issues

    def _validate_group_by(self, parsed: exp.Expression) -> list[ValidationIssue]:
        group = parsed.args.get("group")
        if not group:
            return []

        group_columns = {
            column.sql(dialect="").lower()
            for column in group.find_all(exp.Column)
        }
        issues: list[ValidationIssue] = []
        for projection in parsed.expressions:
            if projection.find(exp.AggFunc):
                continue
            column = projection.find(exp.Column)
            if column and column.sql(dialect="").lower() not in group_columns:
                issues.append(
                    ValidationIssue(
                        ValidationCode.INVALID_COLUMN,
                        f"Column `{column.sql()}` must appear in GROUP BY or be aggregated.",
                        {"column": column.sql()},
                    )
                )
        return issues
