from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp

from backend.security.validation_models import (
    MultiStatementError,
    ValidationCode,
    ValidationIssue,
)


@dataclass(frozen=True)
class ParsedQuery:
    expression: exp.Expression | None
    issues: list[ValidationIssue]


class SQLAstParser:
    def parse(self, sql: str, dialect: str) -> ParsedQuery:
        try:
            statements = sqlglot.parse(sql, read=dialect)
        except Exception as exc:
            return ParsedQuery(
                None,
                [ValidationIssue(ValidationCode.MALFORMED_SQL, f"Could not parse SQL: {exc}")],
            )

        if len(statements) != 1:
            return ParsedQuery(
                None,
                [MultiStatementError("Only one SQL statement is allowed.").issue],
            )
        return ParsedQuery(statements[0], [])
