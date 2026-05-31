from __future__ import annotations

from dataclasses import dataclass

from sqlglot import exp

from backend.db.schema_metadata import SchemaMap, SchemaMetadata
from backend.security.ast_parser import SQLAstParser
from backend.security.limit_injector import LimitInjector
from backend.security.query_sanitizer import QuerySanitizer
from backend.security.schema_checker import SchemaChecker
from backend.security.validation_models import (
    DangerousQueryError,
    ValidationCode,
    ValidationIssue,
)


BLOCKED_KEYWORDS = {
    "DELETE",
    "DROP",
    "INSERT",
    "UPDATE",
    "ALTER",
    "TRUNCATE",
    "CREATE",
    "COPY",
    "MERGE",
    "GRANT",
    "REVOKE",
    "CALL",
    "EXECUTE",
    "VACUUM",
    "ANALYZE",
}


@dataclass(frozen=True)
class ValidationResult:
    sql: str
    valid: bool
    errors: list[str]
    issues: list[ValidationIssue]


class SQLValidator:
    def __init__(self, schema: SchemaMap | SchemaMetadata, limit: int) -> None:
        self.schema_checker = SchemaChecker(schema)
        self.limit = limit
        self.parser = SQLAstParser()
        self.sanitizer = QuerySanitizer()
        self.limit_injector = LimitInjector(limit)

    def validate(self, sql: str, dialect: str) -> ValidationResult:
        cleaned = sql.strip().rstrip(";")
        issues: list[ValidationIssue] = []

        if not cleaned:
            issues.append(ValidationIssue(ValidationCode.MALFORMED_SQL, "SQL is empty."))
            return self._result(cleaned, issues)

        issues.extend(self.sanitizer.validate_raw_sql_text(sql))
        parsed_query = self.parser.parse(cleaned, dialect)
        issues.extend(parsed_query.issues)
        if parsed_query.expression is None:
            return self._result(cleaned, issues)

        parsed = parsed_query.expression
        dangerous_issue = self._dangerous_issue(parsed)
        if dangerous_issue:
            issues.append(dangerous_issue)
            return self._result(cleaned, issues)

        if parsed.find(exp.Command):
            issues.append(
                DangerousQueryError("Database commands are not allowed.").issue
            )

        schema_issues = self.schema_checker.validate(cleaned, dialect)
        issues.extend(schema_issues)

        safe_sql = cleaned
        if not issues:
            safe_sql = self.limit_injector.apply(parsed, dialect)

        return self._result(safe_sql, issues)

    def _dangerous_issue(self, parsed: exp.Expression) -> ValidationIssue | None:
        class_name = parsed.__class__.__name__.upper()
        if class_name in BLOCKED_KEYWORDS:
            return DangerousQueryError(f"`{class_name}` is not allowed.").issue
        if not isinstance(parsed, exp.Select):
            return DangerousQueryError("Only SELECT and WITH queries are allowed.").issue

        for node in parsed.walk():
            node_name = node.__class__.__name__.upper()
            if node_name in BLOCKED_KEYWORDS:
                return DangerousQueryError(f"`{node_name}` is not allowed.").issue
        return None

    @staticmethod
    def _result(sql: str, issues: list[ValidationIssue]) -> ValidationResult:
        return ValidationResult(
            sql=sql,
            valid=not issues,
            errors=[issue.message for issue in issues],
            issues=issues,
        )
