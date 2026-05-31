from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ValidationCode(str, Enum):
    DANGEROUS_QUERY = "dangerous_query"
    INVALID_COLUMN = "invalid_column"
    INVALID_JOIN = "invalid_join"
    INVALID_TABLE = "invalid_table"
    MALFORMED_SQL = "malformed_sql"
    MULTI_STATEMENT = "multi_statement"
    PROMPT_INJECTION = "prompt_injection"
    TIMEOUT = "timeout"
    UNSUPPORTED_SQL = "unsupported_sql"


@dataclass(frozen=True)
class ValidationIssue:
    code: ValidationCode
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ValidationFailure(ValueError):
    code = ValidationCode.UNSUPPORTED_SQL

    def __init__(self, message: str, **metadata: Any) -> None:
        super().__init__(message)
        self.issue = ValidationIssue(self.code, message, metadata)


class InvalidTableError(ValidationFailure):
    code = ValidationCode.INVALID_TABLE


class InvalidColumnError(ValidationFailure):
    code = ValidationCode.INVALID_COLUMN


class DangerousQueryError(ValidationFailure):
    code = ValidationCode.DANGEROUS_QUERY


class MultiStatementError(ValidationFailure):
    code = ValidationCode.MULTI_STATEMENT


class SQLTimeoutError(ValidationFailure):
    code = ValidationCode.TIMEOUT
