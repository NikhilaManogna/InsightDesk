from __future__ import annotations

from sqlglot import exp

from backend.utils.logger import get_logger

logger = get_logger(__name__)


class LimitInjector:
    def __init__(self, max_limit: int) -> None:
        self.max_limit = max_limit

    def apply(self, parsed: exp.Expression, dialect: str) -> str:
        current_limit = parsed.args.get("limit")
        if current_limit:
            parsed_limit = self._limit_value(current_limit)
            if parsed_limit is None or parsed_limit > self.max_limit:
                current_limit.set("expression", exp.Literal.number(self.max_limit))
                logger.info("limit_clamped max_limit=%s", self.max_limit)
            return parsed.sql(dialect=dialect, pretty=True)

        parsed.set("limit", exp.Limit(expression=exp.Literal.number(self.max_limit)))
        logger.info("limit_injected max_limit=%s", self.max_limit)
        return parsed.sql(dialect=dialect, pretty=True)

    @staticmethod
    def _limit_value(limit: exp.Expression) -> int | None:
        expression = limit.args.get("expression")
        if isinstance(expression, exp.Literal) and expression.is_number:
            try:
                return int(expression.this)
            except ValueError:
                return None
        return None
