from __future__ import annotations

import re

from backend.llm.prompts import (
    SQL_REPAIR_PROMPT,
    SQL_SYSTEM_PROMPT,
    build_sql_prompt,
    build_sql_repair_prompt,
)
from backend.db.schema_metadata import SchemaMap, SchemaMetadata
from backend.llm.ambiguity import AmbiguousQuestionError, detect_ambiguity
from backend.llm.business_aliases import parse_business_aliases
from backend.llm.schema_context import SchemaContextBuilder
from backend.llm.providers.base import LLMRequest
from backend.llm.providers.factory import build_llm_provider
from backend.utils.config import Settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class SQLGenerationError(RuntimeError):
    """Raised when the selected LLM cannot produce a usable SQL string."""


class SQLGenerator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.provider = build_llm_provider(settings)
        aliases = parse_business_aliases(settings.business_aliases)
        self.schema_context = SchemaContextBuilder(
            max_tables=settings.prompt_max_tables,
            use_rag=settings.schema_rag_enabled,
            embedding_model=settings.embedding_model,
            aliases=aliases,
        )

    def generate(self, question: str, schema: SchemaMap | SchemaMetadata, dialect: str) -> str:
        clarification = detect_ambiguity(question)
        if clarification:
            logger.info("ambiguous_question question=%s", question)
            raise AmbiguousQuestionError(clarification)

        context = self.schema_context.build(question, schema)
        prompt = build_sql_prompt(question, context.schema, dialect, context.relationships)
        logger.info(
            "sql_prompt_generated provider=%s dialect=%s tables=%s",
            self.provider.name,
            dialect,
            ",".join(context.selected_tables),
        )
        request = LLMRequest(
            system_prompt=SQL_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=self.settings.llm_temperature,
            max_tokens=1200,
        )
        sql = self._complete_sql(request)
        if not self._looks_like_read_query(sql):
            logger.warning("llm_invalid_sql_shape provider=%s", self.provider.name)
            sql = self.repair(
                question,
                schema,
                dialect,
                sql,
                ["SQL must start with SELECT or WITH."],
            )
        return sql

    def repair(
        self,
        question: str,
        schema: SchemaMap | SchemaMetadata,
        dialect: str,
        bad_sql: str,
        errors: list[str],
    ) -> str:
        context = self.schema_context.build(question, schema)
        prompt = build_sql_repair_prompt(
            question,
            context.schema,
            dialect,
            bad_sql,
            errors,
            context.relationships,
        )
        logger.info(
            "sql_repair_prompt_generated provider=%s dialect=%s errors=%s",
            self.provider.name,
            dialect,
            " | ".join(errors),
        )
        request = LLMRequest(
            system_prompt=SQL_REPAIR_PROMPT,
            user_prompt=prompt,
            temperature=self.settings.llm_temperature,
            max_tokens=1200,
        )
        return self._complete_sql(request)

    def _complete_sql(self, request: LLMRequest) -> str:
        try:
            text = self.provider.complete(request)
        except Exception as exc:
            logger.exception("sql_generation_failed provider=%s", self.provider.name)
            raise SQLGenerationError(str(exc)) from exc

        sql = self._clean_sql(text)
        if not sql:
            raise SQLGenerationError("LLM returned an empty SQL response.")
        return sql

    @staticmethod
    def _clean_sql(text: str) -> str:
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:sql)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
        return cleaned.rstrip(";")

    @staticmethod
    def _looks_like_read_query(sql: str) -> bool:
        return sql.lstrip().upper().startswith(("SELECT", "WITH"))


# Backward-compatible name so the Streamlit app does not need to know which
# provider is active.
GeminiSQLGenerator = SQLGenerator
