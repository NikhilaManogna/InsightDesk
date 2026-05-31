from __future__ import annotations

import re
from dataclasses import dataclass

from backend.db.schema_metadata import Relationship, SchemaMap, SchemaMetadata
from backend.llm.business_aliases import expand_question
from backend.rag.schema_retriever import SchemaRetriever
from backend.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PromptSchemaContext:
    schema: SchemaMap
    relationships: tuple[Relationship, ...]
    selected_tables: tuple[str, ...]


class SchemaContextBuilder:
    """Keeps prompts small while preserving the likely join path."""

    def __init__(
        self,
        max_tables: int = 6,
        use_rag: bool = False,
        embedding_model: str = "all-MiniLM-L6-v2",
        aliases: dict[str, str] | None = None,
    ) -> None:
        self.max_tables = max_tables
        self.use_rag = use_rag
        self.aliases = aliases or {}
        self.retriever = (
            SchemaRetriever(embedding_model, max_tables, self.aliases)
            if use_rag
            else None
        )

    def build(
        self,
        question: str,
        metadata: SchemaMetadata | SchemaMap,
    ) -> PromptSchemaContext:
        full = (
            metadata
            if isinstance(metadata, SchemaMetadata)
            else SchemaMetadata.from_schema_map(metadata)
        )
        expanded_question = expand_question(question, self.aliases)
        rag_tables = self.retriever.retrieve(expanded_question, full) if self.retriever else []
        if rag_tables:
            selected = rag_tables[: self.max_tables]
        else:
            selected = self._lexical_selection(expanded_question, full)

        expanded = self._include_join_neighbors(selected, full.relationships)
        schema = {name: full.tables[name].columns for name in expanded if name in full.tables}
        relationships = tuple(
            rel
            for rel in full.relationships
            if rel.source_table in schema and rel.target_table in schema
        )
        logger.info(
            "schema_context selected_tables=%s relationship_count=%s rag=%s",
            ",".join(schema),
            len(relationships),
            bool(rag_tables),
        )
        return PromptSchemaContext(schema=schema, relationships=relationships, selected_tables=tuple(schema))

    def _lexical_selection(self, question: str, full: SchemaMetadata) -> list[str]:
        scores = {
            name: self._score_table(question, name, table.columns)
            for name, table in full.tables.items()
        }
        ranked = sorted(scores, key=lambda table: (-scores[table], table))
        selected = [table for table in ranked if scores[table] > 0][: self.max_tables]
        if not selected:
            selected = ranked[: self.max_tables]
        return selected

    def _score_table(self, question: str, table_name: str, columns: dict[str, str]) -> int:
        words = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", question.lower()))
        score = 0
        table_tokens = set(table_name.lower().replace("_", " ").split())
        score += 4 * len(words & table_tokens)
        for column in columns:
            column_tokens = set(column.lower().replace("_", " ").split())
            if column.lower() in question.lower():
                score += 3
            score += len(words & column_tokens)
        return score

    def _include_join_neighbors(
        self,
        selected: list[str],
        relationships: tuple[Relationship, ...],
    ) -> list[str]:
        result = list(selected)
        for rel in relationships:
            if rel.source_table in result and rel.target_table not in result:
                result.append(rel.target_table)
            elif rel.target_table in result and rel.source_table not in result:
                result.append(rel.source_table)
            if len(result) >= self.max_tables:
                break
        return result[: self.max_tables]
