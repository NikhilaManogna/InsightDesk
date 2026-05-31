from __future__ import annotations

from backend.db.schema_metadata import SchemaMetadata
from backend.rag.rag_models import SchemaDocument


class SchemaEmbedder:
    def documents(self, metadata: SchemaMetadata, aliases: dict[str, str] | None = None) -> list[SchemaDocument]:
        alias_text = aliases or {}
        docs: list[SchemaDocument] = []
        for table_name, table in metadata.tables.items():
            columns = ", ".join(f"{name} {dtype}" for name, dtype in table.columns.items())
            related_aliases = [
                alias for alias, target in alias_text.items() if target.lower().startswith(table_name.lower())
            ]
            text = f"table {table_name}. columns: {columns}."
            if table.primary_keys:
                text += f" primary keys: {', '.join(table.primary_keys)}."
            if related_aliases:
                text += f" business terms: {', '.join(related_aliases)}."
            docs.append(SchemaDocument(id=f"table:{table_name}", table=table_name, text=text))

        for rel in metadata.relationships:
            docs.append(
                SchemaDocument(
                    id=f"rel:{rel.render()}",
                    table=rel.source_table,
                    text=f"relationship {rel.render()}",
                    kind="relationship",
                )
            )
        return docs
