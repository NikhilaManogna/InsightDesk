from __future__ import annotations

from backend.rag.rag_models import RetrievalResult


class RetrievalRanker:
    def table_names(self, results: list[RetrievalResult], max_tables: int) -> list[str]:
        seen: set[str] = set()
        tables: list[str] = []
        for result in sorted(results, key=lambda item: item.score, reverse=True):
            table = result.document.table
            if table not in seen:
                seen.add(table)
                tables.append(table)
            if len(tables) >= max_tables:
                break
        return tables
