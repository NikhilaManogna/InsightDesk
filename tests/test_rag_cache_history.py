from __future__ import annotations

import pandas as pd

from backend.cache.cache_service import CacheService
from backend.db.schema_metadata import Relationship, SchemaMetadata, TableMetadata
from backend.history.query_history import QueryHistoryRecord, QueryHistoryStore
from backend.llm.business_aliases import expand_question, parse_business_aliases
from backend.rag.schema_retriever import SchemaRetriever


def _metadata() -> SchemaMetadata:
    return SchemaMetadata(
        tables={
            "orders": TableMetadata(
                "orders",
                {"order_id": "INTEGER", "customer_id": "INTEGER", "order_amount": "DOUBLE"},
            ),
            "customers": TableMetadata(
                "customers",
                {"id": "INTEGER", "name": "VARCHAR"},
                primary_keys=("id",),
            ),
        },
        relationships=(Relationship("orders", "customer_id", "customers", "id"),),
    )


def test_business_alias_resolution() -> None:
    aliases = parse_business_aliases("revenue=order_amount sales")
    expanded = expand_question("show revenue by customer", aliases)
    assert "order_amount" in expanded


def test_schema_retriever_returns_relevant_table() -> None:
    retriever = SchemaRetriever(
        model_name="missing-local-model",
        max_tables=2,
        aliases={"revenue": "order_amount"},
    )
    tables = retriever.retrieve("revenue by customer", _metadata())
    assert "orders" in tables


def test_cache_service_memory_roundtrip() -> None:
    cache = CacheService(ttl_seconds=60)
    key = cache.make_key("test", "abc")
    frame = pd.DataFrame({"x": [1, 2]})
    cache.set(key, frame)
    cached = cache.get(key)
    assert cached.equals(frame)


def test_query_history_store_roundtrip(tmp_path) -> None:
    store = QueryHistoryStore(str(tmp_path / "history.jsonl"), max_records=3)
    store.append(QueryHistoryRecord("q1", "select 1", "DuckDB", 1))
    store.append(QueryHistoryRecord("q2", "select 2", "DuckDB", 1))
    rows = store.recent(10)
    assert len(rows) == 2
    assert rows[-1]["question"] == "q2"
