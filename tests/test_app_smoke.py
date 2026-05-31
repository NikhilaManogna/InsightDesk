from __future__ import annotations

import pandas as pd

from app import markdown_safe, normalize_uploaded_table_reference, result_key


def test_result_keys_are_unique_for_repeated_sql_payloads() -> None:
    frame = pd.DataFrame({"x": [1]})
    first = {"result_id": "one", "sql": "select 1", "frame": frame}
    second = {"result_id": "two", "sql": "select 1", "frame": frame}
    assert result_key(first, "csv") != result_key(second, "csv")


def test_markdown_safe_escapes_currency_symbols() -> None:
    assert markdown_safe("Revenue was $1,000") == "Revenue was \\$1,000"


def test_uploaded_filename_reference_is_normalized() -> None:
    sql = 'select region from "sample_sales.csv"'
    fixed = normalize_uploaded_table_reference(sql, "sample_sales", "sample_sales.csv")
    assert fixed == "select region from sample_sales"
