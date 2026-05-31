from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine


class UploadLoader:
    def __init__(self, upload_dir: str = "cache/uploads") -> None:
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def save_and_load(self, uploaded_file, engine: Engine) -> str:
        filename = self._safe_name(uploaded_file.name)
        path = self.upload_dir / filename
        path.write_bytes(uploaded_file.getbuffer())
        table_name = self._table_name(path.stem)

        suffix = path.suffix.lower()
        if suffix == ".csv":
            sql = f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_csv_auto('{path.as_posix()}')"
            with engine.begin() as conn:
                conn.execute(text(sql))
        elif suffix == ".parquet":
            sql = f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_parquet('{path.as_posix()}')"
            with engine.begin() as conn:
                conn.execute(text(sql))
        elif suffix in {".xlsx", ".xls"}:
            frame = pd.read_excel(path)
            frame.to_sql(table_name, engine, if_exists="replace", index=False)
        else:
            raise ValueError("Supported uploads are CSV, Excel, and Parquet files.")
        return table_name

    @staticmethod
    def _safe_name(name: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_.-]", "_", name)

    @staticmethod
    def _table_name(stem: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", stem.lower()).strip("_")
        if not cleaned:
            cleaned = "uploaded_data"
        if cleaned[0].isdigit():
            cleaned = f"data_{cleaned}"
        return cleaned[:48]
