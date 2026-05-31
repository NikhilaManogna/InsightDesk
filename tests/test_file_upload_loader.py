from __future__ import annotations

from io import BytesIO

from sqlalchemy import create_engine, text

from backend.files.upload_loader import UploadLoader


class FakeUpload:
    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._data = data

    def getbuffer(self):
        return memoryview(self._data)


def test_upload_loader_safe_table_name(tmp_path) -> None:
    loader = UploadLoader(str(tmp_path))
    assert loader._table_name("2026 Sales Data!") == "data_2026_sales_data"
    assert loader._safe_name("bad name.csv") == "bad_name.csv"


def test_upload_loader_creates_duckdb_table(tmp_path) -> None:
    engine = create_engine(f"duckdb:///{tmp_path / 'test.duckdb'}")
    upload = FakeUpload("sample_sales.csv", b"region,revenue\nNorth,10\nSouth,5\n")
    table = UploadLoader(str(tmp_path)).save_and_load(upload, engine)
    with engine.connect() as conn:
        rows = conn.execute(text(f"select count(*) from {table}")).scalar()
    assert table == "sample_sales"
    assert rows == 2
