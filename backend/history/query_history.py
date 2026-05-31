from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QueryHistoryRecord:
    question: str
    sql: str
    database: str
    rows: int
    execution_ms: int | None = None
    retries: int = 0
    chart: str | None = None
    created_at: str = ""


class QueryHistoryStore:
    def __init__(self, path: str, max_records: int = 200) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_records = max_records

    def append(self, record: QueryHistoryRecord) -> None:
        payload = asdict(record)
        payload["created_at"] = payload["created_at"] or datetime.now(UTC).isoformat()
        records = self.recent(self.max_records - 1)
        records.append(payload)
        self.path.write_text(
            "\n".join(json.dumps(item, default=str) for item in records) + "\n",
            encoding="utf-8",
        )

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rows.append(json.loads(line))
        return rows[-limit:]
