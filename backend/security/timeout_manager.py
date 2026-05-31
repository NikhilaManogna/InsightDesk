from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimeoutConfig:
    seconds: int

    def validate(self) -> None:
        if self.seconds < 1:
            raise ValueError("Query timeout must be at least one second.")
