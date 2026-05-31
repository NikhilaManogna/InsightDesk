from __future__ import annotations

from backend.utils.config import get_settings
from backend.utils.startup import run_startup_checks


def test_startup_checks_return_statuses() -> None:
    checks = run_startup_checks(get_settings())
    assert checks
    assert all(isinstance(check.message, str) for check in checks)
