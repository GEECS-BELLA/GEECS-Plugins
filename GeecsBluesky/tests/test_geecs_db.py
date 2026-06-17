"""Tests for lightweight GEECS database lookup helpers."""

from __future__ import annotations

import sys
import types

from geecs_bluesky.db import geecs_db
from geecs_bluesky.db.geecs_db import GeecsDb


class _FakeCursor:
    """Minimal cursor fake for a single device lookup."""

    def execute(self, *_args, **_kwargs) -> None:
        """Accept any query without side effects."""
        pass

    def fetchone(self) -> tuple[str, int]:
        """Return one fake device row."""
        return ("192.168.1.10", 12345)


class _FakeConnection:
    """Minimal connection fake returning a fake cursor."""

    def cursor(self) -> _FakeCursor:
        """Return a cursor fake."""
        return _FakeCursor()

    def close(self) -> None:
        """Close without side effects."""
        pass


def test_find_device_uses_pure_python_mysql_connector(monkeypatch) -> None:
    """DB lookups should avoid the crash-prone mysql C extension on Windows."""
    calls: list[dict] = []

    def connect(**kwargs):
        calls.append(kwargs)
        return _FakeConnection()

    mysql_pkg = types.ModuleType("mysql")
    connector_mod = types.ModuleType("mysql.connector")
    connector_mod.connect = connect
    mysql_pkg.connector = connector_mod
    monkeypatch.setitem(sys.modules, "mysql", mysql_pkg)
    monkeypatch.setitem(sys.modules, "mysql.connector", connector_mod)
    monkeypatch.setattr(
        geecs_db,
        "_find_credentials",
        lambda: {
            "host": "db",
            "port": 3306,
            "database": "geecs",
            "user": "user",
            "password": "pw",
        },
    )

    assert GeecsDb.find_device("U_TestDevice") == ("192.168.1.10", 12345)
    assert calls and calls[0]["use_pure"] is True
