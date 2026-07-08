"""Tests for lightweight GEECS database lookup helpers."""

from __future__ import annotations

import sys
import types

from geecs_ca_gateway.alarms import AlarmLimits, AlarmSeverityName
from geecs_ca_gateway.db import geecs_db
from geecs_ca_gateway.db.geecs_db import GeecsDb


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


def test_get_device_type_queries_device_table(monkeypatch) -> None:
    """Device type lookup should return the database ``device.devicetype`` value."""
    queries: list[tuple[str, tuple[str, ...]]] = []

    class _DeviceTypeCursor:
        def execute(self, query: str, params: tuple[str, ...]) -> None:
            queries.append((query, params))

        def fetchone(self) -> tuple[str]:
            return ("Point Grey Camera",)

    class _DeviceTypeConnection:
        def cursor(self) -> _DeviceTypeCursor:
            return _DeviceTypeCursor()

        def close(self) -> None:
            pass

    mysql_pkg = types.ModuleType("mysql")
    connector_mod = types.ModuleType("mysql.connector")
    mysql_pkg.connector = connector_mod
    monkeypatch.setitem(sys.modules, "mysql", mysql_pkg)
    monkeypatch.setitem(sys.modules, "mysql.connector", connector_mod)
    monkeypatch.setattr(
        geecs_db,
        "_connect_mysql",
        lambda _connector: _DeviceTypeConnection(),
    )

    assert GeecsDb.get_device_type("UC_TopView") == "Point Grey Camera"
    assert queries == [
        ("SELECT devicetype FROM device WHERE name = %s", ("UC_TopView",))
    ]


def _patch_rows(monkeypatch, rows: list[tuple], queries: list) -> None:
    """Route _connect_mysql to a fake connection returning *rows* for any query."""

    class _Cursor:
        def execute(self, query: str, params: tuple) -> None:
            queries.append((query, params))

        def fetchall(self) -> list[tuple]:
            return rows

    class _Connection:
        def cursor(self) -> _Cursor:
            return _Cursor()

        def close(self) -> None:
            pass

    import sys
    import types

    mysql_pkg = types.ModuleType("mysql")
    connector_mod = types.ModuleType("mysql.connector")
    mysql_pkg.connector = connector_mod
    monkeypatch.setitem(sys.modules, "mysql", mysql_pkg)
    monkeypatch.setitem(sys.modules, "mysql.connector", connector_mod)
    monkeypatch.setattr(geecs_db, "_connect_mysql", lambda _connector: _Connection())


def test_get_experiment_devices_batches_endpoints(monkeypatch) -> None:
    """One query returns every device endpoint; enabled filter is in the SQL."""
    queries: list = []
    _patch_rows(
        monkeypatch,
        [("U_A", " 192.168.1.10 ", "111"), ("U_B", "192.168.1.11", 222)],
        queries,
    )

    result = GeecsDb.get_experiment_devices("Undulator")
    assert result == {"U_A": ("192.168.1.10", 111), "U_B": ("192.168.1.11", 222)}
    assert len(queries) == 1
    query, params = queries[0]
    assert params == ("Undulator",)
    assert "LOWER(ed.enabled) = 'yes'" in query

    queries.clear()
    GeecsDb.get_experiment_devices("Undulator", enabled_only=False)
    assert "enabled" not in queries[0][0]


def test_get_experiment_device_variables_batches_metadata(monkeypatch) -> None:
    """One query returns all devices' variable metadata, grouped and mapped."""
    queries: list = []
    _patch_rows(
        monkeypatch,
        [
            # d.name, dtv.name, units, min, max, set, variabletype, choices, tol
            ("U_A", "Current", "A", "-5", "5", "yes", "numeric", None, "0.05"),
            ("U_A", "Enable", "", None, None, "yes", "choice", "on,off", None),
            ("U_B", "Voltage", "V", None, None, "no", None, None, None),
        ],
        queries,
    )

    result = GeecsDb.get_experiment_device_variables("Undulator")
    assert set(result) == {"U_A", "U_B"}
    assert len(queries) == 1 and queries[0][1] == ("Undulator",)

    current = result["U_A"][0]
    # same row shape as get_device_variables — from_db_metadata consumes both
    assert current == {
        "name": "Current",
        "units": "A",
        "min": -5.0,
        "max": 5.0,
        "settable": True,
        "variabletype": "numeric",
        "choices": None,
        "tolerance": 0.05,
    }
    assert result["U_A"][1]["choices"] == "on,off"
    assert result["U_B"][0]["settable"] is False


def test_get_ca_alarm_limits_returns_validated_rows(monkeypatch) -> None:
    """Optional alarm-limit rows are keyed by device/variable."""
    queries: list = []
    _patch_rows(
        monkeypatch,
        [
            (
                "Undulator",
                "U_A",
                "Current",
                None,
                "1.0",
                "4.0",
                "5.0",
                None,
                "MINOR",
                "MINOR",
                "MAJOR",
                "0.1",
                "current alarm",
            )
        ],
        queries,
    )

    result = GeecsDb.get_ca_alarm_limits("Undulator")
    assert queries and "ca_alarm_limits" in queries[0][0]
    assert result == {
        ("U_A", "Current"): AlarmLimits(
            low=1.0,
            high=4.0,
            hihi=5.0,
            high_severity=AlarmSeverityName.MINOR,
            hihi_severity=AlarmSeverityName.MAJOR,
            hysteresis=0.1,
            description="current alarm",
        )
    }


def test_get_ca_alarm_limits_missing_table_is_fail_open(monkeypatch) -> None:
    """The optional alarm table can be absent during rollout."""

    class _Cursor:
        def execute(self, _query: str, _params: tuple) -> None:
            raise RuntimeError("table does not exist")

    class _Connection:
        def cursor(self) -> _Cursor:
            return _Cursor()

        def close(self) -> None:
            pass

    mysql_pkg = types.ModuleType("mysql")
    connector_mod = types.ModuleType("mysql.connector")
    mysql_pkg.connector = connector_mod
    monkeypatch.setitem(sys.modules, "mysql", mysql_pkg)
    monkeypatch.setitem(sys.modules, "mysql.connector", connector_mod)
    monkeypatch.setattr(geecs_db, "_connect_mysql", lambda _connector: _Connection())

    assert GeecsDb.get_ca_alarm_limits("Undulator") == {}
