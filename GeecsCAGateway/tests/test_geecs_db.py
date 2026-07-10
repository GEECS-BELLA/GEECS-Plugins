"""Tests for lightweight GEECS database lookup helpers."""

from __future__ import annotations

import logging
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


def _patch_query_sequence(
    monkeypatch, responses: list[list[tuple]], queries: list
) -> None:
    """Route _connect_mysql to a fake returning responses[i] for the i-th query."""

    class _Cursor:
        def __init__(self) -> None:
            self._rows: list[tuple] = []

        def execute(self, query: str, params: tuple) -> None:
            self._rows = responses[len(queries)]
            queries.append((query, params))

        def fetchall(self) -> list[tuple]:
            return self._rows

    class _Connection:
        def __init__(self) -> None:
            self._cursor = _Cursor()

        def cursor(self) -> _Cursor:
            return self._cursor

        def close(self) -> None:
            pass

    mysql_pkg = types.ModuleType("mysql")
    connector_mod = types.ModuleType("mysql.connector")
    mysql_pkg.connector = connector_mod
    monkeypatch.setitem(sys.modules, "mysql", mysql_pkg)
    monkeypatch.setitem(sys.modules, "mysql.connector", connector_mod)
    monkeypatch.setattr(geecs_db, "_connect_mysql", lambda _connector: _Connection())


# devicetype_variable rows: (id, name, units, min, max, set, variabletype,
# choices, tolerance) — the type-default half of the capability chain.
_TYPE_ROWS = [
    (11, "Current", "A", "-5", "5", "no", "numeric", None, "0.05"),
    (12, "Enable", "", None, None, "yes", "choice", "on,off", None),
    (13, "Voltage", "V", "-10", "10", "yes", "numeric", None, "0.1"),
]


def test_get_device_variables_type_only_inherits_type_defaults(monkeypatch) -> None:
    """No instance rows: metadata comes from devicetype_variable unchanged.

    Regression guard: the meta dict shape for the plain (inherited) case is
    exactly what config.from_db_metadata has always consumed.
    """
    queries: list = []
    _patch_query_sequence(monkeypatch, [_TYPE_ROWS, []], queries)

    result = GeecsDb.get_device_variables("U_S1H")
    assert len(queries) == 2
    assert queries[0][1] == ("U_S1H",) and queries[1][1] == ("U_S1H",)
    assert "devicetype_variable" in queries[0][0]
    assert "FROM variable v" in queries[1][0]

    assert result[0] == {
        "name": "Current",
        "units": "A",
        "min": -5.0,
        "max": 5.0,
        "settable": False,
        "variabletype": "numeric",
        "choices": None,
        "tolerance": 0.05,
        "description": "",
    }
    assert sorted(result[0]) == sorted(result[1]) == sorted(result[2])
    assert result[1]["choices"] == "on,off"
    assert result[2]["settable"] is True


def test_instance_row_overrides_type_row_wholesale(monkeypatch) -> None:
    """An instance `variable` row replaces its type row wholesale, not per field.

    Settability flips no→yes (the missing-:SP live case), instance NULL limits
    erase the type limits, and units/tolerance/choices all come from the
    instance row — no field-level fallback to the type defaults.
    """
    instance_rows = [
        # (devicetype_variable_id, name, units, min, max, set, variabletype,
        #  choices, tolerance)
        (11, "Current", "mA", None, None, "yes", "numeric", None, None),
        (12, "Enable", "", None, None, "yes", "choice", "off,auto,on", "0.5"),
    ]
    _patch_query_sequence(monkeypatch, [_TYPE_ROWS, instance_rows], [])

    result = GeecsDb.get_device_variables("U_S1H")
    by_name = {m["name"]: m for m in result}

    current = by_name["Current"]
    assert current["settable"] is True  # no→yes flip: gains its :SP PV
    assert current["units"] == "mA"
    assert current["min"] is None and current["max"] is None  # NULL erases limits
    assert current["tolerance"] is None  # NULL erases type tolerance too

    enable = by_name["Enable"]
    assert enable["choices"] == "off,auto,on"  # instance choice_id join wins
    assert enable["tolerance"] == 0.5

    # Voltage has no instance row — inherited type defaults untouched.
    assert by_name["Voltage"]["settable"] is True
    assert by_name["Voltage"]["min"] == -10.0


def test_instance_set_flip_yes_to_no_loses_settability(monkeypatch) -> None:
    """A yes→no instance override makes a type-settable variable read-only."""
    instance_rows = [
        (13, "Voltage", "V", "-10", "10", "no", "numeric", None, "0.1"),
    ]
    _patch_query_sequence(monkeypatch, [_TYPE_ROWS, instance_rows], [])

    result = GeecsDb.get_device_variables("U_S1H")
    by_name = {m["name"]: m for m in result}
    assert by_name["Voltage"]["settable"] is False


def test_instance_only_variable_is_served(monkeypatch) -> None:
    """Instance rows with a NULL or dangling type link still get served.

    They define instance-only variables and are keyed by name: a same-named
    type row is replaced (instance rows are ground truth, and two entries
    would collide on one PV name), otherwise the variable is appended.
    """
    instance_rows = [
        # NULL link, no same-named type row: pure instance-only variable.
        (None, "LocalOnly", "mm", "0", "1", "yes", "numeric", None, None),
        # Dangling link (no type row id 999): same treatment.
        (999, "Orphan", "", None, None, "no", "string", None, None),
        # NULL link but colliding name: replaces the type row, no duplicate.
        (None, "Voltage", "kV", None, None, "no", "numeric", None, None),
    ]
    _patch_query_sequence(monkeypatch, [_TYPE_ROWS, instance_rows], [])

    result = GeecsDb.get_device_variables("U_S1H")
    names = [m["name"] for m in result]
    assert names == ["Current", "Enable", "Voltage", "LocalOnly", "Orphan"]

    by_name = {m["name"]: m for m in result}
    assert by_name["LocalOnly"]["settable"] is True
    assert by_name["Orphan"]["variabletype"] == "string"
    # the name-keyed collision: instance row wins, type limits gone
    assert by_name["Voltage"]["units"] == "kV"
    assert by_name["Voltage"]["settable"] is False
    assert by_name["Voltage"]["min"] is None


def test_instance_description_flows_through(monkeypatch) -> None:
    """A per-instance ``variable.description`` reaches the meta dict.

    ``description`` lives only on the instance table; the type query selects
    ``NULL`` in that slot, so a description is inherently a per-instance fact.
    Instance rows carry a 10th column (after the link id); type rows do not.
    """
    type_rows = [
        # id, name, units, min, max, set, variabletype, choices, tol, NULL(desc)
        (11, "Current", "A", "-5", "5", "yes", "numeric", None, "0.05", None),
    ]
    instance_rows = [
        # link, name, units, min, max, set, variabletype, choices, tol, desc
        (
            11,
            "Current",
            "A",
            "-5",
            "5",
            "yes",
            "numeric",
            None,
            "0.05",
            "S1H steering current",
        ),
    ]
    _patch_query_sequence(monkeypatch, [type_rows, instance_rows], [])

    result = GeecsDb.get_device_variables("U_S1H")
    assert result[0]["description"] == "S1H steering current"


def test_type_only_description_defaults_empty(monkeypatch) -> None:
    """With no instance row, description is empty (type table has no such column)."""
    type_rows = [
        (11, "Current", "A", "-5", "5", "no", "numeric", None, "0.05", None),
    ]
    _patch_query_sequence(monkeypatch, [type_rows, []], [])

    result = GeecsDb.get_device_variables("U_S1H")
    assert result[0]["description"] == ""


def test_get_experiment_device_variables_batches_metadata(monkeypatch) -> None:
    """Two batched queries return all devices' metadata, grouped and mapped."""
    queries: list = []
    type_rows = [
        # d.name, dtv.id, dtv.name, units, min, max, set, variabletype,
        # choices, tol
        ("U_A", 1, "Current", "A", "-5", "5", "yes", "numeric", None, "0.05"),
        ("U_A", 2, "Enable", "", None, None, "yes", "choice", "on,off", None),
        ("U_B", 3, "Voltage", "V", None, None, "no", None, None, None),
    ]
    _patch_query_sequence(monkeypatch, [type_rows, []], queries)

    result = GeecsDb.get_experiment_device_variables("Undulator")
    assert set(result) == {"U_A", "U_B"}
    assert len(queries) == 2
    assert queries[0][1] == queries[1][1] == ("Undulator",)
    assert all("LOWER(ed.enabled) = 'yes'" in q for q, _ in queries)

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
        "description": "",
    }
    assert result["U_A"][1]["choices"] == "on,off"
    assert result["U_B"][0]["settable"] is False

    queries.clear()
    _patch_query_sequence(monkeypatch, [type_rows, []], queries)
    GeecsDb.get_experiment_device_variables("Undulator", enabled_only=False)
    assert all("enabled" not in q for q, _ in queries)


def test_get_experiment_device_variables_resolves_instance_overrides(
    monkeypatch,
) -> None:
    """The experiment-wide variant applies the same wholesale resolution."""
    type_rows = [
        ("U_A", 1, "Current", "A", "-5", "5", "no", "numeric", None, "0.05"),
        ("U_B", 2, "Voltage", "V", "-10", "10", "yes", "numeric", None, None),
    ]
    instance_rows = [
        # v.device, v.devicetype_variable_id, v.name, units, min, max, set,
        # variabletype, choices, tol
        ("U_A", 1, "Current", "A", "-1", "1", "yes", "numeric", None, None),
        # instance-only variable on a device with no type rows at all
        ("U_C", None, "Special", "", None, None, "yes", "string", None, None),
    ]
    _patch_query_sequence(monkeypatch, [type_rows, instance_rows], [])

    result = GeecsDb.get_experiment_device_variables("Undulator")
    assert set(result) == {"U_A", "U_B", "U_C"}

    current = result["U_A"][0]
    assert current["settable"] is True  # instance no→yes flip
    assert (current["min"], current["max"]) == (-1.0, 1.0)
    assert current["tolerance"] is None  # wholesale: type tolerance erased

    assert result["U_B"][0]["settable"] is True  # untouched inheritance
    assert result["U_C"][0]["name"] == "Special"  # instance-only device served


def test_get_all_experiment_variables_groups_and_dedupes(monkeypatch) -> None:
    """Every expt_device_variable row per device, deduped, in stable order."""
    queries: list = []
    _patch_rows(
        monkeypatch,
        [
            ("U_A", "MaxCounts"),
            ("U_A", "centroidx"),
            ("U_A", "centroidx"),  # duplicate DB row (known quirk) → collapsed
            ("U_B", "Pressure"),
        ],
        queries,
    )

    result = GeecsDb.get_all_experiment_variables("Undulator")
    assert result == {"U_A": ["MaxCounts", "centroidx"], "U_B": ["Pressure"]}
    assert len(queries) == 1 and queries[0][1] == ("Undulator",)
    assert "LOWER(ed.enabled) = 'yes'" in queries[0][0]

    queries.clear()
    GeecsDb.get_all_experiment_variables("Undulator", enabled_only=False)
    assert "enabled" not in queries[0][0]


def test_get_scan_boundary_writes_returns_set_yes_rows(monkeypatch) -> None:
    """set='yes' rows carry startvalue/endvalue as strings (None preserved)."""
    queries: list = []
    _patch_rows(
        monkeypatch,
        [
            ("U_DG", "TriggerMode", "Scan", "Standby"),
            ("U_DG", "Amplitude", 4.0, None),  # numeric coerced to str; None kept
            ("U_Cam", "Shutter", "Open", "Closed"),
        ],
        queries,
    )

    result = GeecsDb.get_scan_boundary_writes("Undulator")
    assert result["U_DG"] == [
        {"variable": "TriggerMode", "startvalue": "Scan", "endvalue": "Standby"},
        {"variable": "Amplitude", "startvalue": "4.0", "endvalue": None},
    ]
    assert result["U_Cam"] == [
        {"variable": "Shutter", "startvalue": "Open", "endvalue": "Closed"}
    ]
    assert len(queries) == 1 and queries[0][1] == ("Undulator",)
    query = queries[0][0]
    assert "`set` = 'yes'" in query
    assert "LOWER(ed.enabled) = 'yes'" in query

    queries.clear()
    GeecsDb.get_scan_boundary_writes("Undulator", enabled_only=False)
    assert "enabled" not in queries[0][0]


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


def test_get_ca_alarm_limits_missing_table_is_fail_open(monkeypatch, caplog) -> None:
    """The optional alarm table can be absent during rollout."""
    caplog.set_level(logging.INFO)

    class _MissingTableError(Exception):
        errno = 1146

    class _Cursor:
        def execute(self, _query: str, _params: tuple) -> None:
            raise _MissingTableError("table does not exist")

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
    assert "table is absent" in caplog.text


def test_get_ca_alarm_limits_query_failure_is_warning(monkeypatch, caplog) -> None:
    """Unexpected DB errors stay fail-open but are visible above INFO."""

    class _Cursor:
        def execute(self, _query: str, _params: tuple) -> None:
            raise RuntimeError("permission denied")

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
    assert "ca_alarm_limits lookup failed" in caplog.text
