"""Hermetic unit tests for the M3c DB-integration runtime (pure logic).

No MySQL, no gateway, no network — a fake :class:`ScalarPolicyProvider` stands
in for :class:`~geecs_ca_gateway.db.geecs_db.GeecsDb`.  Covers:

- db_scalars resolution (True = get∪explicit, all_scalars = all∪explicit,
  False = explicit-only, no-provider = explicit-only);
- start/end write selection (set='yes' rows, save/localsavingpath skipped,
  override value / null-suppression / absent cases, uncurated-variable
  forced override);
- participants assembly (save-set devices + scan-variable devices);
- telemetry selection (get='yes'-not-in-save-set, empty dropped);
- GeecsDbScalarPolicy tolerance (a DB failure degrades to empty policy).
"""

from __future__ import annotations

import logging

from geecs_bluesky.db_runtime import (
    GeecsDbScalarPolicy,
    collect_scan_boundary_writes,
    resolve_boundary_writes,
    resolve_entry_scalars,
    select_telemetry_variables,
)
from geecs_schemas import SaveSet, SaveSetEntry


class _FakePolicy:
    """In-memory ScalarPolicyProvider (no DB)."""

    def __init__(
        self,
        subscribed: dict[str, list[str]] | None = None,
        all_vars: dict[str, list[str]] | None = None,
        boundary: dict[str, list[dict]] | None = None,
    ) -> None:
        self._subscribed = subscribed or {}
        self._all = all_vars or {}
        self._boundary = boundary or {}

    def get_variables(self, device: str) -> list[str]:
        return list(self._subscribed.get(device, []))

    def all_variables(self, device: str) -> list[str]:
        return list(self._all.get(device, []))

    def subscribed_by_device(self) -> dict[str, list[str]]:
        return dict(self._subscribed)

    def boundary_writes(self, device: str) -> list[dict]:
        return list(self._boundary.get(device, []))


# ---------------------------------------------------------------------------
# db_scalars resolution
# ---------------------------------------------------------------------------


def test_db_scalars_true_unions_get_yes_with_explicit() -> None:
    policy = _FakePolicy(subscribed={"U_Cam": ["MaxCounts", "centroidx"]})
    out = resolve_entry_scalars(
        "U_Cam", ["Extra"], db_scalars=True, all_scalars=False, provider=policy
    )
    # DB get='yes' first (DB order), then explicit extras not already present.
    assert out == ["MaxCounts", "centroidx", "Extra"]


def test_db_scalars_true_dedupes_overlap() -> None:
    policy = _FakePolicy(subscribed={"U_Cam": ["MaxCounts", "centroidx"]})
    out = resolve_entry_scalars(
        "U_Cam", ["centroidx"], db_scalars=True, all_scalars=False, provider=policy
    )
    assert out == ["MaxCounts", "centroidx"]


def test_all_scalars_unions_every_db_variable() -> None:
    policy = _FakePolicy(
        subscribed={"U_Cam": ["MaxCounts"]},
        all_vars={"U_Cam": ["MaxCounts", "centroidx", "Exposure"]},
    )
    out = resolve_entry_scalars(
        "U_Cam", [], db_scalars=True, all_scalars=True, provider=policy
    )
    assert out == ["MaxCounts", "centroidx", "Exposure"]


def test_db_scalars_false_is_explicit_only() -> None:
    policy = _FakePolicy(subscribed={"U_Cam": ["MaxCounts", "centroidx"]})
    out = resolve_entry_scalars(
        "U_Cam", ["Val"], db_scalars=False, all_scalars=False, provider=policy
    )
    # The legacy-converter pin: only the explicit list, DB ignored entirely.
    assert out == ["Val"]


def test_no_provider_is_explicit_only_even_with_db_scalars_true() -> None:
    out = resolve_entry_scalars(
        "U_Cam", ["Val"], db_scalars=True, all_scalars=False, provider=None
    )
    assert out == ["Val"]


# ---------------------------------------------------------------------------
# start/end write selection
# ---------------------------------------------------------------------------


def _rows() -> list[dict]:
    return [
        {"variable": "TriggerMode", "startvalue": "Scan", "endvalue": "Standby"},
        {"variable": "Shutter", "startvalue": "Open", "endvalue": "Closed"},
        {"variable": "save", "startvalue": "on", "endvalue": "off"},
        {"variable": "localsavingpath", "startvalue": "C:/x", "endvalue": "C:/x"},
    ]


def test_start_writes_use_startvalue_and_skip_save_paths() -> None:
    writes = resolve_boundary_writes("U_DG", _rows(), which="start", overrides={})
    assert [(w.variable, w.value, w.source) for w in writes] == [
        ("TriggerMode", "Scan", "db"),
        ("Shutter", "Open", "db"),
    ]


def test_end_writes_use_endvalue() -> None:
    writes = resolve_boundary_writes("U_DG", _rows(), which="end", overrides={})
    assert [(w.variable, w.value) for w in writes] == [
        ("TriggerMode", "Standby"),
        ("Shutter", "Closed"),
    ]


def test_override_value_replaces_db_value() -> None:
    writes = resolve_boundary_writes(
        "U_DG", _rows(), which="start", overrides={"TriggerMode": "Single"}
    )
    trig = next(w for w in writes if w.variable == "TriggerMode")
    assert trig.value == "Single"
    assert trig.source == "override"


def test_null_override_suppresses_the_write() -> None:
    writes = resolve_boundary_writes(
        "U_DG", _rows(), which="start", overrides={"TriggerMode": None}
    )
    assert [w.variable for w in writes] == ["Shutter"]


def test_absent_override_uses_db_value() -> None:
    writes = resolve_boundary_writes(
        "U_DG", _rows(), which="start", overrides={"Shutter": "Half"}
    )
    trig = next(w for w in writes if w.variable == "TriggerMode")
    assert trig.value == "Scan" and trig.source == "db"


def test_null_db_value_without_override_is_no_write() -> None:
    rows = [{"variable": "Foo", "startvalue": None, "endvalue": "1"}]
    assert resolve_boundary_writes("U_D", rows, which="start", overrides={}) == []
    end = resolve_boundary_writes("U_D", rows, which="end", overrides={})
    assert [(w.variable, w.value) for w in end] == [("Foo", "1")]


def test_forced_override_for_uncurated_variable() -> None:
    # A variable with no DB row but an override value → a forced write.
    writes = resolve_boundary_writes(
        "U_DG", _rows(), which="start", overrides={"NewVar": "7"}
    )
    forced = next(w for w in writes if w.variable == "NewVar")
    assert forced.value == "7" and forced.source == "override"


def test_null_override_for_uncurated_variable_is_noop() -> None:
    writes = resolve_boundary_writes(
        "U_DG", _rows(), which="start", overrides={"NewVar": None}
    )
    assert all(w.variable != "NewVar" for w in writes)


# ---------------------------------------------------------------------------
# participants collection
# ---------------------------------------------------------------------------


def test_collect_boundary_writes_participants_only() -> None:
    policy = _FakePolicy(
        boundary={
            "U_DG": [{"variable": "Mode", "startvalue": "A", "endvalue": "B"}],
            "U_NotAParticipant": [
                {"variable": "X", "startvalue": "1", "endvalue": "2"}
            ],
        }
    )
    participants = {
        "U_DG": {"at_scan_start": {}, "at_scan_end": {}},
    }
    start, end = collect_scan_boundary_writes(participants, policy)
    assert [(w.device, w.variable) for w in start] == [("U_DG", "Mode")]
    # A device with rows but not in the participants map contributes nothing.
    assert all(w.device != "U_NotAParticipant" for w in start + end)


def test_collect_boundary_writes_no_provider_only_forced_overrides() -> None:
    participants = {
        "U_DG": {"at_scan_start": {"Mode": "A"}, "at_scan_end": {}},
    }
    start, end = collect_scan_boundary_writes(participants, None)
    assert [(w.device, w.variable, w.source) for w in start] == [
        ("U_DG", "Mode", "override")
    ]
    assert end == []


# ---------------------------------------------------------------------------
# telemetry selection
# ---------------------------------------------------------------------------


def test_telemetry_selects_get_yes_not_in_save_set() -> None:
    save_set = SaveSet(name="s", entries=[SaveSetEntry(device="U_Cam", scalars=["x"])])
    subscribed = {
        "U_Cam": ["MaxCounts"],  # in save set → excluded wholesale
        "U_Press": ["Pressure"],  # not in save set → telemetry
        "U_Empty": [],  # no get-vars → dropped
    }
    selected = select_telemetry_variables(save_set, subscribed)
    assert selected == {"U_Press": ["Pressure"]}


def test_telemetry_no_save_set_selects_everything() -> None:
    subscribed = {"U_A": ["v1"], "U_B": ["v2"]}
    assert select_telemetry_variables(None, subscribed) == subscribed


# ---------------------------------------------------------------------------
# GeecsDbScalarPolicy tolerance (no real DB)
# ---------------------------------------------------------------------------


class _RaisingDb:
    @staticmethod
    def get_subscribed_variables(experiment, *, enabled_only=True):
        raise RuntimeError("no network")

    @staticmethod
    def get_all_experiment_variables(experiment, *, enabled_only=True):
        raise RuntimeError("no network")

    @staticmethod
    def get_scan_boundary_writes(experiment, *, enabled_only=True):
        raise RuntimeError("no network")


def test_policy_degrades_to_empty_on_db_failure(caplog) -> None:
    policy = GeecsDbScalarPolicy("Undulator", db=_RaisingDb)
    with caplog.at_level(logging.WARNING):
        assert policy.subscribed_by_device() == {}
        assert policy.get_variables("U_Cam") == []
        assert policy.all_variables("U_Cam") == []
        assert policy.boundary_writes("U_Cam") == []
    assert any("Could not read" in r.message for r in caplog.records)


class _CountingDb:
    calls = 0

    @classmethod
    def get_subscribed_variables(cls, experiment, *, enabled_only=True):
        cls.calls += 1
        return {"U_Cam": ["MaxCounts"]}


def test_policy_caches_queries() -> None:
    _CountingDb.calls = 0
    policy = GeecsDbScalarPolicy("Undulator", db=_CountingDb)
    policy.get_variables("U_Cam")
    policy.get_variables("U_Other")
    policy.subscribed_by_device()
    assert _CountingDb.calls == 1  # one batched query, cached
