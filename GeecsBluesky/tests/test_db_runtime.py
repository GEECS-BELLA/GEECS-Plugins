"""Hermetic unit tests for the M3c DB-integration runtime (get-side, pure logic).

No MySQL, no gateway, no network — a fake :class:`ScalarPolicyProvider` stands
in for :class:`~geecs_core.db.geecs_db.GeecsDb`.  Covers:

- db_scalars resolution (True = get∪explicit, all_scalars = all∪explicit,
  False = explicit-only, no-provider = explicit-only);
- telemetry selection (get='yes'-not-in-save-set, empty dropped);
- GeecsDbScalarPolicy tolerance (a DB failure degrades to empty policy).

The DB set-side (scan start/end writes) is intentionally disabled in this
version, so there is no boundary-write resolution to test here (see the
reserved-not-honored WARNING test in ``test_scan_request_runner.py``).
"""

from __future__ import annotations

import logging

from geecs_bluesky.db_runtime import GeecsDbScalarPolicy


class _FakePolicy:
    """In-memory get-side ScalarPolicyProvider (no DB)."""

    def __init__(
        self,
        subscribed: dict[str, list[str]] | None = None,
        all_vars: dict[str, list[str]] | None = None,
    ) -> None:
        self._subscribed = subscribed or {}
        self._all = all_vars or {}

    def get_variables(self, device: str) -> list[str]:
        return list(self._subscribed.get(device, []))

    def all_variables(self, device: str) -> list[str]:
        return list(self._all.get(device, []))

    def subscribed_by_device(self) -> dict[str, list[str]]:
        return dict(self._subscribed)


# ---------------------------------------------------------------------------
# DB-backed policy
# ---------------------------------------------------------------------------


class _RaisingDb:
    @staticmethod
    def get_subscribed_variables(experiment, *, enabled_only=True):
        raise RuntimeError("no network")

    @staticmethod
    def get_all_experiment_variables(experiment, *, enabled_only=True):
        raise RuntimeError("no network")


def test_policy_degrades_to_empty_on_db_failure(caplog) -> None:
    policy = GeecsDbScalarPolicy("Undulator", db=_RaisingDb)
    with caplog.at_level(logging.WARNING):
        assert policy.subscribed_by_device() == {}
        assert policy.get_variables("U_Cam") == []
        assert policy.all_variables("U_Cam") == []
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


# ---------------------------------------------------------------------------
# GeecsDbServedSetProvider (the unserved-variables pre-flight source)
# ---------------------------------------------------------------------------


class _ServedDb:
    """Fake DB with a subscribed set and a settable control surface."""

    calls = 0

    @classmethod
    def get_subscribed_variables(cls, experiment, *, enabled_only=True):
        cls.calls += 1
        return {"UC_TopView": ["centroidx", "centroidy"]}

    @classmethod
    def get_experiment_device_variables(cls, experiment, *, enabled_only=True):
        return {
            "UC_TopView": [
                {"name": "centroidx", "settable": False},
                {"name": "2ndmomW0x", "settable": False},  # real but unserved
                {"name": "save", "settable": True},
                {"name": "localsavingpath", "settable": True},
            ],
            "U_SettableOnly": [
                {"name": "Setpoint", "settable": True},
            ],
            "U_NothingServed": [
                {"name": "ReadOnlyThing", "settable": False},
            ],
        }


def test_served_set_is_subscribed_union_settable() -> None:
    from geecs_bluesky.db_runtime import GeecsDbServedSetProvider

    provider = GeecsDbServedSetProvider("Undulator", db=_ServedDb)
    served = provider.served_by_device()
    assert served is not None
    # get='yes' ∪ settable — 2ndmomW0x is a real DB variable but in neither.
    assert served["UC_TopView"] == {
        "centroidx",
        "centroidy",
        "save",
        "localsavingpath",
    }
    # A device with zero get='yes' variables keeps its control surface.
    assert served["U_SettableOnly"] == {"Setpoint"}
    # Nothing subscribed and nothing settable → the gateway skips the device.
    assert "U_NothingServed" not in served


def test_served_set_is_cached() -> None:
    from geecs_bluesky.db_runtime import GeecsDbServedSetProvider

    _ServedDb.calls = 0
    provider = GeecsDbServedSetProvider("Undulator", db=_ServedDb)
    provider.served_by_device()
    provider.served_by_device()
    assert _ServedDb.calls == 1


def test_served_set_db_failure_returns_none_not_empty(caplog) -> None:
    from geecs_bluesky.db_runtime import GeecsDbServedSetProvider

    provider = GeecsDbServedSetProvider("Undulator", db=_RaisingDb)
    with caplog.at_level(logging.WARNING):
        assert provider.served_by_device() is None  # unknown, NOT empty
        assert provider.served_by_device() is None  # failure is cached too
    warnings = [r for r in caplog.records if "served set" in r.getMessage()]
    assert len(warnings) == 1
