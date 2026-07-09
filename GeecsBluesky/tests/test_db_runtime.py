"""Hermetic unit tests for the M3c DB-integration runtime (get-side, pure logic).

No MySQL, no gateway, no network — a fake :class:`ScalarPolicyProvider` stands
in for :class:`~geecs_ca_gateway.db.geecs_db.GeecsDb`.  Covers:

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

from geecs_bluesky.db_runtime import (
    GeecsDbScalarPolicy,
    resolve_entry_scalars,
    select_telemetry_variables,
)
from geecs_schemas import SaveSet, SaveSetEntry


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
