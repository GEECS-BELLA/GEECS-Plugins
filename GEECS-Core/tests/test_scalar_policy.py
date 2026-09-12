"""GeecsDbScalarPolicy — the subscribed-scalars rule (moved from geecs_bluesky.db_runtime)."""

from __future__ import annotations

import logging

from geecs_core.db.scalar_policy import GeecsDbScalarPolicy, ScalarPolicyProvider


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
    assert policy.get_variables("U_Cam") == ["MaxCounts"]
    policy.get_variables("U_Other")
    policy.subscribed_by_device()
    assert _CountingDb.calls == 1  # one batched query, cached
    assert isinstance(policy, ScalarPolicyProvider)


def test_db_resolved_on_first_query_not_at_construction() -> None:
    """Constructing a policy never imports the MySQL driver (lab access optional)."""
    policy = GeecsDbScalarPolicy("Undulator")
    assert policy.db is None
