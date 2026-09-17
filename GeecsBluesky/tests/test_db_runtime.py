"""Hermetic unit tests for the DB-integration runtime's served-set provider.

No MySQL, no gateway, no network — fake ``GeecsDb`` classes stand in for
:class:`~geecs_core.db.geecs_db.GeecsDb`.  Covers
``GeecsDbServedSetProvider`` (served set = subscribed ∪ settable, cached; a
DB failure reads as *unknown*, never empty).  The scalar policy's own tests
moved with it to GEECS-Core (``tests/test_scalar_policy.py``).

The DB set-side (scan start/end writes) is intentionally disabled, so there
is no boundary-write resolution to test here.
"""

from __future__ import annotations

import logging


class _RaisingDb:
    @staticmethod
    def get_subscribed_variables(experiment, *, enabled_only=True):
        raise RuntimeError("no network")


def test_served_set_db_failure_returns_none_not_empty(caplog) -> None:
    from geecs_bluesky.db_runtime import GeecsDbServedSetProvider

    provider = GeecsDbServedSetProvider("Undulator", db=_RaisingDb)
    with caplog.at_level(logging.WARNING):
        assert provider.served_by_device() is None  # unknown, NOT empty
        assert provider.served_by_device() is None  # failure is cached too
    warnings = [r for r in caplog.records if "served set" in r.getMessage()]
    assert len(warnings) == 1
