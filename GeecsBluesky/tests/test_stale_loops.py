"""A frozen LabVIEW loop (connected, its systimestamp stale): the probe (the run's warning is in test_plan_registry)."""

from __future__ import annotations

import pytest

from geecs_bluesky.devices.ca.liveness import STALE_AFTER_S, probe_stale_pvs, stale_text

LV = 2_082_844_800


def test_the_probe_flags_an_old_or_unset_stamp_and_skips_an_unread_one(monkeypatch):
    now = 1_000_000.0
    readings = {
        "fresh:systimestamp": now + LV - 2.0,
        "old:systimestamp": now + LV - (STALE_AFTER_S + 10),
        "unset:systimestamp": 0.0,
        "unread:systimestamp": None,
    }
    monkeypatch.setattr(
        "geecs_bluesky.devices.ca.oneshot.try_caget_many",
        lambda pvs, *, timeout: [readings[p] for p in pvs],
    )
    pvs = {n: f"{n}:systimestamp" for n in ("fresh", "old", "unset", "unread")}
    stale = probe_stale_pvs(pvs, now=now)
    assert [name for name, _ in stale] == ["old", "unset"]
    assert stale[0][1] == pytest.approx(STALE_AFTER_S + 10)
    assert stale[1][1] is None
    assert stale_text(stale) == "old (not updated in 40 s), unset (never updated)"
