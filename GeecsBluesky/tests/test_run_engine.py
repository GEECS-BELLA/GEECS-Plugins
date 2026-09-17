"""make_run_engine / install_telemetry: the baseline only holds objects that connect."""

from __future__ import annotations

import logging

import pytest

pytest.importorskip("aioca")

import bluesky.plans as bp  # noqa: E402
from bluesky.preprocessors import SupplementalData  # noqa: E402
from ophyd_async.core import Device, NotConnectedError  # noqa: E402

from geecs_bluesky.devices.ca import CaSnapshotReadable  # noqa: E402
from geecs_bluesky.preprocessors import connect_on_demand  # noqa: E402
from geecs_bluesky.run_engine import install_telemetry, make_run_engine  # noqa: E402
from tests.ca_mock_helpers import DocCollector, connect_mock  # noqa: E402


class Unservable(Device):
    """A device the gateway does not serve: its connect always fails."""

    async def connect(self, mock=False, timeout=10.0, force_reconnect=False):
        raise NotConnectedError("no PV")


def test_unconnectable_telemetry_is_dropped_loudly_at_build(caplog):
    gauge = CaSnapshotReadable(
        "U_Gauge", ["Pressure"], experiment="TestExp", name="u_gauge"
    )
    dead = Unservable(name="u_dead")
    with caplog.at_level(logging.WARNING):
        RE = make_run_engine(mock=True, telemetry=[gauge, dead])
    assert "u_dead" in caplog.text and "left out of the baseline" in caplog.text
    baselines = [p for p in RE.preprocessors if isinstance(p, SupplementalData)]
    assert len(baselines) == 1 and baselines[0].baseline == [gauge]
    funcs = [getattr(p, "func", p) for p in RE.preprocessors]
    assert funcs[-1] is connect_on_demand
    # the baseline runs: two rows around a count of another device
    other = CaSnapshotReadable("U_Other", ["X"], experiment="TestExp", name="u_other")
    connect_mock(RE, other)
    col = DocCollector()
    RE.subscribe(col)
    RE(bp.count([other], 1))
    streams = {d["name"] for d in col.docs["descriptor"]}
    assert {"baseline", "primary"} <= streams
    baseline_uid = next(
        d["uid"] for d in col.docs["descriptor"] if d["name"] == "baseline"
    )
    rows = [e for e in col.docs["event"] if e["descriptor"] == baseline_uid]
    assert len(rows) == 2 and "u_gauge-pressure" in rows[0]["data"]


def test_install_telemetry_returns_the_installed_list():
    RE = make_run_engine(mock=True)
    gauge = CaSnapshotReadable(
        "U_Gauge", ["Pressure"], experiment="TestExp", name="u_gauge"
    )
    assert install_telemetry(RE, [gauge], mock=True) == [gauge]


def test_claim_needs_the_experiment():
    with pytest.raises(ValueError, match="experiment"):
        make_run_engine(claim=True)
