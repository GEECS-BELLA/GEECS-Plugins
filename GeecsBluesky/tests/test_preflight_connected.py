"""Pin the pre-claim CONNECTED liveness re-check (#664).

The check runs at execution on both paths (runner + queue plan) because the
client-side pre-submit read cannot cover the submission-to-execution gap —
live incident 2026-08-22: a camera server rebooted while items sat queued
and the dead reference surfaced only as a mid-scan trigger timeout.

The rule is mode-independent: any dead **synchronous** device refuses
(strict rows await all of them; free-run rows are paced by the reference
and a dead sync contributor would fail t0 sync post-claim anyway — see
`_preflight_connected`'s docstring), while a dead asynchronous (snapshot)
device warns and continues.

CA reads are faked at the shared batch-read seam
(`oneshot.try_caget_many`, which `probe_disconnected` imports lazily) —
the fixture also pins the DBR_ENUM contract: the probe must request the
choice STRING (`datatype=str`), never the native integer index.
"""

from __future__ import annotations

import pytest

from geecs_bluesky.exceptions import GeecsDeviceDownError
from geecs_bluesky.scan_request_runner import _preflight_connected


class _Session:
    def __init__(self, experiment: str | None = "Test") -> None:
        self.experiment = experiment


def _config() -> dict[str, dict]:
    return {
        "UC_Ref": {"synchronous": True, "variable_list": ["a"]},
        "UC_Contrib": {"synchronous": True, "variable_list": ["b"]},
        "UC_Snap": {"synchronous": False, "variable_list": ["c"]},
    }


@pytest.fixture
def ca_reads(monkeypatch):
    """Fake the shared batch read: per-device CONNECTED choice strings."""
    state = {"values": {}, "batches": 0}

    def fake(pvs, *, timeout, datatype=None):
        state["batches"] += 1
        # The enum contract: the probe must read the choice STRING.
        assert datatype is str
        return [state["values"].get(pv.split(":")[-2]) for pv in pvs]

    monkeypatch.setattr("geecs_bluesky.devices.ca.oneshot.try_caget_many", fake)
    return state


def test_all_connected_returns_empty(ca_reads):
    ca_reads["values"] = {
        "uc_ref": "Connected",
        "uc_contrib": "Connected",
        "uc_snap": "Connected",
    }
    assert _preflight_connected(_Session(), _config()) == []
    # One concurrent batch — never one read per device (RE-loop blocking).
    assert ca_reads["batches"] == 1


def test_dead_reference_refuses(ca_reads):
    ca_reads["values"] = {"uc_ref": "Disconnected", "uc_contrib": "Connected"}
    with pytest.raises(GeecsDeviceDownError, match="UC_Ref") as excinfo:
        _preflight_connected(_Session(), _config())
    assert excinfo.value.device_name == "UC_Ref"
    # Pre-claim refusal message must say no scan number was burned.
    assert "no scan number" in str(excinfo.value)


def test_dead_sync_contributor_refuses(ca_reads):
    # Mode-independent: a dead sync contributor would die at t0 sync
    # post-claim (stale cache blows the spread window; the seed gate
    # refuses it) — a warn-and-continue here would be a fiction, so the
    # refusal happens pre-claim, named.
    ca_reads["values"] = {"uc_ref": "Connected", "uc_contrib": "Disconnected"}
    with pytest.raises(GeecsDeviceDownError, match="UC_Contrib"):
        _preflight_connected(_Session(), _config())


def test_dead_snapshot_device_warns_and_continues(ca_reads, caplog):
    import logging

    ca_reads["values"] = {
        "uc_ref": "Connected",
        "uc_contrib": "Connected",
        "uc_snap": "Disconnected",
    }
    with caplog.at_level(logging.WARNING, logger="geecs_bluesky.scan_request_runner"):
        down = _preflight_connected(_Session(), _config())
    assert down == ["UC_Snap"]
    assert any("UC_Snap" in r.getMessage() for r in caplog.records)


def test_unreadable_is_fail_open(ca_reads):
    # No values set → every read returns None: not a verdict (the liveness
    # doctrine) — the scan proceeds with nothing recorded.
    assert _preflight_connected(_Session(), _config()) == []
    assert ca_reads["batches"] == 1


def test_no_experiment_skips_without_reading(ca_reads):
    assert _preflight_connected(_Session(None), _config()) == []
    assert ca_reads["batches"] == 0


def test_empty_config_skips_without_reading(ca_reads):
    assert _preflight_connected(_Session(), {}) == []
    assert ca_reads["batches"] == 0
