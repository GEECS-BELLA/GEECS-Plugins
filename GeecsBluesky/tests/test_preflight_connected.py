"""Pin the pre-claim CONNECTED liveness re-check (#664).

The check runs at execution on both paths (runner + queue plan) because the
client-side pre-submit read cannot cover the submission-to-execution gap —
live incident 2026-08-22: a camera server rebooted while items sat queued
and the dead reference surfaced only as a mid-scan trigger timeout.

CA reads are faked at the oneshot seam (the check imports it lazily, so a
module-attribute patch lands).
"""

from __future__ import annotations

import pytest

from geecs_bluesky.exceptions import GeecsDeviceDownError
from geecs_bluesky.scan_request_runner import _preflight_connected


class _Session:
    def __init__(self, experiment: str | None = "Test") -> None:
        self.experiment = experiment


def _config() -> dict[str, dict]:
    # Role-derived order (save_set_to_devices_config's contract): the
    # reference is the FIRST synchronous entry.
    return {
        "UC_Ref": {"synchronous": True, "variable_list": ["a"]},
        "UC_Contrib": {"synchronous": True, "variable_list": ["b"]},
        "UC_Snap": {"synchronous": False, "variable_list": ["c"]},
    }


@pytest.fixture
def ca_reads(monkeypatch):
    """Fake the oneshot read: per-device CONNECTED choice strings."""
    state = {"values": {}, "calls": 0}

    def fake(pv, *, timeout, datatype=None):
        state["calls"] += 1
        # The enum contract: the check must read the choice STRING.
        assert datatype is str
        device = pv.split(":")[-2]
        return state["values"].get(device)

    monkeypatch.setattr("geecs_bluesky.devices.ca.oneshot.try_caget_once", fake)
    return state


def test_all_connected_returns_empty(ca_reads):
    ca_reads["values"] = {
        "uc_ref": "Connected",
        "uc_contrib": "Connected",
        "uc_snap": "Connected",
    }
    assert _preflight_connected(_Session(), _config(), free_run=True) == []


def test_free_run_dead_reference_refuses(ca_reads):
    ca_reads["values"] = {"uc_ref": "Disconnected", "uc_contrib": "Connected"}
    with pytest.raises(GeecsDeviceDownError, match="UC_Ref") as excinfo:
        _preflight_connected(_Session(), _config(), free_run=True)
    assert excinfo.value.device_name == "UC_Ref"
    # Pre-claim refusal message must say no scan number was burned.
    assert "no scan number" in str(excinfo.value)


def test_free_run_dead_contributor_warns_and_continues(ca_reads, caplog):
    ca_reads["values"] = {"uc_ref": "Connected", "uc_contrib": "Disconnected"}
    import logging

    with caplog.at_level(logging.WARNING, logger="geecs_bluesky.scan_request_runner"):
        down = _preflight_connected(_Session(), _config(), free_run=True)
    assert down == ["UC_Contrib"]
    assert any("UC_Contrib" in r.getMessage() for r in caplog.records)


def test_strict_dead_sync_device_refuses(ca_reads):
    # Strict rows await ALL synchronous devices — any dead one is fatal.
    ca_reads["values"] = {"uc_ref": "Connected", "uc_contrib": "Disconnected"}
    with pytest.raises(GeecsDeviceDownError, match="UC_Contrib"):
        _preflight_connected(_Session(), _config(), free_run=False)


def test_dead_snapshot_device_warns_in_both_modes(ca_reads):
    ca_reads["values"] = {
        "uc_ref": "Connected",
        "uc_contrib": "Connected",
        "uc_snap": "Disconnected",
    }
    assert _preflight_connected(_Session(), _config(), free_run=True) == ["UC_Snap"]
    assert _preflight_connected(_Session(), _config(), free_run=False) == ["UC_Snap"]


def test_unreadable_is_fail_open(ca_reads):
    # No values set → every read returns None: not a verdict (the liveness
    # doctrine) — the scan proceeds with nothing recorded.
    assert _preflight_connected(_Session(), _config(), free_run=True) == []
    assert ca_reads["calls"] == len(_config())


def test_no_experiment_skips_without_reading(ca_reads):
    assert _preflight_connected(_Session(None), _config(), free_run=True) == []
    assert ca_reads["calls"] == 0


def test_empty_config_skips_without_reading(ca_reads):
    assert _preflight_connected(_Session(), {}, free_run=True) == []
    assert ca_reads["calls"] == 0
