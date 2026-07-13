"""Pins for phase 4 — timestamp-only telemetry shot context (0.35.0).

Owner decisions (Planning/device_read_path/01_telemetry_attribution.md):
async devices carry NO derived labels (D1a); there is no classification
stage — strict synchronization stays a save-set concept, and telemetry
devices simply log their own ``acq_timestamp`` (D2, tabled); devices
observed to have fired (positive ``acq_timestamp`` at the quiesced t0
snapshot) get the full contributor companion set (D3).
"""

from __future__ import annotations

import pytest
from bluesky import RunEngine

pytest.importorskip("aioca")

from ophyd_async.core import set_mock_value  # noqa: E402

from geecs_bluesky.devices.ca import CaGenericDetector  # noqa: E402
from geecs_bluesky.devices.ca.telemetry import (  # noqa: E402
    CaTelemetryGroup,
    CaTelemetryReadable,
)
from geecs_bluesky.plans.orchestration import build_step_scan_plan  # noqa: E402
from geecs_bluesky.session import GeecsSession  # noqa: E402
from tests.ca_mock_helpers import connect_mock, start_pacer  # noqa: E402

REF_T0 = 1000.0
TEL_T0 = 1000.03


def _run_scan(fire_telemetry: bool) -> tuple[list[dict], dict]:
    """Mock free-run NOSCAN with one shot-context telemetry member."""
    RE = RunEngine()
    start_docs: list[dict] = []
    events: list[dict] = []

    def collect(name: str, doc: dict) -> None:
        if name == "start":
            start_docs.append(doc)
        elif name == "event":
            events.append(doc)

    RE.subscribe(collect)

    ref = CaGenericDetector("U_Ref", ["Sig"], name="ref")
    ref.configure_shot_id(rep_rate_hz=1.0)
    tel = CaTelemetryReadable(
        "U_Tel", ["Val"], name="telemetry_u_tel", shot_rep_rate_hz=1.0
    )
    connect_mock(RE, ref, tel)
    set_mock_value(ref.acq_timestamp, REF_T0)
    if fire_telemetry:
        set_mock_value(tel.acq_timestamp, TEL_T0)
    # else: the acq_timestamp mock stays at its 0.0 default — the gateway's
    # never-acquired placeholder.

    group = CaTelemetryGroup([tel])
    targets = [(ref, REF_T0)] + ([(tel, TEL_T0)] if fire_telemetry else [])
    pacer = start_pacer(RE, targets, initial_delay=1.0, interval=0.3)
    try:
        RE(
            build_step_scan_plan(
                strict=False,
                motor=None,
                positions=[None],
                reference=ref,
                detectors=[ref, group],
                shots_per_step=2,
                controller=None,
                experiment="",
                scan_number=None,
                scan_folder=None,
                saving_detectors=[],
            )
        )
    finally:
        pacer.cancel()
    return events, start_docs[0]


class TestFiredTelemetryGetsCompanions:
    def test_companions_present_and_valid(self) -> None:
        events, start = _run_scan(fire_telemetry=True)
        assert start["telemetry_shot_seeded"] == ["telemetry_u_tel"]
        primary = [e for e in events if "telemetry_u_tel-acq_timestamp" in e["data"]]
        assert primary
        row = primary[-1]["data"]
        assert row["telemetry_u_tel-acq_timestamp"] > 0
        assert row["telemetry_u_tel-shot_id"] == row["ref-shot_id"]
        assert row["telemetry_u_tel-shot_offset"] == 0
        assert row["telemetry_u_tel-valid"] is True


class TestNeverFiredTelemetryStaysPlain:
    def test_no_companions_no_labels(self) -> None:
        events, start = _run_scan(fire_telemetry=False)
        assert start["telemetry_shot_seeded"] == []
        row = events[0]["data"]
        # The raw acq_timestamp column exists and shows the honest 0.0
        # placeholder; NO derived labels are manufactured (owner call D1a).
        assert row["telemetry_u_tel-acq_timestamp"] == 0.0
        assert "telemetry_u_tel-shot_id" not in row
        assert "telemetry_u_tel-shot_offset" not in row
        assert "telemetry_u_tel-valid" not in row
        # Value columns unaffected either way.
        assert "telemetry_u_tel-val" in row


class TestSessionWiresShotContext:
    def test_telemetry_batch_configures_trackers(self) -> None:
        session = GeecsSession("Test", tiled=False, mock=True, rep_rate_hz=1.0)
        members = session.telemetry_batch({"U_A": ["X"]})
        assert len(members) == 1
        assert members[0].shot_id_tracker is not None
        assert hasattr(members[0], "acq_timestamp")
