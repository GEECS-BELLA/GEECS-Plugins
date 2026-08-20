"""Pins for read-path hardening phases 2 & 3 (0.34.0).

Phase 2 — ``CaTelemetryGroup``: the whole soft tier costs one RunEngine
``read`` message per row; merged event columns are identical to the
ungrouped layout (members keep their own names — the EVENT_SCHEMA.md
contract).

Phase 3 — t0 seed check: the free-run plan warns when the scan's first row
shows a contributor at ``shot_offset != 0`` (a seeding error is constant
from row one; a healthy scan is silent).
"""

from __future__ import annotations

import asyncio
import logging

import pytest
from bluesky import RunEngine

pytest.importorskip("aioca")

from ophyd_async.core import set_mock_value  # noqa: E402

from geecs_bluesky.devices.ca import (  # noqa: E402
    CaGenericDetector,
    CaTimestampedReadable,
)
from geecs_bluesky.devices.ca.telemetry import (  # noqa: E402
    CaTelemetryGroup,
    CaTelemetryReadable,
)
from geecs_bluesky.plans.orchestration import build_step_scan_plan  # noqa: E402
from tests.ca_mock_helpers import connect_mock, start_pacer  # noqa: E402

REF_T0 = 1000.0
CAM_T0 = 1000.05


def _run_on(loop_owner: RunEngine, coro):
    return asyncio.run_coroutine_threadsafe(coro, loop_owner._loop).result(timeout=10.0)


class TestTelemetryGroup:
    """One read Msg for the soft tier; columns identical to ungrouped."""

    def _members(self, RE: RunEngine) -> list[CaTelemetryReadable]:
        members = [
            CaTelemetryReadable("U_Dev1", ["A", "B"], name="telemetry_u_dev1"),
            CaTelemetryReadable("U_Dev2", ["C"], name="telemetry_u_dev2"),
        ]
        connect_mock(RE, *members)
        return members

    def test_group_merges_readings_and_datakeys_unrenamed(self) -> None:
        RE = RunEngine()
        members = self._members(RE)
        group = CaTelemetryGroup(members)

        member_desc: dict = {}
        member_read: dict = {}
        for m in members:
            member_desc.update(_run_on(RE, m.describe()))
            member_read.update(_run_on(RE, m.read()))
        group_desc = _run_on(RE, group.describe())
        group_read = _run_on(RE, group.read())

        # Byte-identical column layout: same keys, member names untouched.
        assert set(group_desc) == set(member_desc)
        assert set(group_read) == set(member_read)
        assert all(k.startswith("telemetry_u_dev") for k in group_desc)

    def test_group_stage_serves_member_reads_from_cache(self) -> None:
        RE = RunEngine()
        members = self._members(RE)
        group = CaTelemetryGroup(members)

        async def stage_and_probe() -> tuple:
            await group.stage()
            set_mock_value(members[0]._telemetry_signals[0], 42.0)
            reading = await group.read()
            await group.unstage()
            return reading

        reading = _run_on(RE, stage_and_probe())
        key = members[0]._telemetry_signals[0].name
        # The staged cache tracked the monitor update.
        assert reading[key]["value"] == 42.0

    def test_group_costs_one_read_msg_per_row(self) -> None:
        RE = RunEngine()
        commands: list[str] = []
        RE.msg_hook = lambda msg: commands.append(msg.command)

        ref = CaGenericDetector("U_Ref", ["Sig"], name="ref")
        cam = CaTimestampedReadable("U_Cam", ["Val"], name="cam")
        ref.configure_shot_id(rep_rate_hz=1.0)
        cam.configure_shot_id(rep_rate_hz=1.0)
        cam.set_reference(ref, grace_wait_s=0.05)
        connect_mock(RE, ref, cam)
        members = self._members(RE)
        group = CaTelemetryGroup(members)
        set_mock_value(ref.acq_timestamp, REF_T0)
        set_mock_value(cam.acq_timestamp, CAM_T0)

        pacer = start_pacer(
            RE, [(ref, REF_T0), (cam, CAM_T0)], initial_delay=1.0, interval=0.3
        )
        try:
            RE(
                build_step_scan_plan(
                    strict=False,
                    motor=None,
                    positions=[None],
                    reference=ref,
                    detectors=[ref, cam, group],
                    shots_per_step=1,
                    controller=None,
                    experiment="",
                    scan_number=None,
                    scan_folder=None,
                    saving_detectors=[],
                )
            )
        finally:
            pacer.cancel()

        # Primary row + tail flush each read: ref, cam, group, scan_context
        # — 4 reads/row regardless of how many devices the group holds.
        assert commands.count("read") == 8

    def test_group_disconnect_forwards_to_members(self) -> None:
        RE = RunEngine()
        members = self._members(RE)
        group = CaTelemetryGroup(members)
        _run_on(RE, group.disconnect())  # must not raise; forwards to members


class TestT0SeedCheck:
    """First-row shot_offset != 0 warns; a healthy first row is silent."""

    def _scan(self, fire_cam: bool, caplog) -> None:
        RE = RunEngine()
        ref = CaGenericDetector("U_Ref", ["Sig"], name="ref")
        cam = CaTimestampedReadable("U_Cam", ["Val"], name="cam")
        ref.configure_shot_id(rep_rate_hz=1.0)
        cam.configure_shot_id(rep_rate_hz=1.0)
        cam.set_reference(ref, grace_wait_s=0.05)
        connect_mock(RE, ref, cam)
        set_mock_value(ref.acq_timestamp, REF_T0)
        set_mock_value(cam.acq_timestamp, CAM_T0)

        targets = [(ref, REF_T0)] + ([(cam, CAM_T0)] if fire_cam else [])
        pacer = start_pacer(RE, targets, initial_delay=1.0, interval=0.3)
        try:
            with caplog.at_level(
                logging.WARNING, logger="geecs_bluesky.plans.free_run_step_scan"
            ):
                RE(
                    build_step_scan_plan(
                        strict=False,
                        motor=None,
                        positions=[None],
                        reference=ref,
                        detectors=[ref, cam],
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

    def test_lagging_contributor_warns_on_first_row(self, caplog) -> None:
        self._scan(fire_cam=False, caplog=caplog)
        warnings = [r for r in caplog.records if "t0 seed check" in r.getMessage()]
        assert warnings and "cam" in warnings[0].getMessage()

    def test_healthy_first_row_is_silent(self, caplog) -> None:
        self._scan(fire_cam=True, caplog=caplog)
        assert not [r for r in caplog.records if "t0 seed check" in r.getMessage()]
