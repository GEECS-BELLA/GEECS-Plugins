"""BlueskyScanner bridge — hardware integration test.

Exercises the ScanManager-compatible API that RunControl uses, without launching
the GUI. Requires lab network access (GEECS DB + one camera + U_ESP_JetXYZ +
U_DG645_ShotControl). The camera defaults to UC_TopView and can be overridden
with ``GEECS_BLUESKY_TEST_CAMERA``.

Scenarios
---------
1. NOSCAN  — fixed-position collection from the camera (acq_timestamp only)
             Verifies: N events collected.
2. STANDARD — step scan U_ESP_JetXYZ 4→5 mm with the camera as detector
             Verifies: N positions × M shots events, motor readback in each event.
3. NOSCAN with shot control — as scenario 1 but with U_DG645_ShotControl wired;
             Verifies: DG645 Trigger.Source returns to STANDBY after scan.

Run from GeecsBluesky/:
    poetry run python test_bluesky_scanner.py
    GEECS_BLUESKY_TEST_CAMERA=UC_Amp2_IR_input poetry run python test_bluesky_scanner.py
    poetry run pytest test_bluesky_scanner.py -m "integration and hardware" -v
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from types import SimpleNamespace

import pytest

from geecs_bluesky.db.geecs_db import GeecsDb
from geecs_bluesky.scanner_bridge import BlueskyScanner
from geecs_bluesky.transport.udp_client import GeecsUdpClient

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

TEST_CAMERA_DEVICE = os.environ.get("GEECS_BLUESKY_TEST_CAMERA", "UC_TopView")

DETECTOR_DEVICES = {
    TEST_CAMERA_DEVICE: {
        "variable_list": ["acq_timestamp"],
        "save_nonscalar_data": False,
    },
}

SHOT_CONTROL_INFO = {
    "device": "U_DG645_ShotControl",
    "variables": {
        "Trigger.ExecuteSingleShot": {
            "OFF": "",
            "SCAN": "",
            "SINGLESHOT": "on",
            "STANDBY": "",
        },
        "Trigger.Source": {
            "OFF": "Single shot external rising edges",
            "SCAN": "External rising edges",
            "STANDBY": "External rising edges",
        },
    },
}

SHOTS_PER_STEP = 3
REP_RATE_HZ = 1.0

PREFLIGHT_READS = {
    TEST_CAMERA_DEVICE: ("acq_timestamp",),
    "U_ESP_JetXYZ": ("Position.Axis 1",),
    "U_DG645_ShotControl": ("Trigger.Source",),
}


def _make_exec_config(
    scan_mode: str,
    wait_time: float = SHOTS_PER_STEP / REP_RATE_HZ,
    device_var: str | None = None,
    start: float = 0.0,
    end: float = 1.0,
    step: float = 0.5,
    description: str = "",
) -> SimpleNamespace:
    """Build a fully duck-typed ScanExecutionConfig — no geecs_scanner import needed."""
    scan_config = SimpleNamespace(
        scan_mode=scan_mode,
        device_var=device_var,
        start=start,
        end=end,
        step=step,
        wait_time=wait_time,
        additional_description=description,
    )
    return SimpleNamespace(
        scan_config=scan_config,
        options=SimpleNamespace(rep_rate_hz=REP_RATE_HZ),
        save_config=SimpleNamespace(Devices=DETECTOR_DEVICES),
    )


def poll(scanner: BlueskyScanner) -> None:
    """Block until the scanner finishes, printing progress."""
    while scanner.is_scanning_active():
        pct = scanner.estimate_current_completion() * 100
        print(f"  {pct:.0f}%", end="\r")
        time.sleep(0.5)
    print()


async def _preflight_device(device_name: str, variables: tuple[str, ...]) -> None:
    """Verify one required hardware device is reachable before scan scenarios."""
    host, port = GeecsDb.find_device(device_name)
    udp = GeecsUdpClient(host, port, device_name=device_name)
    await udp.connect()
    try:
        for variable in variables:
            await udp.get(variable)
    finally:
        await udp.close()


def preflight_hardware() -> bool:
    """Return whether all required hardware devices are reachable."""
    ok = True
    print("\n=== Hardware preflight ===")
    for device_name, variables in PREFLIGHT_READS.items():
        try:
            asyncio.run(_preflight_device(device_name, variables))
        except Exception as exc:
            ok = False
            print(f"  ✗  {device_name}: unavailable ({exc})")
        else:
            print(f"  ✓  {device_name}: reachable")
    if not ok:
        print("\nHardware preflight failed; turn on the devices above and rerun.")
    return ok


def run_hardware_scan_checks() -> list[str]:
    """Run the hardware checks and return any failed labels."""
    passed: list[str] = []
    failed: list[str] = []

    def check(label: str, condition: bool, detail: str = "") -> None:
        """Print pass/fail and accumulate results."""
        if condition:
            print(f"  ✓  {label}")
            passed.append(label)
        else:
            msg = label + (f" — {detail}" if detail else "")
            print(f"  ✗  {msg}")
            failed.append(msg)

    if not preflight_hardware():
        return ["hardware preflight"]

    print(f"\n=== Scenario 1: NOSCAN — {TEST_CAMERA_DEVICE}, no shot control ===")

    scanner1 = BlueskyScanner(experiment_dir="Undulator")
    scanner1.reinitialize(
        exec_config=_make_exec_config(
            "noscan",
            description="BlueskyScanner NOSCAN test",
        )
    )
    scanner1.start_scan_thread()
    poll(scanner1)

    check(
        "NOSCAN: event count",
        scanner1._completed_shots == SHOTS_PER_STEP,
        f"expected {SHOTS_PER_STEP}, got {scanner1._completed_shots}",
    )

    print(f"\n=== Scenario 2: STANDARD — JetXYZ 4→5 mm, {TEST_CAMERA_DEVICE} ===")

    positions = 3  # 4.0, 4.5, 5.0
    expected = positions * SHOTS_PER_STEP

    scanner2 = BlueskyScanner(experiment_dir="Undulator")
    events2: list[dict] = []
    scanner2._RE.subscribe(
        lambda name, doc: events2.append(doc) if name == "event" else None
    )

    scanner2.reinitialize(
        exec_config=_make_exec_config(
            "standard",
            device_var="U_ESP_JetXYZ:Position.Axis 1",
            start=4.0,
            end=5.0,
            step=0.5,
            description="BlueskyScanner STANDARD test",
        )
    )
    scanner2.start_scan_thread()
    poll(scanner2)

    check(
        "STANDARD: event count",
        len(events2) == expected,
        f"expected {expected}, got {len(events2)}",
    )
    motor_key = next(
        (k for k in (events2[0]["data"] if events2 else {}) if "position" in k),
        None,
    )
    check(
        "STANDARD: motor readback in every event",
        motor_key is not None and all(motor_key in e["data"] for e in events2),
    )
    check(
        "STANDARD: acq_timestamp in every event",
        all(any("acq_timestamp" in k for k in e["data"]) for e in events2),
    )

    print("\n=== Scenario 3: NOSCAN with shot control (U_DG645_ShotControl) ===")

    scanner3 = BlueskyScanner(
        experiment_dir="Undulator",
        shot_control_information=SHOT_CONTROL_INFO,
    )
    scanner3.reinitialize(
        exec_config=_make_exec_config(
            "noscan",
            description="BlueskyScanner shot-control test",
        )
    )

    events3: list[dict] = []
    scanner3._RE.subscribe(
        lambda name, doc: events3.append(doc) if name == "event" else None
    )

    scanner3.start_scan_thread()
    poll(scanner3)

    check(
        "SHOT CTRL NOSCAN: event count",
        scanner3._completed_shots == SHOTS_PER_STEP,
        f"expected {SHOTS_PER_STEP}, got {scanner3._completed_shots}",
    )

    try:
        host, port = GeecsDb.find_device("U_DG645_ShotControl")
        udp = GeecsUdpClient(host, port, device_name="U_DG645_ShotControl")

        async def _read_trigger_source() -> str:
            await udp.connect()
            val = await udp.get("Trigger.Source")
            await udp.close()
            return str(val)

        trigger_source = asyncio.run(_read_trigger_source())
        print(f"  DG645 Trigger.Source after scan: {trigger_source!r}")
        check(
            "SHOT CTRL NOSCAN: Trigger.Source returned to STANDBY value",
            trigger_source == "External rising edges",
            f"got {trigger_source!r}",
        )
    except Exception as exc:
        print(f"  Could not verify DG645 state: {exc}")
        failed.append("SHOT CTRL NOSCAN: DG645 post-scan readback (exception)")

    print(f"\n=== Results: {len(passed)} passed, {len(failed)} failed ===")
    for failure in failed:
        print(f"  FAILED: {failure}")
    if not failed:
        print("ALL CHECKS PASSED")
    return failed


@pytest.mark.integration
@pytest.mark.hardware
def test_bluesky_scanner_hardware_integration() -> None:
    """Run the real-lab BlueskyScanner hardware smoke checks."""
    assert not run_hardware_scan_checks()


if __name__ == "__main__":
    raise SystemExit(1 if run_hardware_scan_checks() else 0)
