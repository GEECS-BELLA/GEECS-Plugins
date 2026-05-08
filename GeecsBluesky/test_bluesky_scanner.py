"""BlueskyScanner bridge — hardware integration test.

Exercises the ScanManager-compatible API that RunControl uses, without launching
the GUI.  Requires lab network access (GEECS DB + UC_TopView + U_ESP_JetXYZ +
U_DG645_ShotControl).

Scenarios
---------
1. NOSCAN  — fixed-position collection from UC_TopView (acq_timestamp only)
             Verifies: N events collected.
2. STANDARD — step scan U_ESP_JetXYZ 4→5 mm with UC_TopView as detector
             Verifies: N positions × M shots events, motor readback in each event.
3. NOSCAN with shot control — as scenario 1 but with U_DG645_ShotControl wired;
             Verifies: DG645 Trigger.Source returns to STANDBY after scan.

Run from GeecsBluesky/:
    poetry run python test_bluesky_scanner.py
"""

from __future__ import annotations

import logging
import time
from types import SimpleNamespace

from geecs_bluesky.scanner_bridge import BlueskyScanner

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

DETECTOR_DEVICES = {
    "UC_TopView": {
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


def poll(scanner: BlueskyScanner) -> None:
    """Block until the scanner finishes, printing progress."""
    while scanner.is_scanning_active():
        pct = scanner.estimate_current_completion() * 100
        print(f"  {pct:.0f}%", end="\r")
        time.sleep(0.5)
    print()


# ---------------------------------------------------------------------------
# Scenario 1: NOSCAN (no shot control)
# ---------------------------------------------------------------------------
print("\n=== Scenario 1: NOSCAN — UC_TopView, no shot control ===")

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

# ---------------------------------------------------------------------------
# Scenario 2: STANDARD step scan (no shot control)
# ---------------------------------------------------------------------------
print("\n=== Scenario 2: STANDARD — JetXYZ 4→5 mm, UC_TopView ===")

POSITIONS = 3  # 4.0, 4.5, 5.0
EXPECTED = POSITIONS * SHOTS_PER_STEP

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
    len(events2) == EXPECTED,
    f"expected {EXPECTED}, got {len(events2)}",
)
motor_key = next(
    (k for k in (events2[0]["data"] if events2 else {}) if "position" in k), None
)
check(
    "STANDARD: motor readback in every event",
    motor_key is not None and all(motor_key in e["data"] for e in events2),
)
check(
    "STANDARD: acq_timestamp in every event",
    all(any("acq_timestamp" in k for k in e["data"]) for e in events2),
)

# ---------------------------------------------------------------------------
# Scenario 3: NOSCAN with shot control (U_DG645_ShotControl)
# ---------------------------------------------------------------------------
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

# Subscribe to see what's happening
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

# Verify DG645 returned to STANDBY after scan.
# Query the device directly via its UDP client (re-use geecs_data_utils DB lookup).
try:
    import asyncio

    from geecs_bluesky.db.geecs_db import GeecsDb
    from geecs_bluesky.transport.udp_client import GeecsUdpClient

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

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n=== Results: {len(passed)} passed, {len(failed)} failed ===")
for f in failed:
    print(f"  FAILED: {f}")
if not failed:
    print("ALL CHECKS PASSED")
