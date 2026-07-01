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
4. Full-output STANDARD — opt-in scan with native camera saving enabled.
             Verifies: scan folder, ScanInfo, scan.log, scalar files, s-file,
             saved images, and event save-path metadata.

Run from GeecsBluesky/:
    poetry run python test_bluesky_scanner.py
    GEECS_BLUESKY_TEST_CAMERA=UC_Amp2_IR_input poetry run python test_bluesky_scanner.py
    poetry run pytest test_bluesky_scanner.py -m "integration and hardware" -v
    poetry install --extras tiled
    GEECS_BLUESKY_TEST_CAMERA=UC_Amp2_IR_input GEECS_BLUESKY_FULL_OUTPUT_TEST=1 \
        poetry run pytest test_bluesky_scanner.py::test_bluesky_scanner_full_output_hardware_integration -v -s
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import time
from pathlib import Path
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
        "synchronous": True,
        "save_nonscalar_data": False,
    },
}

SHOT_CONTROL_INFO = {
    "device": "U_DG645_ShotControl",
    "variables": {
        "Trigger.ExecuteSingleShot": {
            "OFF": "",
            "SCAN": "",
            "ARMED": "",
            "SINGLESHOT": "on",
            "STANDBY": "",
        },
        "Trigger.Source": {
            "OFF": "Single shot external rising edges",
            "SCAN": "External rising edges",
            "ARMED": "Single shot external rising edges",
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
    detector_devices: dict | None = None,
    acquisition_mode: str = "free_run_time_sync",
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
        options=SimpleNamespace(
            rep_rate_hz=REP_RATE_HZ,
            acquisition_mode=acquisition_mode,
        ),
        save_config=SimpleNamespace(Devices=detector_devices or DETECTOR_DEVICES),
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
            acquisition_mode="strict_shot_control",
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


def _analysis_sfile_for(scan_folder: Path) -> Path:
    scan_number = int(scan_folder.name.removeprefix("Scan"))
    parts = list(scan_folder.parts)
    parts[-2] = "analysis"
    return Path(*parts[:-1]) / f"s{scan_number}.txt"


def _wait_for_nonempty_files(
    folder: Path, *, since: float, timeout: float = 15.0
) -> list[Path]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        files = [
            path
            for path in folder.rglob("*")
            if path.is_file()
            and path.stat().st_size > 0
            and path.stat().st_mtime >= since - 1.0
        ]
        if files:
            return sorted(files)
        time.sleep(0.5)
    return []


def _has_tiled_client() -> bool:
    """Return whether the Tiled client needed for scalar export is installed."""
    try:
        return importlib.util.find_spec("tiled.client") is not None
    except ModuleNotFoundError:
        return False


def run_hardware_full_output_checks() -> list[str]:
    """Run one live scan and verify on-disk scan artifacts."""
    passed: list[str] = []
    failed: list[str] = []

    def check(label: str, condition: bool, detail: str = "") -> None:
        if condition:
            print(f"  ✓  {label}")
            passed.append(label)
        else:
            msg = label + (f" — {detail}" if detail else "")
            print(f"  ✗  {msg}")
            failed.append(msg)

    if not _has_tiled_client():
        check(
            "FULL OUTPUT: Tiled client dependency is installed",
            False,
            "run `poetry install --extras tiled` before enabling "
            "GEECS_BLUESKY_FULL_OUTPUT_TEST=1",
        )
        return failed

    if not preflight_hardware():
        return ["hardware preflight"]

    print(
        f"\n=== Full-output STANDARD scan: U_ESP_JetXYZ fixed at 4 mm, "
        f"{TEST_CAMERA_DEVICE} saving enabled ==="
    )

    detector_devices = {
        TEST_CAMERA_DEVICE: {
            "variable_list": ["acq_timestamp"],
            "synchronous": True,
            "save_nonscalar_data": True,
        },
    }
    scanner = BlueskyScanner(
        experiment_dir="Undulator",
        shot_control_information=SHOT_CONTROL_INFO,
    )
    events: list[dict] = []
    scanner._RE.subscribe(
        lambda name, doc: events.append(doc) if name == "event" else None
    )
    scanner.reinitialize(
        exec_config=_make_exec_config(
            "standard",
            device_var="U_ESP_JetXYZ:Position.Axis 1",
            start=4.0,
            end=4.0,
            step=0.5,
            description="BlueskyScanner full-output hardware test",
            detector_devices=detector_devices,
            acquisition_mode="strict_shot_control",
        )
    )

    started_at = time.time()
    scanner.start_scan_thread()
    poll(scanner)

    check(
        "FULL OUTPUT: event count",
        len(events) == SHOTS_PER_STEP,
        f"expected {SHOTS_PER_STEP}, got {len(events)}",
    )
    save_path = scanner._nonscalar_save_paths.get(TEST_CAMERA_DEVICE)
    check(
        "FULL OUTPUT: scanner configured native-save path",
        bool(save_path),
    )
    if not save_path:
        return failed

    camera_folder = Path(save_path)
    scan_folder = camera_folder.parent
    scan_number = int(scan_folder.name.removeprefix("Scan"))
    scan_data_file = scan_folder / f"ScanDataScan{scan_number:03d}.txt"
    analysis_sfile = _analysis_sfile_for(scan_folder)
    scan_info = scan_folder / f"ScanInfo{scan_folder.name}.ini"
    scan_log = scan_folder / "scan.log"

    native_files = _wait_for_nonempty_files(camera_folder, since=started_at)
    print(f"  scan folder: {scan_folder}")
    print(f"  native camera folder: {camera_folder}")
    for path in native_files:
        print(f"  native file: {path}")

    check("FULL OUTPUT: scan folder exists", scan_folder.is_dir(), str(scan_folder))
    check("FULL OUTPUT: ScanInfo ini exists", scan_info.is_file(), str(scan_info))
    check(
        "FULL OUTPUT: scan.log exists",
        scan_log.is_file() and scan_log.stat().st_size > 0,
        str(scan_log),
    )
    check(
        "FULL OUTPUT: ScanData scalar file exists",
        scan_data_file.is_file() and scan_data_file.stat().st_size > 0,
        str(scan_data_file),
    )
    check(
        "FULL OUTPUT: analysis s-file exists",
        analysis_sfile.is_file() and analysis_sfile.stat().st_size > 0,
        str(analysis_sfile),
    )
    check(
        "FULL OUTPUT: native camera files saved",
        bool(native_files),
        str(camera_folder),
    )
    check(
        "FULL OUTPUT: nonscalar save path in events",
        all(
            any("nonscalar_save_path" in key for key in event["data"])
            for event in events
        ),
    )

    print(f"\n=== Full-output results: {len(passed)} passed, {len(failed)} failed ===")
    for failure in failed:
        print(f"  FAILED: {failure}")
    if not failed:
        print("ALL FULL-OUTPUT CHECKS PASSED")
    return failed


@pytest.mark.integration
@pytest.mark.hardware
def test_bluesky_scanner_hardware_integration() -> None:
    """Run the real-lab BlueskyScanner hardware smoke checks."""
    assert not run_hardware_scan_checks()


@pytest.mark.integration
@pytest.mark.hardware
def test_bluesky_scanner_full_output_hardware_integration() -> None:
    """Run a real scan and verify scan folder, scalar, log, and native files."""
    if os.environ.get("GEECS_BLUESKY_FULL_OUTPUT_TEST") != "1":
        pytest.skip("set GEECS_BLUESKY_FULL_OUTPUT_TEST=1 to run full-output scan")
    assert not run_hardware_full_output_checks()


if __name__ == "__main__":
    raise SystemExit(1 if run_hardware_scan_checks() else 0)
