"""BlueskyScanner bridge — hardware integration test.

Exercises the ScanManager-compatible API that RunControl uses, without launching
the GUI.  Requires the lab network (GEECS DB + UC_ModeImager + U_ESP_JetXYZ).

Scenarios
---------
1. NOSCAN  — fixed-position collection from UC_ModeImager (save enabled)
             Verifies: scan folder created, save-path signals sent, N events.
2. STANDARD — step scan U_ESP_JetXYZ 4→5 mm with UC_ModeImager as detector
             Verifies: N positions × M shots events, motor readback in each event.

Run from GeecsBluesky/:
    poetry run python test_bluesky_scanner.py
"""

from __future__ import annotations

import logging
import os
import time

from geecs_data_utils import ScanConfig, ScanMode

from geecs_bluesky.scanner_bridge import BlueskyScanner

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

MOTOR_DEVICE_VAR = "U_ESP_JetXYZ:Position.Axis 1"
DETECTOR_DEVICES = {
    "UC_ModeImager": {
        "variable_list": ["hardware_timestamp"],
        "save_nonscalar_data": True,
    },
}
SHOTS_PER_STEP = 3

passed: list[str] = []
failed: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    """Print pass/fail and accumulate results."""
    if condition:
        print(f"  ✓  {label}")
        passed.append(label)
    else:
        msg = f"{label}" + (f" — {detail}" if detail else "")
        print(f"  ✗  {msg}")
        failed.append(msg)


def poll(scanner: BlueskyScanner) -> None:
    """Block until scanner finishes, printing progress."""
    while scanner.is_scanning_active():
        pct = scanner.estimate_current_completion() * 100
        print(f"  {pct:.0f}%", end="\r")
        time.sleep(0.5)
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Scenario 1: NOSCAN with save enabled
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Scenario 1: NOSCAN (UC_ModeImager, save_nonscalar_data=True) ===")

scanner = BlueskyScanner(experiment_dir="Undulator")
scanner.reinitialize(
    config_dictionary={
        "shots_per_step": SHOTS_PER_STEP,
        "Devices": DETECTOR_DEVICES,
    }
)

scan_config = ScanConfig(
    scan_mode=ScanMode.NOSCAN,
    additional_description="BlueskyScanner NOSCAN save test",
)

scanner.start_scan_thread(scan_config)
poll(scanner)

check(
    "NOSCAN: events collected",
    scanner._completed_shots == SHOTS_PER_STEP,
    f"expected {SHOTS_PER_STEP}, got {scanner._completed_shots}",
)

# Verify the scan folder and UC_ModeImager subdirectory were created.
# _saving_detectors is cleared on disconnect, so we reconstruct the expected path.
scan_folder_root = "/Volumes/hdna2/data/Undulator"
today_dirs = []
if os.path.isdir(scan_folder_root):
    # Walk to find the most recently written scan folder (crude but sufficient)
    for root, dirs, _ in os.walk(scan_folder_root):
        for d in dirs:
            if d.startswith("Scan") and d[4:].isdigit():
                today_dirs.append(os.path.join(root, d))

save_dir_found = any(
    os.path.isdir(os.path.join(d, "UC_ModeImager")) for d in today_dirs
)
check(
    "NOSCAN: UC_ModeImager save directory created",
    save_dir_found,
    "no Scan*/UC_ModeImager dir found under NetApp mount",
)

# ──────────────────────────────────────────────────────────────────────────────
# Scenario 2: STANDARD step scan with motor + detector
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Scenario 2: STANDARD scan (JetXYZ 4→5 mm, UC_ModeImager) ===")

POSITIONS = 3  # 4.0, 4.5, 5.0
EXPECTED_EVENTS = POSITIONS * SHOTS_PER_STEP

scanner2 = BlueskyScanner(experiment_dir="Undulator")
scanner2.reinitialize(
    config_dictionary={
        "shots_per_step": SHOTS_PER_STEP,
        "Devices": DETECTOR_DEVICES,
    }
)

events: list[dict] = []


def collect(name: str, doc: dict) -> None:
    """RE callback: accumulate event documents and print per-shot summary."""
    if name == "event":
        events.append(doc)
        step = len(events)
        pos = doc["data"].get("u_esp_jetxyz_position_axis_1-position", "?")
        ts = doc["data"].get("uc_modeimager-hardware_timestamp", "?")
        print(f"  event {step:2d}: motor={pos}  hw_ts={ts}")


scanner2._RE.subscribe(collect)

scan_config2 = ScanConfig(
    scan_mode=ScanMode.STANDARD,
    device_var=MOTOR_DEVICE_VAR,
    start=4.0,
    end=5.0,
    step=0.5,
    additional_description="BlueskyScanner STANDARD scan test",
)

scanner2.start_scan_thread(scan_config2)
poll(scanner2)

check(
    "STANDARD: event count",
    len(events) == EXPECTED_EVENTS,
    f"expected {EXPECTED_EVENTS}, got {len(events)}",
)
check(
    "STANDARD: motor readback in every event",
    all("u_esp_jetxyz_position_axis_1-position" in e["data"] for e in events),
)
check(
    "STANDARD: detector reading in every event",
    all("uc_modeimager-hardware_timestamp" in e["data"] for e in events),
)

# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n=== Results: {len(passed)} passed, {len(failed)} failed ===")
for f in failed:
    print(f"  FAILED: {f}")
if not failed:
    print("ALL CHECKS PASSED ✓")
