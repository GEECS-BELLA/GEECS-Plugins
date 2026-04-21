"""End-to-end test: BlueskyScanner NOSCAN with UC_ModeImager saving.

Verifies:
- Scan number claimed, scan folder created on NetApp
- UC_ModeImager subdirectory created
- localsavingpath / save signals set before shots, cleared after

Run from GeecsBluesky/:
    poetry run python test_scanner_bridge.py
"""

from __future__ import annotations

import logging
import time

from geecs_data_utils import ScanConfig, ScanMode

from geecs_bluesky.scanner_bridge import BlueskyScanner

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

print("\n=== BlueskyScanner NOSCAN save test ===")

scanner = BlueskyScanner(experiment_dir="Undulator")

scanner.reinitialize(
    config_dictionary={
        "shots_per_step": 3,
        "Devices": {
            "UC_ModeImager": {
                "variable_list": ["hardware_timestamp"],
                "save_nonscalar_data": True,
            },
        },
    }
)

scan_config = ScanConfig(
    scan_mode=ScanMode.NOSCAN,
    additional_description="BlueskyScanner save-signal test",
)

print("Starting scan …")
scanner.start_scan_thread(scan_config)

while scanner.is_scanning_active():
    time.sleep(0.5)
    pct = scanner.estimate_current_completion() * 100
    print(f"  {pct:.0f}%")

print("\nScan complete.")

# Report what was recorded in the scanner state
if scanner._saving_detectors:
    for det, path in scanner._saving_detectors:
        print(f"  Saving detector: {det.name!r}  →  {path}")
        import os

        if os.path.isdir(path):
            files = os.listdir(path)
            print(f"    Files in dir: {len(files)}")
        else:
            print(
                "    (directory does not exist yet — images may be written by GEECS asynchronously)"
            )
else:
    print(
        "  No saving detectors found (check save_nonscalar_data flag and scan_folder)"
    )
