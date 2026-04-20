"""Standalone test for BlueskyScanner bridge (real hardware).

Exercises the same API surface that RunControl uses, without launching the GUI:
  - reinitialize()
  - start_scan_thread()
  - is_scanning_active() / estimate_current_completion() polling
  - scan completes → devices disconnected automatically

Run from the GeecsBluesky directory:
    poetry run python test_bluesky_scanner.py
"""

from __future__ import annotations

import logging
import time
from types import SimpleNamespace

from geecs_bluesky.scanner_bridge import BlueskyScanner


# Minimal stand-in for ScanConfig/ScanMode — avoids the pandas dependency
# in geecs_data_utils when running from the GeecsBluesky poetry venv.
# In the real GEECS Scanner GUI, the genuine ScanConfig is passed.
class _ScanMode:
    STANDARD = SimpleNamespace(value="standard")
    NOSCAN = SimpleNamespace(value="noscan")


def ScanConfig(  # noqa: N802
    scan_mode, device_var, start, end, step, wait_time=1.0, additional_description=None
):
    """Build a minimal ScanConfig-compatible namespace for offline testing."""
    return SimpleNamespace(
        scan_mode=scan_mode,
        device_var=device_var,
        start=start,
        end=end,
        step=step,
        wait_time=wait_time,
        additional_description=additional_description,
    )


ScanMode = _ScanMode

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

# ──────────────────────────────────────────────────────────────────────────────
# 1. Create scanner (RunEngine starts here)
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Creating BlueskyScanner ===")
scanner = BlueskyScanner(experiment_dir="Undulator")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Reinitialize (mirroring RunControl.submit_run → scan_manager.reinitialize)
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Reinitializing ===")
ok = scanner.reinitialize(
    config_path=None,
    config_dictionary={"shots_per_step": 3},
)
print(f"reinitialize() → {ok}")
assert ok, "reinitialize() must return True"

# ──────────────────────────────────────────────────────────────────────────────
# 3. Submit a STANDARD step scan
# ──────────────────────────────────────────────────────────────────────────────
scan_config = ScanConfig(
    scan_mode=ScanMode.STANDARD,
    device_var="U_ESP_JetXYZ:Position.Axis 1",
    start=4.0,
    end=5.0,
    step=0.5,  # 3 positions: 4.0, 4.5, 5.0
    wait_time=1.0,
    additional_description="BlueskyScanner bridge test",
)

print(f"\n=== Starting scan: {scan_config} ===")
scanner.start_scan_thread(scan_config)

# ──────────────────────────────────────────────────────────────────────────────
# 4. Poll progress (same as the GUI timer would do)
# ──────────────────────────────────────────────────────────────────────────────
print("\n--- Polling progress ---")
while scanner.is_scanning_active():
    pct = scanner.estimate_current_completion() * 100
    print(f"  {pct:.0f}% complete", end="\r")
    time.sleep(0.5)

print("\n  100% complete")

# ──────────────────────────────────────────────────────────────────────────────
# 5. Results
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Done ===")
print(f"Events collected: {scanner._completed_shots}")
expected = 3 * 3  # 3 positions × 3 shots/step
if scanner._completed_shots == expected:
    print(f"ALL CHECKS PASSED ✓  ({expected} events)")
else:
    print(f"WARNING: expected {expected} events, got {scanner._completed_shots}")
