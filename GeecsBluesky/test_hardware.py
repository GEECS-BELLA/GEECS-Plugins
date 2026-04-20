"""Hardware validation for the TCP-backed signal refactor.

Tests the refactored GeecsBluesky stack (TCP cache + event-driven trigger)
against real GEECS hardware at BELLA:
  - Motor: U_ESP_JetXYZ, Position.Axis 1 (4→6 mm, 4 steps)
  - Camera: UC_ModeImager (3 shots/step → 12 events)

Run from the GeecsBluesky directory:
    poetry run python test_hardware.py
"""

from __future__ import annotations

import asyncio
import logging
import pprint

from bluesky import RunEngine
from bluesky.callbacks import LiveTable

from geecs_bluesky.devices.motor import GeecsMotor
from geecs_bluesky.devices.mode_imager import ModeImager
from geecs_bluesky.plans.step_scan import geecs_step_scan

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

# ──────────────────────────────────────────────────────────────────────────────
# 1. Construct devices (from DB lookup)
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Constructing devices from DB ===")
motor = GeecsMotor.from_db_axis(
    "U_ESP_JetXYZ", "Position.Axis 1", name="jet_x", units="mm"
)
cam = ModeImager.from_db(name="mode_imager")
print(f"Motor: {motor.name}")
print(f"Camera: {cam.name}")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Connect devices in the RunEngine's persistent event loop
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Connecting devices ===")
RE = RunEngine()

asyncio.run_coroutine_threadsafe(motor.connect(), RE._loop).result(timeout=15)
print(f"Motor connected. Initial cache: {dict(motor._shot_cache)}")

asyncio.run_coroutine_threadsafe(cam.connect(), RE._loop).result(timeout=15)
print(f"Camera connected. Initial cache: {dict(cam._shot_cache)}")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Quick coherence check: read both devices once, print values
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Quick read (from TCP cache) ===")
motor_reading = asyncio.run_coroutine_threadsafe(motor.read(), RE._loop).result(
    timeout=5
)
cam_reading = asyncio.run_coroutine_threadsafe(cam.read(), RE._loop).result(timeout=5)
print("Motor reading:")
pprint.pprint(motor_reading)
print("Camera reading:")
pprint.pprint(cam_reading)

# ──────────────────────────────────────────────────────────────────────────────
# 4. Step scan: 4.0 → 6.0 mm, 4 steps, 3 shots/step → 12 events expected
# ──────────────────────────────────────────────────────────────────────────────
POSITIONS = [4.0, 4.667, 5.333, 6.0]
SHOTS_PER_STEP = 3

events: list[dict] = []


def _collect(name, doc):
    if name == "event":
        events.append(doc)
        idx = len(events)
        pos = doc["data"].get("jet_x-position", "?")
        ts = doc["data"].get("mode_imager-hardware_ts", "?")
        print(f"  event {idx:2d}: jet_x={pos:.4f} mm  hardware_ts={ts}")


RE.subscribe(_collect)

tbl = LiveTable(["jet_x-position", "mode_imager-hardware_ts"])
RE.subscribe(tbl)

print(f"\n=== Step scan: {POSITIONS} mm, {SHOTS_PER_STEP} shots/step ===")
RE(
    geecs_step_scan(
        motor=motor,
        positions=POSITIONS,
        detectors=[cam],
        shots_per_step=SHOTS_PER_STEP,
        md={"operator": "test", "purpose": "TCP-cache refactor validation"},
    )
)

# ──────────────────────────────────────────────────────────────────────────────
# 5. Validate results
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Results ===")
expected = len(POSITIONS) * SHOTS_PER_STEP
print(f"Events collected: {len(events)} / {expected} expected")

ok = True
for ev in events:
    if "jet_x-position" not in ev["data"]:
        print("  MISSING: jet_x-position in event")
        ok = False
    if "mode_imager-hardware_ts" not in ev["data"]:
        print("  MISSING: mode_imager-hardware_ts in event")
        ok = False

if len(events) == expected and ok:
    print("ALL CHECKS PASSED ✓")
else:
    print("SOME CHECKS FAILED ✗")

# ──────────────────────────────────────────────────────────────────────────────
# 6. Clean up
# ──────────────────────────────────────────────────────────────────────────────
asyncio.run_coroutine_threadsafe(motor.disconnect(), RE._loop).result(timeout=5)
asyncio.run_coroutine_threadsafe(cam.disconnect(), RE._loop).result(timeout=5)
print("\nDevices disconnected.")
