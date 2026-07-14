"""Live A/B verification of read-path staging (landed as PR #541, 2026-07-13).

**Requires lab network, a running gateway, and a FREE-RUNNING trigger** (the
reference camera must be seeing shots — check with the trigger probe below
before running; coordinate with the operator).  Otherwise read-only:
``save_data=False`` claims no scan folder, performs no shot-control puts,
and writes nothing to Tiled — it purely observes the free-running cameras,
with N_LOAD snapshot devices over real PVs emulating a telemetry read load.

Run A: staging bypassed — reproduces #540 when the row cycle exceeds one
trigger period (camera deltas quantize to exact multiples of the period;
at N_LOAD=120 over VPN, expect exactly 2.0 s at a 1 Hz trigger).
Run B: staged (the production code path) — expect rows on every tick.

Run from the GeecsBluesky poetry env (extras: ca)::

    poetry run python scripts/verify_staging_live.py

Historical results (2026-07-13, VPN): A median 2.00 s/row, B 1.00 s/row;
isolated read cost 62 devices: 527 ms unstaged → 6.7 ms staged (79x).
Adjust CAM_IN/CAM_OUT for the experiment's live cameras.
"""

import logging
import statistics
import time

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("geecs_bluesky.plans.free_run_step_scan").setLevel(logging.DEBUG)

import bluesky.preprocessors as bpp  # noqa: E402

import geecs_bluesky.plans.orchestration as orch  # noqa: E402
from geecs_bluesky.session import GeecsSession  # noqa: E402

CAM_IN = "UC_Amp4_IR_input"
CAM_OUT = "UC_Amp4_IR_output"
N_LOAD = 120
SHOTS = 8


def run_scan(staged: bool) -> dict:
    """Run one A/B scan; return connect/scan timing + camera deltas."""
    if staged:
        orch.bpp = bpp  # the real stage_wrapper
    else:

        class _NoStage:
            def __getattr__(self, name):
                if name == "stage_wrapper":
                    return lambda plan, devices: plan
                return getattr(bpp, name)

        orch.bpp = _NoStage()

    session = GeecsSession("Undulator", rep_rate_hz=1.0, tiled=False)
    events: list[tuple[float, dict]] = []
    session.RE.subscribe(
        lambda name, doc: events.append((time.monotonic(), doc))
        if name == "event"
        else None
    )

    t_connect0 = time.monotonic()
    ref = session.detector(CAM_IN, ["MaxCounts"])
    cam = session.contributor(CAM_OUT, ["MaxCounts"])
    load = [
        session.snapshot(
            CAM_IN if i % 2 == 0 else CAM_OUT,
            ["MaxCounts"],
            name=f"load_{i:02d}",
        )
        for i in range(N_LOAD)
    ]
    connect_s = time.monotonic() - t_connect0

    devices = [ref, cam, *load]
    try:
        t0 = time.monotonic()
        session.scan(
            detectors=devices,
            motor=None,
            positions=[None],
            shots_per_step=SHOTS,
            mode="free_run",
            save_data=False,
            description=f"staging verification ({'staged' if staged else 'baseline'})",
        )
        total_s = time.monotonic() - t0
    finally:
        session.disconnect(*devices)

    # Primary-row cadence from the reference camera's own timestamps.
    ref_key = f"{ref.name}-acq_timestamp"
    ts = [doc["data"][ref_key] for _, doc in events if ref_key in doc["data"]]
    deltas = [round(b - a, 3) for a, b in zip(ts, ts[1:])]
    walls = [t for t, doc in events if ref_key in doc["data"]]
    wall_deltas = [round(b - a, 3) for a, b in zip(walls, walls[1:])]
    return {
        "connect_s": round(connect_s, 1),
        "total_s": round(total_s, 1),
        "cam_deltas": deltas,
        "wall_deltas": wall_deltas,
        "median_delta": statistics.median(deltas) if deltas else None,
    }


print(f"\n=== Run A: baseline (staging bypassed), {N_LOAD} load devices ===")
a = run_scan(staged=False)
print(f"A: connect {a['connect_s']}s, scan {a['total_s']}s")
print(f"A: camera deltas {a['cam_deltas']}  (median {a['median_delta']})")
print(f"A: wall-row deltas {a['wall_deltas']}")

print(f"\n=== Run B: staged (PR #541), {N_LOAD} load devices ===")
b = run_scan(staged=True)
print(f"B: connect {b['connect_s']}s, scan {b['total_s']}s")
print(f"B: camera deltas {b['cam_deltas']}  (median {b['median_delta']})")
print(f"B: wall-row deltas {b['wall_deltas']}")

print(
    f"\nVERDICT: baseline median {a['median_delta']}s/row -> "
    f"staged median {b['median_delta']}s/row at a 1.0s trigger period"
)
