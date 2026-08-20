"""Hermetic plan-layer benchmark for GeecsBluesky (zero network).

Runs the REAL geecs_step_scan plan (NOSCAN shape: motor=None, one no-move bin)
through a real RunEngine with ophyd-async MOCK-backend devices, so every
bps.read is an in-process cache hit.  Isolates pure framework overhead:
RE message processing + event-document assembly + subscription dispatch.

Sweeps:
  A. device count (4 signals each): 2, 10, 50, 137 devices x 50 events
  B. column shape at fixed/varied column counts (2x100, 2x392, 87x9)
  C. strict-shaped row (geecs_single_shot with a no-op fire) at 50 devices
Each case runs bare (timer callback only) and with a RunNormalizer->no-op
subscription (the synchronous front half of TiledWriter, incl. per-event
deepcopy + jsonschema validation) to measure emission overhead.

No Tiled client, no CA, no gateway, no hardware.
"""

from __future__ import annotations

import asyncio
import statistics
import time

import bluesky.plan_stubs as bps
from bluesky import RunEngine
from bluesky.callbacks.tiled_writer import RunNormalizer
from ophyd_async.core import StandardReadable
from ophyd_async.epics.core import epics_signal_r

from geecs_bluesky.plans.step_scan import geecs_step_scan

N_EVENTS = 50


class BenchDevice(StandardReadable):
    """N float signals on mock backends — stands in for one GEECS readable."""

    def __init__(self, name: str, n_signals: int) -> None:
        with self.add_children_as_readables():
            for i in range(n_signals):
                setattr(self, f"v{i}", epics_signal_r(float, f"BENCH:{name}:v{i}"))
        super().__init__(name=name)


def connect_mock(RE: RunEngine, *devices) -> None:
    """Connect devices with mock backends on the RE loop."""
    for device in devices:
        asyncio.run_coroutine_threadsafe(device.connect(mock=True), RE._loop).result(
            timeout=30.0
        )


def make_devices(RE: RunEngine, n_dev: int, n_sig: int, tag: str):
    """Build *n* mock BenchDevices with *signals* float signals each."""
    devs = [BenchDevice(f"{tag}_d{i}", n_sig) for i in range(n_dev)]
    connect_mock(RE, *devs)
    return devs


def run_case(RE: RunEngine, devices, *, strict: bool, normalizer: bool):
    """Run one benchmark case; return per-event ms stats."""
    event_times: list[float] = []

    def timer(name, doc):
        if name == "event":
            event_times.append(time.perf_counter())

    tokens = [RE.subscribe(timer)]
    if normalizer:
        norm = RunNormalizer()
        norm.subscribe(lambda n, d: None)
        tokens.append(RE.subscribe(norm))

    fire = (lambda: bps.null()) if strict else None
    plan = geecs_step_scan(
        motor=None,
        positions=[None],
        detectors=list(devices),
        shots_per_step=N_EVENTS,
        fire_shot=fire,
    )
    t0 = time.perf_counter()
    RE(plan)
    total = time.perf_counter() - t0
    for tk in tokens:
        RE.unsubscribe(tk)

    deltas = [b - a for a, b in zip(event_times, event_times[1:])]
    return {
        "total_s": total,
        "n_events": len(event_times),
        "ms_per_event_median": 1e3 * statistics.median(deltas),
        "ms_per_event_mean": 1e3 * statistics.fmean(deltas),
        "ms_per_event_p95": 1e3 * sorted(deltas)[int(0.95 * len(deltas))],
        "ms_open_close": 1e3 * (total - (event_times[-1] - event_times[0])),
    }


def count_msgs_per_event(devices, strict: bool) -> int:
    """Messages in one full NOSCAN run / events, from a dry message listing."""
    fire = (lambda: bps.null()) if strict else None
    msgs = list(
        geecs_step_scan(
            motor=None,
            positions=[None],
            detectors=list(devices),
            shots_per_step=N_EVENTS,
            fire_shot=fire,
        )
    )
    return len(msgs)


def main() -> None:
    """Run the full sweep and print the results table."""
    RE = RunEngine(context_managers=[])

    # ---- message-sequence dump for a tiny case (task 1 cross-check) ----
    tiny = [BenchDevice(f"seq_d{i}", 2) for i in range(2)]
    for strict in (False, True):
        fire = (lambda: bps.null()) if strict else None
        msgs = list(
            geecs_step_scan(
                motor=None,
                positions=[None],
                detectors=tiny,
                shots_per_step=2,
                fire_shot=fire,
            )
        )
        label = "strict(no-op fire)" if strict else "free-fallback(trigger_and_read)"
        print(f"\n== message sequence, 2 devices x 2 shots, {label} ==")
        print(" ".join(m.command for m in msgs))

    cases = [
        # (label, n_dev, n_sig, strict)
        ("A: 2 dev x 4 sig", 2, 4, False),
        ("A: 10 dev x 4 sig", 10, 4, False),
        ("A: 50 dev x 4 sig", 50, 4, False),
        ("A: 137 dev x 4 sig", 137, 4, False),
        ("B: 2 dev x 100 sig", 2, 100, False),
        ("B: 2 dev x 392 sig", 2, 392, False),
        ("B: 87 dev x 9 sig", 87, 9, False),
        ("C: strict 50 dev x 4", 50, 4, True),
        ("C: strict 87 dev x 9", 87, 9, True),
    ]

    # warmup
    w = make_devices(RE, 2, 2, "warm")
    run_case(RE, w, strict=False, normalizer=False)

    print(
        f"\n{'case':24} {'cols':>5} {'msgs/ev':>8} "
        f"{'bare med':>9} {'bare p95':>9} {'+norm med':>10} {'norm cost':>10} {'open+close':>11}"
    )
    for label, n_dev, n_sig, strict in cases:
        tag = label.split(":")[0] + f"{n_dev}x{n_sig}" + ("s" if strict else "")
        devs = make_devices(RE, n_dev, n_sig, tag)
        cols = n_dev * n_sig + 3  # + ScanContext columns
        n_msgs = count_msgs_per_event(devs, strict)
        bare = run_case(RE, devs, strict=strict, normalizer=False)
        norm = run_case(RE, devs, strict=strict, normalizer=True)
        print(
            f"{label:24} {cols:>5} {n_msgs / N_EVENTS:>8.1f} "
            f"{bare['ms_per_event_median']:>7.2f}ms {bare['ms_per_event_p95']:>7.2f}ms "
            f"{norm['ms_per_event_median']:>8.2f}ms "
            f"{norm['ms_per_event_median'] - bare['ms_per_event_median']:>8.2f}ms "
            f"{bare['ms_open_close']:>9.1f}ms"
        )

    RE._loop.call_soon_threadsafe(lambda: None)


if __name__ == "__main__":
    main()
