"""Phase-0 probe G1: test whether the LabVIEW image feed is every-frame or best-effort.

Subscribes directly to one GEECS device's TCP push stream (the same feed the
PVA gateway consumes) with ``include_shot=True`` and records every update's
wire shot counter. Shot-counter gaps during a strict scan are the ground
truth for whether central PVA capture can ever be lossless: if the device
itself skips counters, no downstream work recovers those frames.

Read-only: opens a TCP subscription, never writes to the device.

Run from the GeecsPvaGateway poetry env (it has geecs-core + geecs-ca-gateway):

    cd GeecsPvaGateway
    poetry run python ../GeecsBluesky/capture/probes/probe_g1_shot_counter.py \
        --experiment Undulator --device UC_TopView --duration 300

Output: one JSONL record per update plus a printed summary (update count,
distinct shots, gap list, inter-arrival stats).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from geecs_core.db.geecs_db import GeecsDb
from geecs_core.transport.tcp_subscriber import GeecsTcpSubscriber


def resolve_image_variables(experiment: str, device: str) -> list[str]:
    """Image-typed variables for *device*, per the PVA gateway's own predicate."""
    from geecs_ca_gateway.config import effective_vartype

    metas = GeecsDb.get_experiment_device_variables(experiment).get(device, [])
    image_vars = sorted(
        m["name"]
        for m in metas
        if effective_vartype(m.get("variabletype"), m.get("choices")) == "image"
    )
    return image_vars


class ProbeRecorder:
    """Accumulate per-update records and compute the gap/interval summary."""

    def __init__(self, image_var: str, out_path: Path) -> None:
        self.image_var = image_var
        self.out = out_path.open("w")
        self.updates = 0
        self.image_updates = 0
        self.empty_payload_updates = 0
        self.shots: list[int] = []
        self.recv_times: list[float] = []
        self.disconnected_at: float | None = None
        self._t0 = time.time()

    def on_update(self, update: dict[str, Any]) -> None:
        """Record one TCP push update (JSONL row + console line)."""
        now = time.time()
        shot = update.get("shot number")
        blob = update.get(self.image_var)
        rec = {
            "recv": round(now, 6),
            "shot": shot,
            "acq_timestamp": update.get("acq_timestamp"),
            "systimestamp": update.get("systimestamp"),
            "has_image": bool(blob),
            "image_bytes": len(blob) if isinstance(blob, str) else None,
            "keys": sorted(k for k in update if k != self.image_var),
        }
        self.out.write(json.dumps(rec) + "\n")
        self.out.flush()
        self.updates += 1
        if blob:
            self.image_updates += 1
        elif blob is not None:
            # Present-but-empty payload: measured behavior for remote
            # subscribers (images ship only to host-local ones) — count
            # separately so image_updates never inflates on a remote host.
            self.empty_payload_updates += 1
        if isinstance(shot, int):
            self.shots.append(shot)
        self.recv_times.append(now)
        print(
            f"[{now - self._t0:8.2f}s] shot={shot} "
            f"img={'%d B' % len(blob) if isinstance(blob, str) else '-'} "
            f"acq={update.get('acq_timestamp')}",
            flush=True,
        )

    def summary(self) -> dict[str, Any]:
        """Return counts, shot-counter gaps, and inter-arrival stats."""
        gaps: list[tuple[int, int]] = []
        dupes = 0
        regressions = 0
        for prev, cur in zip(self.shots, self.shots[1:]):
            if cur == prev:
                dupes += 1
            elif cur < prev:
                regressions += 1  # counter reset (device restart) — not a drop
            elif cur > prev + 1:
                gaps.append((prev, cur))
        intervals = [b - a for a, b in zip(self.recv_times, self.recv_times[1:])]
        return {
            "updates": self.updates,
            "image_updates": self.image_updates,
            "empty_payload_updates": self.empty_payload_updates,
            "shot_updates": len(self.shots),
            "distinct_shots": len(set(self.shots)),
            "shot_min": min(self.shots) if self.shots else None,
            "shot_max": max(self.shots) if self.shots else None,
            "duplicate_shot_updates": dupes,
            "counter_regressions": regressions,
            "gaps": gaps,
            "missing_via_gaps": sum(b - a - 1 for a, b in gaps),
            "interval_mean_s": round(statistics.mean(intervals), 4)
            if intervals
            else None,
            "interval_max_s": round(max(intervals), 4) if intervals else None,
            "interval_note": "intervals span the whole session (connect ramp "
            "and idle 1 Hz pushes included) — window offline for scan cadence",
            "stream_disconnected_early_at": self.disconnected_at,
        }

    def close(self) -> None:
        """Close the JSONL output file."""
        self.out.close()


async def main() -> int:
    """Run the probe from CLI args."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--image-var", default=None, help="override image variable name")
    ap.add_argument("--duration", type=float, default=300.0, help="seconds to listen")
    ap.add_argument("--out", default=None, help="JSONL output path")
    args = ap.parse_args()

    host, port = GeecsDb.find_device(args.device)
    image_var = args.image_var
    if image_var is None:
        image_vars = resolve_image_variables(args.experiment, args.device)
        if not image_vars:
            print(
                f"ERROR: no image-typed variables found for {args.device}",
                file=sys.stderr,
            )
            return 2
        image_var = image_vars[0]
        if len(image_vars) > 1:
            print(f"NOTE: multiple image vars {image_vars}; probing {image_var!r}")

    out_path = Path(
        args.out or f"probe_g1_{args.device}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    )
    print(f"G1 probe: {args.device} @ {host}:{port}, var={image_var!r} -> {out_path}")

    sub = GeecsTcpSubscriber(host, port)
    await sub.connect()
    rec = ProbeRecorder(image_var, out_path)
    try:
        await sub.subscribe(
            [image_var, "acq_timestamp", "systimestamp"],
            rec.on_update,
            text_variables={image_var},
            include_shot=True,
        )
        # A silently-dead socket must never masquerade as "the device
        # stopped pushing": race the duration against disconnect and
        # record which one ended the session.
        sleep_task = asyncio.create_task(asyncio.sleep(args.duration))
        drop_task = asyncio.create_task(sub.wait_disconnected())
        done, pending = await asyncio.wait(
            {sleep_task, drop_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for t in pending:
            t.cancel()
        if drop_task in done:
            rec.disconnected_at = round(time.time() - rec._t0, 2)
            print("!!! TCP stream DISCONNECTED before the duration elapsed")
    finally:
        await sub.close()
        summary = rec.summary()
        rec.close()
        print("\n=== G1 SUMMARY ===")
        print(json.dumps(summary, indent=2))
        # Interpretation guidance lives in README.md (key on distinct
        # acq_timestamps; the wire shot counter is Master-Control-owned).
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        raise SystemExit(130) from None
