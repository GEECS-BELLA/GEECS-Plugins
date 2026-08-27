"""Phase-0 probe G2: test whether a deep client-side monitor queue defeats p4p squashing.

Opens one (or two) PVA monitors on a camera image PV served by the
GeecsPvaGateway fleet:

- a DEEP monitor with ``record[queueSize=N]`` (the would-be capture client)
- optionally a SHALLOW monitor with the default pvRequest (the viewer twin)

and counts delivered updates. Compared against probe G1's wire counts over
the same window, this separates the two drop stages: gateway latest-wins
(both monitors undercount equally) vs client-side MonitorFIFO squash (deep
outcounts shallow).

Read-only. Subscribing DOES open the gateway's gate (activates the upstream
LabVIEW stream) — same effect as opening the camera in Phoebus.

Run from the GeecsPvaGateway poetry env (has p4p + geecs-core):

    cd GeecsPvaGateway
    poetry run python ../GeecsBluesky/capture/probes/probe_g2_pva_deep_queue.py \
        --experiment Undulator --device UC_TopView --duration 300 --shallow

Note: EPICS_PVA_ADDR_LIST is set in-process from the device's DB endpoint IP
before p4p is imported — no shell env setup needed.

Reading the output (details in README.md): ``disconnect_events: 1`` on a
healthy run is the initial not-yet-connected notification, not instability;
a dead or misnamed PV shows ``disconnect_events: 1, updates: 0``.
``--image-var`` defaults to ``"image"`` and is NOT validated against the DB —
a differently-named image variable produces the dead-PV signature. Compare
``distinct_pv_timestamps``, never raw ``updates``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

from geecs_core.db.geecs_db import GeecsDb
from geecs_core.pv_naming import pv_name


class MonitorRecorder:
    """Count and log delivered updates for one PVA monitor subscription."""

    def __init__(self, label: str, out_path: Path) -> None:
        self.label = label
        self.out = out_path.open("w")
        self.lock = threading.Lock()
        self.updates = 0
        self.disconnects = 0
        self.distinct_ts: set[float] = set()
        self._t0 = time.time()

    def __call__(self, value: Any) -> None:  # p4p ntndarray or Disconnected
        """Handle one monitor delivery (update or Disconnected event)."""
        now = time.time()
        with self.lock:
            if isinstance(value, Exception):
                self.disconnects += 1
                self.out.write(json.dumps({"recv": now, "event": repr(value)}) + "\n")
                self.out.flush()
                print(f"[{self.label}] event: {value!r}", flush=True)
                return
            self.updates += 1
            ts = getattr(value, "timestamp", None)
            if ts is not None:
                self.distinct_ts.add(float(ts))
            rec = {
                "recv": round(now, 6),
                "n": self.updates,
                "shape": getattr(value, "shape", None),
                "pv_timestamp": float(ts) if ts is not None else None,
            }
            self.out.write(json.dumps(rec) + "\n")
            self.out.flush()
            print(
                f"[{self.label}][{now - self._t0:8.2f}s] #{self.updates} "
                f"shape={rec['shape']} ts={rec['pv_timestamp']}",
                flush=True,
            )

    def close(self) -> None:
        """Close the JSONL output file."""
        self.out.close()


def main() -> int:
    """Run the probe from CLI args."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--image-var", default="image")
    ap.add_argument("--pv", default=None, help="override full PV name")
    ap.add_argument("--server-ip", default=None, help="override camera-server IP")
    ap.add_argument("--queue-size", type=int, default=100)
    ap.add_argument("--duration", type=float, default=300.0)
    ap.add_argument(
        "--shallow",
        action="store_true",
        help="also run a default-request monitor for comparison",
    )
    args = ap.parse_args()

    server_ip = args.server_ip
    if server_ip is None:
        server_ip, _port = GeecsDb.find_device(args.device)
    # Must be set before p4p import — the fleet spans subnets, broadcast
    # search won't find it (GeecsPvaGateway/DEPLOYMENT.md).
    os.environ["EPICS_PVA_ADDR_LIST"] = server_ip
    os.environ["EPICS_PVA_AUTO_ADDR_LIST"] = "NO"
    from p4p.client.thread import Context

    pv = args.pv or pv_name(args.experiment, args.device, args.image_var)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    print(f"G2 probe: pv={pv!r} server={server_ip} queueSize={args.queue_size}")

    ctx = Context("pva")
    subs = []
    recs: list[MonitorRecorder] = []
    try:
        deep = MonitorRecorder(
            "deep", Path(f"probe_g2_deep_{args.device}_{stamp}.jsonl")
        )
        recs.append(deep)
        subs.append(
            ctx.monitor(
                pv,
                deep,
                request=f"record[queueSize={args.queue_size}]field()",
                notify_disconnect=True,
            )
        )
        if args.shallow:
            shallow = MonitorRecorder(
                "shallow", Path(f"probe_g2_shallow_{args.device}_{stamp}.jsonl")
            )
            recs.append(shallow)
            subs.append(ctx.monitor(pv, shallow, notify_disconnect=True))

        try:
            time.sleep(args.duration)
        except KeyboardInterrupt:
            pass
    finally:
        for s in subs:
            s.close()
        ctx.close()
        print("\n=== G2 SUMMARY ===")
        print(
            "note: raw updates include the initial cached/stale frame and "
            "idle re-pushes (unchanged timestamp) — compare "
            "distinct_pv_timestamps windowed to the scan; see README.md"
        )
        for r in recs:
            print(
                json.dumps(
                    {
                        "monitor": r.label,
                        "updates": r.updates,
                        "distinct_pv_timestamps": len(r.distinct_ts),
                        "disconnect_events": r.disconnects,
                    }
                )
            )
            r.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
