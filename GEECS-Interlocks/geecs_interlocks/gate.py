"""The shot gate's wire format: the snapshot as the JSON frame GeecsBluesky's ``ShotGate`` reads.

A trigger profile's ``shot_gate: {host, port}`` names this server's gate
port; the scan worker then holds every strict single-shot fire until a
frame says safe (``geecs_bluesky.shot_gate``, protocol v1). The frame is
the same 4-byte big-endian length prefix as the Master Control stream's,
then UTF-8 JSON::

    {"v": 1, "ok": true, "reason": "",
     "interlocks": {"<name>": {"unsafe": false, "diagnostic": "",
                               "data_stamp": 1795562899.04}}}

- ``ok`` is the snapshot's verdict: at least one interlock armed and every
  armed one OK. ``reason`` says why not.
- ``interlocks`` lists the armed interlocks only (an ``off`` one is not in
  the verdict). None armed, or no interlocks file, sends an empty object,
  which the client reads as not safe.
- ``data_stamp`` (Unix seconds, this server's clock) is when the oldest
  device reading the interlock's verdict rests on arrived here. After a
  fire the client counts a verdict only once every stamp is later than the
  fire (+ its ``holdoff_s``), so a shot never fires on readings taken
  before the last one. External (UDP) signals carry no stamp: a sender
  reports on a change, so its last packet may rightly predate the fire.
  An interlock reading no device (external signals only) sends ``null``,
  and the client then requires the frame itself to arrive after the fire.
  A stamp is an arrival time, not the device's acquisition time: a reading
  delayed in transit past the fire counts as fresh, which the profile's
  ``holdoff_s`` covers. The worker compares stamps with its own clock, so
  the two hosts' clocks must agree (the same host on a services box).
- A stalled evaluation sends ``ok: false`` with one unsafe entry naming
  the server, never the last verdict again.

Nothing here imports the web stack.
"""

from __future__ import annotations

import json

from geecs_interlocks.monitor import Snapshot
from geecs_interlocks.wire import STALLED_NAME

#: The protocol version this server sends (``geecs_bluesky.shot_gate.PROTOCOL_VERSION``).
GATE_PROTOCOL_VERSION = 1


def _reason(snapshot: Snapshot) -> str:
    armed = [s for s in snapshot.statuses if s.armed]
    if not snapshot.statuses:
        return (
            f"no interlocks ({snapshot.source})" if snapshot.source else "no interlocks"
        )
    if not armed:
        return "no interlock armed"
    return "; ".join(f"{s.name}: {s.diagnostic or s.state}" for s in armed if s.unsafe)


def render_gate(snapshot: Snapshot) -> str:
    """The JSON body one gate frame carries for *snapshot*."""
    safe = snapshot.safe
    body = {
        "v": GATE_PROTOCOL_VERSION,
        "ok": safe,
        "reason": "" if safe else _reason(snapshot),
        "interlocks": {
            s.name: {
                "unsafe": s.unsafe,
                "diagnostic": s.diagnostic,
                "data_stamp": s.data_stamp,
            }
            for s in snapshot.statuses
            if s.armed
        },
    }
    return json.dumps(body, allow_nan=False)


def render_gate_stalled(age_s: float) -> str:
    """The gate frame sent while no snapshot has replaced the last for *age_s*: not safe."""
    why = f"evaluation stopped {age_s:.0f} s ago"
    return json.dumps(
        {
            "v": GATE_PROTOCOL_VERSION,
            "ok": False,
            "reason": why,
            "interlocks": {
                STALLED_NAME: {"unsafe": True, "diagnostic": why, "data_stamp": None}
            },
        }
    )


__all__ = ["GATE_PROTOCOL_VERSION", "render_gate", "render_gate_stalled"]
