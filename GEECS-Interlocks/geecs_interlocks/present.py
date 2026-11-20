"""How a snapshot reads to people: the verdict's words, a value's text, the kit's state words.

The server window and the web client both draw from here, so they say the
same thing about the same snapshot. Nothing here imports Qt or the web
stack.
"""

from __future__ import annotations

import time

from geecs_interlocks.monitor import State, Snapshot

#: Each interlock state as the GEECS web kit's status word (its chip colour);
#: the chip still shows the interlock's own word.
KIT_STATE: dict[State, str] = {
    "tripped": "failed",
    "waiting": "degraded",
    "settling": "degraded",
    "ok": "ok",
    "off": "unknown",
}


def verdict_text(snapshot: Snapshot) -> tuple[str, str]:
    """The verdict's headline and its detail line."""
    armed = [s for s in snapshot.statuses if s.armed]
    if not snapshot.statuses:
        return "No interlocks", "no interlocks file — reported unsafe"
    if not armed:
        return "Nothing armed", "every interlock is off — reported unsafe"
    if snapshot.safe:
        return "Safe", f"all {len(armed)} armed interlocks OK"
    bad = [s for s in armed if s.unsafe]
    since = min(s.since for s in bad)
    tripped = sum(s.state == "tripped" for s in bad)
    headline = "Tripped" if tripped else "Not ready"
    when = time.strftime("%H:%M:%S", time.localtime(since))
    return headline, f"{len(bad)} of {len(armed)} armed not OK · since {when}"


def is_age(condition_line: str) -> bool:
    """Whether a check's value is an age in seconds (stale and external checks)."""
    return (
        " connected" in condition_line
        or "updating every" in condition_line
        or " on UDP " in condition_line
    )


def number_text(value: float | None, seconds: bool = False) -> str:
    """A value for a table: four significant digits (an age in seconds), a dash for none."""
    if value is None:
        return "—"
    if seconds:
        return f"{value:.1f} s" if value < 10 else f"{value:.0f} s"
    return f"{value:.4g}"


__all__ = ["KIT_STATE", "is_age", "number_text", "verdict_text"]
