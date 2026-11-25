"""How a snapshot reads to people: the verdict's words, a value's text, the kit's state words.

The stream and the web client both draw from here, so they say the
same thing about the same snapshot. Nothing here imports Qt or the web
stack.
"""

from __future__ import annotations

import time

from geecs_interlocks.monitor import TICK_S, InterlockStatus, State, Snapshot

#: Seconds a snapshot may be old before every reader treats the server as
#: stalled: ten ticks without an evaluation is not a slow tick.
STALL_S = 10 * TICK_S


def stall_age(snapshot: Snapshot, now: float | None = None) -> float | None:
    """How long *snapshot* has not been replaced, when that is past :data:`STALL_S` (else ``None``).

    *now* is a ``time.monotonic()`` reading (default: now), compared with
    the snapshot's monotonic stamp.

    The evaluation loop publishes a snapshot every tick; one that stops being
    replaced means the loop is blocked or gone, and the last verdict must not
    stand for the current one.
    """
    age = (time.monotonic() if now is None else now) - snapshot.mono
    return age if age > STALL_S else None


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


#: The condition kinds whose value is an age in seconds.
AGE_KINDS = frozenset({"stale", "external"})


def is_age(kind: str) -> bool:
    """Whether a check of *kind* measures an age in seconds."""
    return kind in AGE_KINDS


def status_is_age(status: InterlockStatus) -> bool:
    """Whether an interlock's own value is an age (it holds one check, and that one measures one)."""
    return len(status.checks) == 1 and is_age(status.checks[0].kind)


def number_text(value: float | None, seconds: bool = False) -> str:
    """A value for a table: four significant digits (an age in seconds), a dash for none."""
    if value is None:
        return "—"
    if seconds:
        return f"{value:.1f} s" if value < 10 else f"{value:.0f} s"
    return f"{value:.4g}"


__all__ = [
    "AGE_KINDS",
    "KIT_STATE",
    "STALL_S",
    "is_age",
    "number_text",
    "stall_age",
    "status_is_age",
    "verdict_text",
]
