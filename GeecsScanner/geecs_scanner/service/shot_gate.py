"""The shot gate's state for the "shot gate" chip — shown, never a gate here.

A trigger profile may name a permission-to-fire service
(``TriggerProfile.shot_gate``).  The **worker** enforces it: every
SINGLESHOT under that profile waits for one fresh safe verdict
(``geecs_bluesky.plans.shot_gate``).  This module only shows it, for the
profile the form will submit under: :class:`ShotGateWatches` keeps one
read-only :class:`~geecs_bluesky.shot_gate.ShotGate` reader per address,
each on its own daemon thread (the service streams one way to every
connected client), started the first time a profile names it, and
:func:`gate_status` projects the latest verdict onto
:class:`~geecs_scanner.service.models.ShotGateOut`.  It never calls
``fired``: the per-fire freshness rule is the worker's, so the chip says
what the service says now.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Optional

from geecs_bluesky.shot_gate import HEARTBEAT_S, ShotGate

from geecs_scanner.service.models import ShotGateOut


def disabled(detail: str) -> ShotGateOut:
    """No gate for this profile: the chip stays hidden."""
    return ShotGateOut(enabled=False, state="unknown", detail=detail)


def unreadable(profile: str, exc: Exception) -> ShotGateOut:
    """A profile that could not be read: shown as unknown, since it may name a gate."""
    return ShotGateOut(
        enabled=True,
        state="unknown",
        detail=f"trigger profile not readable: {exc} — the worker will not load it",
        profile=profile,
    )


def gate_status(
    gate: ShotGate, profile: str = "", connecting: bool = False
) -> ShotGateOut:
    """One gate's latest verdict as kit words.

    ``ok`` the server says safe; ``degraded`` connected and not safe, with
    the server's reason; ``failed`` not connected or silent (the worker
    would refuse a run, or hold a running one); ``unknown`` while a new
    reader waits for its first frame (*connecting*).  This is the server's
    verdict now: the worker's per-fire freshness hold is never shown here,
    so the chip can read safe while a shot waits for a fresh verdict.
    """
    common = {"enabled": True, "address": gate.address, "profile": profile}
    if not gate.connected and connecting:
        return ShotGateOut(state="unknown", detail="connecting", **common)
    if not gate.connected:
        return ShotGateOut(state="failed", detail=gate.latest.reason, **common)
    latest = gate.latest
    if latest.safe:
        return ShotGateOut(state="ok", detail="safe", safe=True, **common)
    return ShotGateOut(state="degraded", detail=latest.reason, **common)


#: Seconds a watch lives without a status read: the page polls every 2 s (a
#: background tab about once a minute), so an address no open page asks
#: about (an edited port, a retired server) is dropped and its connection
#: closed.
WATCH_IDLE_S = 300.0
#: Seconds a new watch reads ``unknown`` ("connecting") rather than
#: ``failed`` while it waits for its first frame.
CONNECTING_S = 2 * HEARTBEAT_S


class _Watch:
    """One read-only reader on its own loop, in a daemon thread, until closed."""

    def __init__(self, gate: ShotGate) -> None:
        self.gate = gate
        self.started = self.used = time.monotonic()
        self._loop = asyncio.new_event_loop()
        threading.Thread(
            target=self._run, name=f"shot-gate-watch {gate.address}", daemon=True
        ).start()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self.gate.start())
        self._loop.run_forever()
        self._loop.run_until_complete(self.gate.stop())
        self._loop.close()

    def close(self) -> None:
        """Stop the loop; its thread stops the reader and closes the connection."""
        self._loop.call_soon_threadsafe(self._loop.stop)


class ShotGateWatches:
    """One watch per gate address, started on first use; :meth:`status` reads it.

    Readers reconnect by themselves; a watch no status read has asked
    about for :data:`WATCH_IDLE_S` is closed at the next read.
    :meth:`status` reads plain attributes the loop threads set, under a
    lock only to start or close a watch.
    """

    def __init__(self, idle_s: float = WATCH_IDLE_S) -> None:
        self._watches: dict[tuple[str, int], _Watch] = {}
        self._lock = threading.Lock()
        self._idle_s = idle_s

    def status(self, spec: Any, profile: str = "") -> ShotGateOut:
        """The verdict of the gate *spec* (a ``ShotGateSpec``) names."""
        now = time.monotonic()
        with self._lock:
            for key in [
                k for k, w in self._watches.items() if now - w.used > self._idle_s
            ]:
                self._watches.pop(key).close()
            watch = self._watches.get(spec.key)
            if watch is None:
                watch = self._watches[spec.key] = _Watch(ShotGate(*spec.key))
            watch.used = now
        return gate_status(watch.gate, profile, now - watch.started < CONNECTING_S)


def profile_gate(resolver: Any, name: Optional[str]) -> tuple[str, Any]:
    """The profile the form submits under (*name*, else the default) and its gate spec.

    Returns ``("", None)`` when no profile can be told (no name and no
    default); raises what the resolver raises for an unreadable profile.
    """
    if not name:
        defaults = resolver.resolve_experiment_defaults()
        name = getattr(defaults, "trigger_profile", None) or ""
    if not name:
        return "", None
    profile = resolver.resolve_trigger_profile(name)
    return name, getattr(profile, "shot_gate", None)


__all__ = ["ShotGateWatches", "disabled", "gate_status", "profile_gate", "unreadable"]
