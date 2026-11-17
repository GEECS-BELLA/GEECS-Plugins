"""The shot gate's state for the "shot gate" chip — shown, never a gate here.

A trigger profile may name a permission-to-fire service
(``TriggerProfile.shot_gate``).  The **worker** enforces it: every
SINGLESHOT under that profile waits for one fresh ``safe``
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
from typing import Any, Optional

from geecs_bluesky.shot_gate import ShotGate

from geecs_scanner.service.models import ShotGateOut


def disabled(detail: str) -> ShotGateOut:
    """No gate for this profile: the chip stays hidden."""
    return ShotGateOut(enabled=False, state="unknown", detail=detail)


def gate_status(gate: ShotGate, profile: str = "") -> ShotGateOut:
    """One gate's latest verdict as kit words.

    ``ok`` safe; ``degraded`` connected and not safe (normal between shots:
    the hardware's cleanup), with the service's reason; ``failed`` not
    connected (the worker would refuse a run, or hold a running one).
    """
    common = {"enabled": True, "address": gate.address, "profile": profile}
    if not gate.connected:
        return ShotGateOut(state="failed", detail=gate.latest.reason, **common)
    latest = gate.latest
    if latest.safe:
        return ShotGateOut(state="ok", detail="safe", safe=True, **common)
    return ShotGateOut(state="degraded", detail=latest.reason, **common)


class _Watch:
    """One read-only reader on its own loop, in a daemon thread never stopped."""

    def __init__(self, gate: ShotGate) -> None:
        self.gate = gate
        self._loop = asyncio.new_event_loop()
        threading.Thread(
            target=self._run, name=f"shot-gate-watch {gate.address}", daemon=True
        ).start()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self.gate.start())
        self._loop.run_forever()


class ShotGateWatches:
    """One watch per gate address, started on first use; :meth:`status` reads it.

    Readers reconnect by themselves and are never stopped (the process's
    exit ends them, like the stream consumers); :meth:`status` reads plain
    attributes the loop threads set, under a lock only to start a watch.
    """

    def __init__(self) -> None:
        self._watches: dict[tuple[str, int], _Watch] = {}
        self._lock = threading.Lock()

    def status(self, spec: Any, profile: str = "") -> ShotGateOut:
        """The verdict of the gate *spec* (a ``ShotGateSpec``) names."""
        key = (str(spec.host), int(spec.port))
        with self._lock:
            watch = self._watches.get(key)
            if watch is None:
                watch = self._watches[key] = _Watch(ShotGate(*key))
        return gate_status(watch.gate, profile)


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


__all__ = ["ShotGateWatches", "disabled", "gate_status", "profile_gate"]
