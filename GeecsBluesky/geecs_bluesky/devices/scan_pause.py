"""Which GEECS devices are paused in scan mode right now.

A run device that acknowledged ``ScanStart`` pauses its acquisition loop
between announced shots (CLAUDE.md, "Scan mode"), so its readbacks freeze
at the last shot.  A move that confirms by readback — :class:`CaMotor` —
cannot confirm on such a device; it takes the device's own executed reply
as the verdict instead (the device has already run its check-values against
the DB tolerance before answering ``no error``).

:class:`~geecs_bluesky.devices.scan_mode.ScanModeControl` is the one writer:
it marks the devices that acknowledged ``ScanStart`` and clears them on
every exit.  One worker drives one run at a time, so a process-wide set is
the run's state.  Names compare case-insensitively, as GEECS names do.
"""

from __future__ import annotations

from collections.abc import Iterable

_paused: frozenset[str] = frozenset()


def mark_paused(devices: Iterable[str]) -> None:
    """Replace the paused set with *devices* (empty clears it)."""
    global _paused
    _paused = frozenset(d.casefold() for d in devices)


def is_paused(device: str) -> bool:
    """Whether *device* is in scan mode with its acquisition paused."""
    return device.casefold() in _paused


__all__ = ["is_paused", "mark_paused"]
