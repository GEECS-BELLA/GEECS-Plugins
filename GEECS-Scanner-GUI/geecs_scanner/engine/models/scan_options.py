"""Validated scan-execution options passed from GUI to ScanManager."""

from __future__ import annotations

from pydantic import BaseModel, field_validator


class ScanOptions(BaseModel):
    """Engine-level options that affect how a scan is executed.

    Constructed by the GUI from its menu-bar settings and passed through
    ``RunControl`` to ``ScanManager``.  User-interface-only preferences
    (sound theme, dark mode, window layout) are not included here.

    Attributes
    ----------
    rep_rate_hz : float
        Shot repetition rate in Hz.  Must be positive.
    enable_global_time_sync : bool
        When True, use cross-device timestamp comparison to synchronise
        devices before each scan step instead of the fixed timeout method.
    global_time_tolerance_ms : int
        Tolerance window (milliseconds) for global time synchronisation.
        Clamped to [0, 60000].
    master_control_ip : str
        IP address of the Master Control server for ECS live dump.
        Empty string disables the dump.
    on_shot_tdms : bool
        When True, write an incremental TDMS update after each shot.
    save_direct_on_network : bool
        When True, save device data directly to the scan folder on the
        network share instead of staging locally first.
    randomized_beeps : bool
        When True, vary the beep pitch between shots.
    """

    rep_rate_hz: float = 1.0
    enable_global_time_sync: bool = False
    global_time_tolerance_ms: int = 0
    master_control_ip: str = ""
    on_shot_tdms: bool = False
    save_direct_on_network: bool = False
    randomized_beeps: bool = False

    @field_validator("rep_rate_hz")
    @classmethod
    def _positive_rep_rate(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"rep_rate_hz must be positive, got {v}")
        return v

    @field_validator("global_time_tolerance_ms")
    @classmethod
    def _clamp_tolerance(cls, v: int) -> int:
        return max(0, min(v, 60_000))
