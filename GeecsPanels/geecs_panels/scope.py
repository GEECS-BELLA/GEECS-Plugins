"""The oscilloscope contract: the PicoScope IOC's PV suffixes, its write allow-list, trace decimation.

Pure: no CA, no I/O. The suffixes are the records of geecs-epics'
``ioc-picoscope`` (``PS5000A.template`` and ``PS5000A_channel.template``)
under the device prefix; a channel's live under ``<prefix><CH>:``.

**Traces are read on request, never monitored** (as the camera's arrays):
a 100 k-sample capture per channel at the scope's rate would flood the
VPN. :func:`decimate` reduces a trace to min/max pairs per pixel column,
so a narrow spike survives the reduction that a stride would drop.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Optional

#: The channels a 5000 Series scope can have; a model's own are ``Present``.
CHANNELS = ("A", "B", "C", "D")

#: Scope-wide readbacks: snapshot key → suffix. Monitored.
SCOPE_FIELDS: dict[str, str] = {
    "model": "Model_RBV",
    "serial": "Serial_RBV",
    "firmware": "Firmware_RBV",
    "driver": "DriverVersion_RBV",
    "num_channels": "NumChannels_RBV",
    "connected": "Connected_RBV",
    "status": "Status_RBV",
    "last_error": "LastError_RBV",
    "resolution": "Resolution_RBV",
    "interval_ns": "SampleInterval_RBV",
    "samples": "NumSamples_RBV",
    "max_samples": "MaxSamples_RBV",
    "pre_trigger": "PreTrigger_RBV",
    "trigger_mode": "TriggerMode_RBV",
    "trigger_source": "TriggerSource_RBV",
    "trigger_level": "TriggerLevel_RBV",
    "trigger_edge": "TriggerEdge_RBV",
    "auto_ms": "AutoTriggerTime_RBV",
    "acquire": "Acquire_RBV",
    "state": "State_RBV",
    "count": "CaptureCount_RBV",
}

#: Per-channel readbacks: key → suffix under ``<CH>:``. Monitored.
CHANNEL_FIELDS: dict[str, str] = {
    "present": "Present_RBV",
    "enable": "Enable_RBV",
    "range": "Range_RBV",
    "coupling": "Coupling_RBV",
    "offset": "Offset_RBV",
    "overrange": "Overrange_RBV",
}

#: Read as strings (enums by their state names; LastError is a char array).
STRING_KEYS = frozenset(
    {
        "model",
        "serial",
        "firmware",
        "driver",
        "connected",
        "status",
        "last_error",
        "resolution",
        "trigger_mode",
        "trigger_source",
        "trigger_edge",
        "acquire",
        "state",
        "present",
        "enable",
        "range",
        "coupling",
        "overrange",
    }
)

TIME_AXIS = "TimeAxis_RBV"
WAVEFORM = "Waveform_RBV"

RESOLUTIONS = ("8 bit", "12 bit", "14 bit", "15 bit", "16 bit")
RANGES = (
    "10 mV",
    "20 mV",
    "50 mV",
    "100 mV",
    "200 mV",
    "500 mV",
    "1 V",
    "2 V",
    "5 V",
    "10 V",
    "20 V",
)
COUPLINGS = ("AC", "DC")
TRIGGER_MODES = ("Free run", "Auto", "Normal", "Single")
TRIGGER_SOURCES = ("A", "B", "C", "D", "External")
EDGES = ("Rising", "Falling")


class NotWritable(ValueError):
    """A write the allow-list refuses; the message says why."""


def _real(lo: float, hi: float):
    def check(suffix: str, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise NotWritable(f"{suffix} takes a number, not {value!r}")
        if not (math.isfinite(value) and lo <= value <= hi):
            raise NotWritable(f"{suffix} takes {lo:g} to {hi:g}")
        return float(value)

    return check


def _whole(lo: int, hi: int):
    def check(suffix: str, value: object) -> int:
        whole = isinstance(value, int) or (
            isinstance(value, float) and value.is_integer()
        )
        if isinstance(value, bool) or not whole or not lo <= value <= hi:  # type: ignore[operator]
            raise NotWritable(f"{suffix} takes a whole number {lo} to {hi}")
        return int(value)  # type: ignore[arg-type]

    return check


def _choice(names: Sequence[str]):
    def check(suffix: str, value: object) -> str:
        if value not in names:
            raise NotWritable(f"{suffix} takes one of {', '.join(names)}")
        return str(value)

    return check


#: The panel's writes: every setting of the scope (it is the panel's whole
#: point), nothing a capture produces. suffix → validator returning the
#: value to put (an enum by its state name).
WRITABLE = {
    "Resolution": _choice(RESOLUTIONS),
    "SampleInterval": _real(0.0, 1e9),
    "NumSamples": _whole(1, 100_000_000),
    "PreTrigger": _real(0.0, 100.0),
    "TriggerMode": _choice(TRIGGER_MODES),
    "TriggerSource": _choice(TRIGGER_SOURCES),
    "TriggerLevel": _real(-20.0, 20.0),
    "TriggerEdge": _choice(EDGES),
    "AutoTriggerTime": _whole(1, 32767),
    "Acquire": _choice(("Stop", "Start")),
    **{
        f"{ch}:{name}": check
        for ch in CHANNELS
        for name, check in (
            ("Enable", _choice(("Off", "On"))),
            ("Range", _choice(RANGES)),
            ("Coupling", _choice(COUPLINGS)),
            ("Offset", _real(-20.0, 20.0)),
        )
    },
}


def check_write(suffix: str, value: object) -> object:
    """The value to put for an allowed write, or :class:`NotWritable`."""
    check = WRITABLE.get(suffix)
    if check is None:
        raise NotWritable(f"{suffix} is not writable from the panel")
    return check(suffix, value)


def monitored() -> dict[str, tuple[str, bool]]:
    """Every monitored suffix → (snapshot key, read as string)."""
    out = {suffix: (key, key in STRING_KEYS) for key, suffix in SCOPE_FIELDS.items()}
    for ch in CHANNELS:
        for key, suffix in CHANNEL_FIELDS.items():
            out[f"{ch}:{suffix}"] = (f"{ch}.{key}", key in STRING_KEYS)
    return out


def decimate(
    times: Sequence[float], values: Sequence[float], columns: int
) -> tuple[list[float], list[Optional[float]]]:
    """At most ``2 * columns`` points: each column's minimum and maximum, in time order.

    A trace already that short is returned whole. Non-finite samples are
    ``None`` (the page's ``JSON.parse`` rejects NaN).
    """
    n = min(len(times), len(values))
    if columns < 1 or n <= 2 * columns:
        return list(times[:n]), [v if math.isfinite(v) else None for v in values[:n]]
    xs: list[float] = []
    ys: list[Optional[float]] = []
    for c in range(columns):
        lo, hi = c * n // columns, (c + 1) * n // columns
        chunk = [(values[i], i) for i in range(lo, hi) if math.isfinite(values[i])]
        if not chunk:
            xs.append(times[lo])
            ys.append(None)
            continue
        (vmin, imin), (vmax, imax) = min(chunk), max(chunk)
        for i, v in sorted(((imin, vmin), (imax, vmax))):
            xs.append(times[i])
            ys.append(v)
    return xs, ys
