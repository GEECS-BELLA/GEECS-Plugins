"""The scope contract: PV suffixes, the write allow-list, trace decimation.

Pure: no CA, no I/O. The suffixes are geecs-epics' **scope contract**:
vendor-neutral settings names at the device
prefix, aliased by each scope driver onto its own records, and the standard
chain behind them (``ARR`` for the whole capture, ``STAT_<CH>`` per
channel). Nothing here names a vendor: enum choices (ranges, sources,
resolutions) are read from the PVs themselves, so one panel fits every
scope.

**Traces are read on request, never monitored** (as the camera's arrays):
a 100 k-sample capture at the scope's rate would flood the VPN. The
capture arrives as one ``ARR:ArrayData`` array, channels varying fastest;
:func:`split_channels` takes it apart and :func:`decimate` reduces each
trace to min/max pairs per screen column, so a one-sample spike survives
the reduction that a stride would drop.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Callable, Optional

#: Channel letters the contract allows; a model's own are ``Present``.
CHANNELS = ("A", "B", "C", "D")

#: Scope-wide readbacks: snapshot key → suffix. Monitored.
SCOPE_FIELDS: dict[str, str] = {
    "manufacturer": "Manufacturer_RBV",
    "model": "Model_RBV",
    "serial": "SerialNumber_RBV",
    "firmware": "FirmwareVersion_RBV",
    "num_channels": "NumChannels_RBV",
    "connected": "Connected_RBV",
    "state": "DetectorState_RBV",
    "status": "StatusMessage_RBV",
    "last_error": "LastError_RBV",
    "acquire": "Acquire_RBV",
    "image_mode": "ImageMode_RBV",
    "num_images": "NumImages_RBV",
    "count": "ArrayCounter_RBV",
    "resolution": "Resolution_RBV",
    "interval_ns": "SampleInterval_RBV",
    "samples": "NumSamples_RBV",
    "pre_trigger": "PreTrigger_RBV",
    "trigger_mode": "TriggerMode_RBV",
    "trigger_source": "TriggerSource_RBV",
    "trigger_level": "TriggerLevel_RBV",
    "trigger_edge": "TriggerEdge_RBV",
    "auto_ms": "AutoTriggerTime_RBV",
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

#: Per-channel statistics of the chain: key → suffix under ``STAT_<CH>:``.
STAT_FIELDS: dict[str, str] = {
    "total": "Total_RBV",
    "mean": "MeanValue_RBV",
    "min": "MinValue_RBV",
    "max": "MaxValue_RBV",
    "sigma": "Sigma_RBV",
}

#: Read as strings: enums by their state names, and the identity and messages.
STRING_KEYS = frozenset(
    {
        "manufacturer", "model", "serial", "firmware", "connected", "state",
        "status", "last_error", "acquire", "image_mode", "resolution",
        "trigger_mode", "trigger_source", "trigger_edge",
        "present", "enable", "range", "coupling", "overrange",
    }
)  # fmt: skip

#: Char-array records holding text (longer than a string record's 40).
CHAR_STRING_SUFFIXES = ("LastError_RBV", "StatusMessage_RBV")

TIME_AXIS = "TimeAxis_RBV"
ARRAY_DATA = "ARR:ArrayData"
#: The whole capture's shape, monitored: [channels, samples].
ARRAY_SHAPE = {
    "arr_channels": "ARR:ArraySize0_RBV",
    "arr_samples": "ARR:ArraySize1_RBV",
}


def stat_prefix(ch: str) -> str:
    """The chain's statistics plugin for channel *ch*."""
    return f"STAT_{ch}:"


#: Settings whose choices are the PV's own enum strings (read at start).
ENUM_SETTINGS = (
    "Resolution",
    "ImageMode",
    "TriggerMode",
    "TriggerSource",
    "TriggerEdge",
    *(f"{ch}:{name}" for ch in CHANNELS for name in ("Enable", "Range", "Coupling")),
)


class NotWritable(ValueError):
    """A write the allow-list refuses; the message says why."""


Check = Callable[[str, object, Sequence[str]], object]


def _real(lo: float, hi: float) -> Check:
    def check(suffix: str, value: object, _choices: Sequence[str]) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise NotWritable(f"{suffix} takes a number, not {value!r}")
        if not (math.isfinite(value) and lo <= value <= hi):
            raise NotWritable(f"{suffix} takes {lo:g} to {hi:g}")
        return float(value)

    return check


def _whole(lo: int, hi: int) -> Check:
    def check(suffix: str, value: object, _choices: Sequence[str]) -> int:
        whole = isinstance(value, int) or (
            isinstance(value, float) and value.is_integer()
        )
        if isinstance(value, bool) or not whole or not lo <= value <= hi:  # type: ignore[operator]
            raise NotWritable(f"{suffix} takes a whole number {lo} to {hi}")
        return int(value)  # type: ignore[arg-type]

    return check


def _choice(suffix: str, value: object, choices: Sequence[str]) -> str:
    if not choices:
        raise NotWritable(
            f"{suffix}: its choices have not been read from the scope yet"
        )
    if value not in choices:
        raise NotWritable(f"{suffix} takes one of {', '.join(choices)}")
    return str(value)


#: The panel's writes: every setting of the scope (driving it is the panel's
#: point), nothing a capture or the chain produces. suffix → validator.
WRITABLE: dict[str, Check] = {
    "Acquire": _whole(0, 1),
    "NumImages": _whole(1, 1_000_000),
    "SampleInterval": _real(0.0, 1e9),
    "NumSamples": _whole(1, 100_000_000),
    "PreTrigger": _real(0.0, 100.0),
    "TriggerLevel": _real(-50.0, 50.0),
    "AutoTriggerTime": _whole(1, 32767),
    **{f"{ch}:Offset": _real(-50.0, 50.0) for ch in CHANNELS},
    **{suffix: _choice for suffix in ENUM_SETTINGS},
}


def check_write(
    suffix: str, value: object, enums: Mapping[str, Sequence[str]]
) -> object:
    """The value to put for an allowed write, or :class:`NotWritable`.

    *enums* maps each enum setting to the choices its PV reported.
    """
    check = WRITABLE.get(suffix)
    if check is None:
        raise NotWritable(f"{suffix} is not writable from the panel")
    return check(suffix, value, enums.get(suffix, ()))


def monitored() -> dict[str, tuple[str, bool]]:
    """Every monitored suffix → (snapshot key, read as string)."""
    out = {suffix: (key, key in STRING_KEYS) for key, suffix in SCOPE_FIELDS.items()}
    out.update({suffix: (key, False) for key, suffix in ARRAY_SHAPE.items()})
    for ch in CHANNELS:
        for key, suffix in CHANNEL_FIELDS.items():
            out[f"{ch}:{suffix}"] = (f"{ch}.{key}", key in STRING_KEYS)
        for key, suffix in STAT_FIELDS.items():
            out[stat_prefix(ch) + suffix] = (f"{ch}.stat.{key}", False)
    return out


def split_channels(
    data: Sequence[float], channels: int, samples: int
) -> list[list[float]]:
    """The capture's channels from ``ARR:ArrayData`` ([channels, samples], channels fastest)."""
    if channels < 1 or samples < 1:
        return []
    n = min(samples, len(data) // channels)
    return [list(data[ch : n * channels : channels]) for ch in range(channels)]


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
