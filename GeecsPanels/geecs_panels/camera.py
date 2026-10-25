"""The camera panel's contract with the IOC, and how a value is judged fresh.

Every camera IOC runs the same plugin chain (geecs-epics, runbook
"Display and analysis chain"), so the panel addresses any camera by these
suffixes under its prefix. A camera whose chain PVs do not connect is shown
as such, naming the missing suffixes — never as a half-working page.

**Fresh is judged by the producing plugin, not by the value's own
monitor.** NDStats' scalar records post only when the value changes, so a
steady centroid sends nothing; the plugin's ``ArrayCounter_RBV`` advances
on every frame it processes. A reading is fresh while its plugin's counter
is advancing — and its plugin is computing it: with ``ComputeCentroid``
off the counter still ticks but the centroid is frozen, which is ``off``,
not fresh. Settings (exposure, detector state) post when they change and
are fresh while connected.

Nothing here touches the network: :mod:`geecs_panels.ca` feeds values in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

from geecs_panels.bridge.freshness import judge_freshness

#: Plugins whose ArrayCounter_RBV the panel watches: freshness for the
#: readings they produce, and (MJPG1) the image's own age.
COUNTER_PLUGINS = ("cam1", "STAT", "STAT0", "MJPG1")

#: The image comes from MJPG1; its age is MJPG1's counter.
IMAGE_PLUGIN = "MJPG1"

Freshness = Literal["fresh", "stale", "off", "disconnected"]


def counter_suffix(plugin: str) -> str:
    """The plugin's frame counter, e.g. ``STAT:ArrayCounter_RBV``."""
    return f"{plugin}:ArrayCounter_RBV"


@dataclass(frozen=True)
class Reading:
    """One live value on the panel.

    ``plugin`` names the producer whose counter judges freshness (``None``
    for a setting: fresh while connected); ``computed_by`` the flag that
    must be on for the value to be computed at all.
    """

    key: str
    label: str
    suffix: str
    unit: str = ""
    plugin: Optional[str] = None
    computed_by: Optional[str] = None
    as_string: bool = False


#: The read-only set of build step 1: the camera's state and the analysis
#: statistics. Order is display order.
READINGS: tuple[Reading, ...] = (
    Reading("state", "detector", "cam1:DetectorState_RBV", as_string=True),
    Reading("rate", "frame rate", "cam1:ArrayRate_RBV", "Hz"),
    # Seconds on the wire; the page shows milliseconds (camera.js SCALE).
    Reading("exposure", "exposure", "cam1:AcquireTime_RBV", "ms"),
    Reading("pixel_format", "pixel format", "cam1:PixelFormat_RBV", as_string=True),
    Reading(
        "cx", "centroid x", "STAT:CentroidX_RBV", "px", "STAT", "STAT:ComputeCentroid"
    ),
    Reading(
        "cy", "centroid y", "STAT:CentroidY_RBV", "px", "STAT", "STAT:ComputeCentroid"
    ),
    Reading("sx", "sigma x", "STAT:SigmaX_RBV", "px", "STAT", "STAT:ComputeCentroid"),
    Reading("sy", "sigma y", "STAT:SigmaY_RBV", "px", "STAT", "STAT:ComputeCentroid"),
    Reading("max", "max", "STAT:MaxValue_RBV", "", "STAT", "STAT:ComputeStatistics"),
    Reading("mean", "mean", "STAT:MeanValue_RBV", "", "STAT", "STAT:ComputeStatistics"),
    Reading("total", "total", "STAT:Total_RBV", "", "STAT", "STAT:ComputeStatistics"),
    Reading(
        "raw_max",
        "raw max",
        "STAT0:MaxValue_RBV",
        "",
        "STAT0",
        "STAT0:ComputeStatistics",
    ),
)

#: The URL of one JPEG frame, published by the IOC (read once, not monitored).
JPG_URL_SUFFIX = "MJPG1:JPG_URL_RBV"

#: The crosshairs the panel draws: OVER1's overlays 1 and 2. The IOC stores
#: them (shared by every viewer, autosaved); the panel draws them, in their
#: stored colour, over the colour-mapped image — the stream carries none.
CROSSHAIRS = (1, 2)
#: Field → readback suffix under ``OVER1:<n>:``; positions are image
#: (PROC1) pixels. ``Center*`` is what the panel sets: the IOC derives
#: ``Position*`` (the autosaved one) from it.
CROSSHAIR_FIELDS = {
    "use": "Use_RBV",
    "x": "CenterX_RBV",
    "y": "CenterY_RBV",
    "sx": "SizeX_RBV",
    "sy": "SizeY_RBV",
    "r": "Red_RBV",
    "g": "Green_RBV",
    "b": "Blue_RBV",
    "name": "Name_RBV",
    "shape": "Shape_RBV",
}
_CROSSHAIR_STRINGS = frozenset({"name", "shape"})

#: The analysis ROI (ROI1: STAT only sees this region), field → readback.
#: Min/Size are in image (PROC1) pixels; STAT's results are in the ROI's own
#: output pixels — :mod:`geecs_panels.coords` converts.
ROI_FIELDS = {
    "minx": "ROI1:MinX_RBV",
    "miny": "ROI1:MinY_RBV",
    "sizex": "ROI1:SizeX_RBV",
    "sizey": "ROI1:SizeY_RBV",
    "binx": "ROI1:BinX_RBV",
    "biny": "ROI1:BinY_RBV",
    "revx": "ROI1:ReverseX_RBV",
    "revy": "ROI1:ReverseY_RBV",
}


#: The image's true size in image (PROC1) pixels: MJPG1's input array. The
#: stream itself may be smaller — ffmpegServer scales a frame down to fit
#: MAXW x MAXH (1600 x 1200 by default; a rotated 1280 x 1024 frame arrives
#: as 960 x 1200) — so the page stretches the JPEG to this size, and every
#: position (crosshairs, ROI, cursor) stays in image pixels.
IMAGE_SIZE_FIELDS = {"width": "MJPG1:ArraySize0_RBV", "height": "MJPG1:ArraySize1_RBV"}

#: Processing state the panel shows and edits: the background (PROC1) and
#: the transform (TRANS1, which has no readback record: the output record
#: itself is monitored). Field → suffix.
PROCESSING_FIELDS = {
    "background_on": "PROC1:EnableBackground_RBV",
    "background_valid": "PROC1:ValidBackground_RBV",
    "background_saving": "PROC1:SaveBackground_RBV",
    "transform": "TRANS1:Type",
}
#: NDTransform's choices, in the record's order.
TRANSFORMS = (
    "None",
    "Rot90",
    "Rot180",
    "Rot270",
    "Mirror",
    "Rot90Mirror",
    "Rot180Mirror",
    "Rot270Mirror",
)

#: Each pixel format's full scale, as the driver's array carries it. A
#: Blackfly's "Mono16" is 12 bits LEFT-aligned (multiples of 16, top 65520):
#: the 99% threshold below catches it at the same full scale as true 16-bit.
FULL_SCALE = {"Mono8": 255, "Mono10": 1023, "Mono12": 4095, "Mono16": 65535}
#: Saturated at or above this fraction of full scale.
SATURATION_FRACTION = 0.99


def saturation(pixel_format: object, raw_max: object) -> dict[str, object]:
    """Whether the raw frame's maximum (STAT0) is at the pixel format's top.

    ``state`` is ``saturated``, ``ok``, or ``unknown`` (a format not in
    :data:`FULL_SCALE`, or no reading yet); ``fraction`` is max / full scale.
    """
    full = FULL_SCALE.get(str(pixel_format)) if pixel_format is not None else None
    if full is None or not isinstance(raw_max, (int, float)):
        return {"state": "unknown", "fraction": None, "full_scale": full}
    fraction = float(raw_max) / full
    state = "saturated" if fraction >= SATURATION_FRACTION else "ok"
    return {"state": state, "fraction": fraction, "full_scale": full}


#: STAT's arrays for the plots. READ ON REQUEST, never monitored: monitored,
#: they would arrive at the camera's full rate (~0.6 MB/s over the VPN at
#: 18 Hz) for plots nobody reads that fast. Key → suffix.
ARRAYS = {
    "profile_x": "STAT:ProfileAverageX_RBV",
    "profile_y": "STAT:ProfileAverageY_RBV",
    "cursor_x": "STAT:ProfileCursorX_RBV",
    "cursor_y": "STAT:ProfileCursorY_RBV",
    "histogram": "STAT:Histogram_RBV",
}
#: The scalars the plots need (monitored). Profiles and the cursor are in
#: the ROI's output pixels; :mod:`geecs_panels.coords` maps them.
PLOT_FIELDS = {
    "profile_size_x": "STAT:ProfileSizeX_RBV",
    "profile_size_y": "STAT:ProfileSizeY_RBV",
    "hist_min": "STAT:HistMin_RBV",
    "hist_max": "STAT:HistMax_RBV",
    "cursor_x": "STAT:CursorX_RBV",
    "cursor_y": "STAT:CursorY_RBV",
    "profiles_on": "STAT:ComputeProfiles",
    "histogram_on": "STAT:ComputeHistogram",
}


#: Writes under these prefixes change what scans save: refused while a run
#: holds the camera (RUN:Lock), or while the lock cannot be read.
LOCKED_PREFIXES = ("cam1:",)

#: The run lock and what the override dialog shows beside it.
LOCK_FIELDS = {
    "lock": "RUN:Lock",
    "owner": "RUN:Owner",
    "capture": "HDF1:Capture_RBV",
}

#: The camera's acquisition state for its controls (readbacks; the
#: exposure is a reading already).
ACQUISITION_FIELDS = {
    "period": "cam1:AcquirePeriod_RBV",
    "frame_rate": "cam1:FrameRate_RBV",
    "gain": "cam1:Gain_RBV",
    "trigger_mode": "cam1:TriggerMode_RBV",
    "trigger_source": "cam1:TriggerSource_RBV",
    "acquire": "cam1:Acquire_RBV",
    "min_x": "cam1:MinX_RBV",
    "min_y": "cam1:MinY_RBV",
    "size_x": "cam1:SizeX_RBV",
    "size_y": "cam1:SizeY_RBV",
    "max_x": "cam1:MaxSizeX_RBV",
    "max_y": "cam1:MaxSizeY_RBV",
}
_STRING_FIELDS = {
    "RUN:Lock",
    "RUN:Owner",
    "HDF1:Capture_RBV",
    "cam1:TriggerMode_RBV",
    "cam1:TriggerSource_RBV",
    "cam1:Acquire_RBV",
}


def lock_state(values: dict[str, object], connected: dict[str, bool]) -> str:
    """``free``, ``locked`` or ``unknown`` (the lock not served or not read)."""
    lock = LOCK_FIELDS["lock"]
    return lock_word(bool(connected.get(lock)), values.get(lock))


def lock_word(connected: bool, value: object) -> Literal["free", "locked", "unknown"]:
    """The run lock's word from its record: ``unknown`` unless connected, ``free`` only on its cleared label."""
    if not connected:
        return "unknown"
    # Free only on the record's cleared label: any other value is held.
    return "free" if value == "Free" else "locked"


def crosshair_suffix(n: int, field: str) -> str:
    """``OVER1:<n>:<readback>`` for a crosshair field."""
    return f"OVER1:{n}:{CROSSHAIR_FIELDS[field]}"


@dataclass(frozen=True)
class Whole:
    """A whole number in ``lo..hi`` (a pixel, a size, a 0..255 level)."""

    lo: int
    hi: int


@dataclass(frozen=True)
class Real:
    """A finite number in ``lo..hi`` (an exposure in seconds, a gain)."""

    lo: float
    hi: float


_PIXEL = Whole(0, 65535)
_SIZE = Whole(1, 65535)
_LEVEL = Whole(0, 255)

#: The PVs the panel may WRITE, and nothing else (design note §4). Display
#: and analysis state freely; the camera driver's (``cam1:``, what scans
#: save) only while the run lock is free (``LOCKED_PREFIXES``). The lock
#: itself is never here: only the override route releases it. Value kinds:
#: :class:`Whole` (a number in a range), ``bool`` (0/1), ``str``, or a tuple
#: of allowed choices (an enum by name, or ``(1,)`` for a press-only record).
WRITABLE: dict[str, object] = {
    **{
        f"OVER1:{n}:{field}": kind
        for n in CROSSHAIRS
        for field, kind in (
            ("Use", bool),
            ("CenterX", _PIXEL),
            ("CenterY", _PIXEL),
            ("SizeX", _PIXEL),  # an overlay may be a 0-wide line
            ("SizeY", _PIXEL),
            ("Red", _LEVEL),
            ("Green", _LEVEL),
            ("Blue", _LEVEL),
            ("Name", str),
            ("Shape", ("Cross", "Rectangle")),
        )
    },
    "ROI1:MinX": _PIXEL,
    "ROI1:MinY": _PIXEL,
    "ROI1:SizeX": _SIZE,
    "ROI1:SizeY": _SIZE,
    # Background: grab one from the next frame (1 only — the record's "Save"),
    # and subtract it or not. Display and statistics only: scans save raw
    # frames from the driver.
    # The lineout cursor (ROI output pixels): the panel converts a clicked
    # image point first. Analysis state only.
    "STAT:CursorX": _PIXEL,
    "STAT:CursorY": _PIXEL,
    # The camera itself (build step 3): what scans save. Every one is in the
    # IOC's ACQUIRE access group — refused while a run holds RUN:Lock — and
    # the panel refuses them itself first (LOCKED_PREFIXES).
    "cam1:AcquireTime": Real(0.0, 60.0),
    "cam1:AcquirePeriod": Real(0.0, 3600.0),
    "cam1:Gain": Real(0.0, 100.0),
    "cam1:TriggerMode": ("Off", "On"),
    "cam1:Acquire": bool,
    "cam1:MinX": _PIXEL,
    "cam1:MinY": _PIXEL,
    "cam1:SizeX": _SIZE,
    "cam1:SizeY": _SIZE,
    "PROC1:SaveBackground": (1,),
    "PROC1:EnableBackground": bool,
    "TRANS1:Type": TRANSFORMS,
}


#: Writes whose put completes only when the work ends: a Continuous acquire's
#: put returns when acquisition stops. Dispatched without waiting, then
#: confirmed by the readback (as GeecsBluesky's NativeAcquireLogic does).
SETTLED_BY_READBACK = {"cam1:Acquire": "cam1:Acquire_RBV"}


class NotWritable(ValueError):
    """A write the allow-list refuses; the message says why."""


def _as_flag(suffix: str, value: object) -> object:
    if value in (0, 1, True, False):
        return int(bool(value))
    raise NotWritable(f"{suffix} takes 0 or 1, not {value!r}")


def _as_whole(suffix: str, value: object, kind: Whole) -> object:
    whole = isinstance(value, int) or (isinstance(value, float) and value.is_integer())
    if isinstance(value, bool) or not whole:
        raise NotWritable(f"{suffix} takes a whole number, not {value!r}")
    if not kind.lo <= int(value) <= kind.hi:
        raise NotWritable(f"{suffix} takes {kind.lo}..{kind.hi}, not {value!r}")
    return int(value)


#: An EPICS string record holds at most this many characters.
MAX_TEXT = 39


def _as_text(suffix: str, value: object) -> object:
    if not isinstance(value, str) or len(value) > MAX_TEXT:
        raise NotWritable(f"{suffix} takes text of at most {MAX_TEXT} characters")
    return value


def _as_real(suffix: str, value: object, kind: Real) -> object:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NotWritable(f"{suffix} takes a number, not {value!r}")
    if not math.isfinite(value) or not kind.lo <= value <= kind.hi:
        raise NotWritable(f"{suffix} takes {kind.lo}..{kind.hi}, not {value!r}")
    return float(value)


_COERCE = {bool: _as_flag, str: _as_text}


def check_write(suffix: str, value: object) -> object:
    """The value to put for an allowed write, or :class:`NotWritable`.

    Coerces JSON's types to the PV's: a pixel or level is an ``int`` (a
    ``bool`` or a non-integral float is refused), a flag 0/1, a choice one of
    its names, text a short string.
    """
    kind = WRITABLE.get(suffix)
    if kind is None:
        raise NotWritable(f"{suffix} is not writable from the panel")
    if isinstance(kind, tuple):
        if isinstance(value, bool) or value not in kind:
            raise NotWritable(f"{suffix} takes one of {', '.join(map(str, kind))}")
        return value
    if isinstance(kind, Whole):
        return _as_whole(suffix, value, kind)
    if isinstance(kind, Real):
        return _as_real(suffix, value, kind)
    return _COERCE[kind](suffix, value)


def monitored_suffixes() -> list[str]:
    """Every PV suffix the panel monitors, without duplicates, in a stable order."""
    groups: list[list[str]] = [
        [r.suffix for r in READINGS],
        [r.computed_by for r in READINGS if r.computed_by],
        [counter_suffix(p) for p in COUNTER_PLUGINS],
        [crosshair_suffix(n, f) for n in CROSSHAIRS for f in CROSSHAIR_FIELDS],
        list(ROI_FIELDS.values()),
        list(PROCESSING_FIELDS.values()),
        list(IMAGE_SIZE_FIELDS.values()),
        list(PLOT_FIELDS.values()),
        list(LOCK_FIELDS.values()),
        list(ACQUISITION_FIELDS.values()),
    ]
    return list(dict.fromkeys(s for group in groups for s in group))


def string_suffixes() -> frozenset[str]:
    """The suffixes read as strings (enums shown by name)."""
    return frozenset(
        [r.suffix for r in READINGS if r.as_string]
        + [crosshair_suffix(n, f) for n in CROSSHAIRS for f in _CROSSHAIR_STRINGS]
        + [PROCESSING_FIELDS["transform"]]
        + sorted(_STRING_FIELDS)
    )


def judge(
    reading: Reading,
    *,
    connected: dict[str, bool],
    values: dict[str, object],
    counter_age: dict[str, Optional[float]],
    stale_after: float,
) -> Freshness:
    """How fresh *reading* is now.

    Parameters
    ----------
    reading : Reading
        The value being judged.
    connected : dict
        Suffix → whether its channel is connected.
    values : dict
        Suffix → latest value (the ``Compute*`` flags are read from here).
    counter_age : dict
        Plugin → seconds since its counter last advanced, ``None`` if it
        has not advanced since the panel started watching.
    stale_after : float
        Seconds without a counter advance after which a value is stale.
    """
    flag = reading.computed_by
    plugin = reading.plugin
    word = judge_freshness(
        bool(connected.get(reading.suffix)),
        computed=(bool(connected.get(flag)), values.get(flag)) if flag else None,
        counter=(
            (bool(connected.get(counter_suffix(plugin))), counter_age.get(plugin))
            if plugin
            else None
        ),
        stale_after=stale_after,
    )
    return "disconnected" if word == "unknown" else word
