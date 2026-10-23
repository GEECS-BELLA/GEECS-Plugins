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

from dataclasses import dataclass
from typing import Literal, Optional

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


def monitored_suffixes() -> list[str]:
    """Every PV suffix the panel monitors, without duplicates, in a stable order."""
    seen: dict[str, None] = {}
    for reading in READINGS:
        seen[reading.suffix] = None
        if reading.computed_by:
            seen[reading.computed_by] = None
    for plugin in COUNTER_PLUGINS:
        seen[counter_suffix(plugin)] = None
    return list(seen)


def string_suffixes() -> frozenset[str]:
    """The suffixes read as strings (enums shown by name)."""
    return frozenset(r.suffix for r in READINGS if r.as_string)


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
    if not connected.get(reading.suffix):
        return "disconnected"
    if reading.computed_by is not None:
        if not connected.get(reading.computed_by):
            return "disconnected"
        if not values.get(reading.computed_by):
            return "off"
    if reading.plugin is None:
        return "fresh"
    if not connected.get(counter_suffix(reading.plugin)):
        return "disconnected"
    age = counter_age.get(reading.plugin)
    if age is None or age > stale_after:
        return "stale"
    return "fresh"
