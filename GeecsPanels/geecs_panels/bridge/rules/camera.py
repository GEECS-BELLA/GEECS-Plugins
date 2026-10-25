"""The areaDetector camera chain's rules (``ad_camera_chain``): the run lock, saturation, the history, the profiles.

- **The run lock** (rule 6): ``RUN:Lock`` read by name, ``free`` only on
  its cleared label; unread or not served is ``unknown``, which refuses
  the locked roles (:func:`geecs_panels.camera.lock_word`). Its owner is
  carried while it is held.
- **Saturation on the raw frame** (rule 5): STAT0's maximum, before
  background and ROI, against the pixel format's full scale
  (:func:`geecs_panels.camera.saturation`), and only while that maximum
  is fresh: a stale one says ``unknown``.
- **The history** is the centroid, mean and max, sampled on STAT's frame
  counter (STAT's own time-series buffers are IOC-global and idle).
- **The profiles** are cut to their valid length (``ProfileSize``), in the
  analysis ROI's output pixels; the page maps them to image pixels.
"""

from __future__ import annotations

from collections.abc import Mapping

from geecs_panels.bridge.arrays import Trace, trace
from geecs_panels.bridge.feed import Feed, HistorySpec
from geecs_panels.bridge.rules import Rules
from geecs_schemas.epics.family import Role

from geecs_panels.camera import check_write, lock_word, saturation
from geecs_panels.contracts import RunLock, Saturation

#: The roles the history keeps, sampled on STAT's counter.
HISTORY = HistorySpec(keys=("cx", "cy", "mean", "max"), tick="stat_counter")

#: Profile role → the role holding its valid length.
_LENGTHS = {
    "profile_x": "profile_size_x",
    "cursor_profile_x": "profile_size_x",
    "profile_y": "profile_size_y",
    "cursor_profile_y": "profile_size_y",
}


def lock(feed: Feed) -> RunLock:
    """The run lock and, while it is held, its owner."""
    if not feed.device.has_role("lock"):
        return RunLock(state="unknown")
    state = lock_word(feed.connected("lock"), feed.value("lock"))
    owner = feed.value("owner") if state == "locked" else None
    return RunLock(state=state, owner=str(owner) if owner else None)


def saturation_chip(feed: Feed, stale_after: float) -> Saturation:
    """Whether the raw frame reaches full scale, judged on a FRESH raw max only."""
    device = feed.device
    if not (device.has_role("raw_max") and device.has_role("pixel_format")):
        return Saturation(state="unknown")
    raw = device.role("raw_max")
    if feed.freshness_of(raw, "", stale_after) != "fresh":
        return Saturation(state="unknown")
    chip = saturation(feed.value("pixel_format"), feed.value("raw_max"))
    return Saturation(state=chip["state"], fraction=chip["fraction"])


def arrays(
    feed: Feed, read: Mapping[str, list[float]], columns: int
) -> dict[str, Trace]:
    """Each array decimated, a profile first cut to its valid length."""
    out: dict[str, Trace] = {}
    for name, values in read.items():
        if feed.device.role(name).type != "array":
            continue
        length = _LENGTHS.get(name)
        n = feed.value(length) if length is not None else None
        if isinstance(n, (int, float)) and not isinstance(n, bool):
            values = values[: int(n)]
        out[name] = trace(values, columns)
    return out


def check(
    feed: Feed, role: Role, axis: str, value: object, choices: list[str]
) -> object:
    """The camera's own table (``camera.WRITABLE``) by the record written: the same bounds the camera page keeps."""
    return check_write(role.write_suffix or role.suffix, value)


RULES = Rules(
    history=HISTORY, lock=lock, saturation=saturation_chip, arrays=arrays, check=check
)
