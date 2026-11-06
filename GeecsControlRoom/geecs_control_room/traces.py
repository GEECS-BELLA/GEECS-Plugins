"""A device's trace, read over Channel Access by this server: read only, at the PVs the module names.

``GET /api/devices/{name}/trace`` serves the trace page (``device_trace.html``):
one array over a calibrated axis, bound by data shape, never by model.
Which roles those are is the family file's: the axis is the role with
``axis_of``, the trace (y) an array it names, ``?y=<role>`` or else the
first (a spectrometer's raw ``spectrum``, or ``processed``, its
background-subtracted twin from PROC's own ``ARR``); a y array with ``channels_by``
is ``[channels, samples]``, channels fastest (a scope's ``ARR`` capture, a
disabled channel NaN), else one channel; the trace is held on the y
array's ``fresh_by`` counter, as ``frame.raw`` is held on ``raw_counter``.

- **200**: float32 little-endian, gzip framed: the axis (``samples``
  values) then the y array as the IOC lays it out (``channels × samples``,
  channels fastest); headers ``X-Trace-Channels``, ``X-Trace-Samples``,
  ``X-Trace-Tag`` = the quoted counter, ``ETag`` the same.
- **304** on an ``If-None-Match`` naming the held ``ETag``.
- **404** when the family names no trace or counter, ``?y=`` names an array
  the axis does not draw, or the module names no such role; **503** with words when a source is silent or the shapes disagree.

Reads: one in flight per device, held ``TRACE_HOLD_S`` and shared. The
counter alone first; unmoved, the held trace is served (a 2 Hz poller costs
one scalar read). Else the y array (asked ``DBR_FLOAT``: float32 on the
wire, half a ``DOUBLE`` waveform's bytes) and its channel count in one
batch, then the counter again; on a mismatch the batch is read once more
and tagged by the second counter. The axis is read with the first trace
and again only when its length is not the trace's, its first two values
moved (a probe of eight bytes per new capture: a new interval or
pre-trigger moves them) or it is older than ``AXIS_EVERY_S``, never whole
per capture. No write ever leaves this module.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from geecs_schemas import FamilyFile, Role, load_families

from geecs_control_room.images import (
    Images,
    Shared,
    Unavailable,
    gzip_frame,
    etag_matches,
)
from geecs_control_room.module_client import ModuleClient

#: A trace is held and shared this long (at most 2 Hz per device, however many viewers poll).
TRACE_HOLD_S = 0.5
#: The axis is read again at most this often while its length matches the trace's.
AXIS_EVERY_S = 10.0


@dataclass(frozen=True)
class Trace:
    """One trace as served: the gzip body (axis, then y), its shape and its tag (the counter)."""

    body: bytes
    channels: int
    samples: int
    tag: str


@dataclass(frozen=True)
class TracePvs:
    """The PVs one trace is read at: the y array, its axis, its counter and (several channels) its channel count."""

    y: str
    axis: str
    counter: str
    channels: Optional[str]


def trace_roles(family: Optional[FamilyFile]) -> Optional[tuple[Role, list[Role]]]:
    """The family's trace: its first axis role (``axis_of``, in the family's order) and the arrays it draws, in its order; ``None`` without one.

    The one reading of the rule, for this route and the page's template alike.
    """
    axis = next((r for r in family.roles if r.axis_of), None) if family else None
    if family is None or axis is None or not axis.axis_of:
        return None
    return axis, [family.role(y) for y in axis.axis_of]


def trace_pvs(family_name: Optional[str], roles: dict[str, dict[str, Any]], name: str, y_name: Optional[str] = None) -> TracePvs:  # fmt: skip
    """The trace's PVs: the family says which roles (*y_name*, else the axis' first array), the module's device route names their PVs.

    Raises
    ------
    Unavailable
        404 when the family names no trace (no ``axis_of``) or no counter for
        it (``fresh_by``), *y_name* is no array the axis draws, or the module
        names none of a role's PVs.
    """
    pair = trace_roles(load_families().get(family_name or ""))
    if pair is None:
        raise Unavailable(404, f"the family of {name} names no trace (no role is an axis_of)")  # fmt: skip
    axis, ys = pair
    y = next((r for r in ys if r.name == y_name), None) if y_name else ys[0]
    if y is None:
        raise Unavailable(404, f"the axis {axis.name} draws no array {y_name!r}: it draws {', '.join(r.name for r in ys)}")  # fmt: skip
    if y.fresh_by is None:
        raise Unavailable(404, f"the family names no counter for {y.name} (fresh_by)")
    wanted = [y.name, axis.name, y.fresh_by] + ([y.channels_by] if y.channels_by else [])  # fmt: skip
    missing = [r for r in wanted if r not in roles]
    if missing:
        raise Unavailable(404, f"the module names no {', '.join(missing)} role{'s' * (len(missing) > 1)} for {name}")  # fmt: skip
    channels = roles[y.channels_by]["pv"] if y.channels_by else None
    return TracePvs(roles[y.name]["pv"], roles[axis.name]["pv"], roles[y.fresh_by]["pv"], channels)  # fmt: skip


class Traces:
    """Every device's trace, shared by its viewers; PV names from the module, through :class:`Images`' cache."""

    def __init__(self, images: Images, clock: Callable[[], float] = time.monotonic) -> None:  # fmt: skip
        self._images, self._ca, self._clock = images, images.ca, clock
        self._sources: dict[tuple[str, TracePvs], Shared[Trace]] = {}
        self._held: dict[tuple[str, TracePvs], Trace] = {}
        self._axes: dict[str, tuple[np.ndarray, float]] = {}

    async def trace(self, module: ModuleClient, name: str, y: Optional[str] = None) -> Trace:  # fmt: skip
        """The latest trace of the array *y* (else the axis' first), shared by every viewer within ``TRACE_HOLD_S``."""
        roles = await self._images.roles(module, name)
        pvs = trace_pvs(self._images.family(name), roles, name, y)
        key = (name, pvs)
        source = self._sources.get(key)
        if source is None:
            source = self._sources[key] = Shared(lambda: self._read(key), TRACE_HOLD_S, self._clock)  # fmt: skip
        return await source.get()

    async def _counter(self, pv: str) -> str:
        try:
            read = await self._ca.scalar(pv)
            if not math.isfinite(read):
                raise ValueError(read)
            return str(int(read))
        except Exception as exc:  # any CA failure (or a NaN): said, never a 500
            raise Unavailable(503, f"the capture counter ({pv}) could not be read") from exc  # fmt: skip

    async def _batch(self, pvs: TracePvs) -> tuple[np.ndarray, int]:
        try:
            if pvs.channels is None:
                (y,) = await self._ca.floats([pvs.y])
                return y, 1
            (y,), channels = await asyncio.gather(self._ca.floats([pvs.y]), self._ca.scalar(pvs.channels))  # fmt: skip
            return y, int(channels)
        # any CA failure (a NaN channel count too): said, never a 500
        except Exception as exc:
            raise Unavailable(503, f"the trace ({pvs.y}) could not be read: {exc}") from exc  # fmt: skip

    async def _axis(self, pv: str, samples: int) -> np.ndarray:
        """The axis, held while its length is the trace's, its first two values unchanged and it is younger than ``AXIS_EVERY_S``.

        The probe (two values, eight bytes) catches a timebase change of the
        same length: a scope's axis is ``(i - pre) * interval``, so a new
        interval or pre-trigger moves its first two values.
        """
        held = self._axes.get(pv)
        if held is not None and held[0].size == samples and self._clock() - held[1] < AXIS_EVERY_S:  # fmt: skip
            # an unread probe (any CA failure) reads the whole axis, which says why if it fails too
            (probe,) = await asyncio.gather(self._ca.floats([pv], count=2), return_exceptions=True)  # fmt: skip
            if isinstance(probe, list) and np.array_equal(probe[0][:2], held[0][:2], equal_nan=True):  # fmt: skip
                return held[0]
        asked = self._clock()
        try:
            (axis,) = await self._ca.floats([pv])
        except Exception as exc:  # any CA failure: said, never a 500
            raise Unavailable(503, f"the axis ({pv}) could not be read: {exc}") from exc
        self._axes[pv] = (axis, asked)
        return axis

    async def _read(self, key: tuple[str, TracePvs]) -> Trace:
        pvs = key[1]
        tag = await self._counter(pvs.counter)
        held = self._held.get(key)
        if held is not None and held.tag == tag:
            return held  # no new capture: one scalar read, no array read
        y, channels = await self._batch(pvs)
        after = await self._counter(pvs.counter)
        # a capture landed mid-batch: read once more, tagged by the second counter
        if after != tag:
            tag = after
            y, channels = await self._batch(pvs)
        n = int(np.size(y))
        if channels < 1 or n == 0 or n % channels:
            raise Unavailable(503, f"shape disagrees: {n} values for {channels} channels")  # fmt: skip
        samples = n // channels
        axis = await self._axis(pvs.axis, samples)
        if axis.size != samples:
            raise Unavailable(503, f"the axis ({pvs.axis}) has {axis.size} values for {samples} samples")  # fmt: skip
        body = await asyncio.to_thread(_pack, axis, y)
        trace = self._held[key] = Trace(body, channels, samples, tag)
        return trace


def _pack(axis: np.ndarray, y: np.ndarray) -> bytes:
    """The axis then the y array, float32 little-endian, gzip framed."""
    return gzip_frame(np.concatenate([np.ravel(axis), np.ravel(y)]).astype("<f4", copy=False))  # fmt: skip


def router(traces: Traces) -> APIRouter:
    """The trace, read only."""
    r = APIRouter()

    @r.get("/api/devices/{name}/trace")
    async def trace(request: Request, name: str, y: Optional[str] = None) -> Response:  # fmt: skip
        try:
            t = await traces.trace(request.app.state.module, name, y)
        except Unavailable as exc:
            return JSONResponse({"code": "unavailable", "detail": exc.detail}, status_code=exc.status)  # fmt: skip
        etag = f'"{t.tag}"'
        headers = {"Cache-Control": "no-store", "ETag": etag, "X-Trace-Tag": etag}
        if etag_matches(request.headers.get("if-none-match"), etag):
            return Response(status_code=304, headers=headers)
        headers |= {"Content-Encoding": "gzip", "X-Trace-Channels": str(t.channels), "X-Trace-Samples": str(t.samples)}  # fmt: skip
        return Response(t.body, media_type="application/octet-stream", headers=headers)  # fmt: skip

    return r
