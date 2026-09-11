"""``ScalarsView`` — a device's scalars-only view, addressable as ``X.scalars``.

Every namespace device carries one so a preset's ``save_images: false``
expands to ``X.scalars`` whatever the device is (plan of record §4.D): on
a :class:`~geecs_bluesky.devices.detector.GeecsDetector` the view waits
for the shot and writes no files; on a scalar-only device it reads what
the device reads.  A childless ophyd-async ``Device``: the RE Manager
discovers it as the sub-device ``X.scalars`` (``profile_ops`` walks
``children()``), ``stage_wrapper`` stages the **root** (the parent), and
the readings keep the parent's column names.
"""

from __future__ import annotations

from typing import Any

from bluesky.protocols import Reading
from event_model import DataKey
from ophyd_async.core import DEFAULT_TIMEOUT, Device, DeviceMock


class ScalarsView(Device):
    """The scalars-only view of *owner*; subclasses say what a read is."""

    def __init__(self, owner: Device) -> None:
        # A plain attribute, not a child: Device.__setattr__ would register
        # the owner as this view's child and the naming walk would cycle.
        object.__setattr__(self, "_owner", owner)
        super().__init__()

    @property
    def _geecs_device_name(self) -> str:
        return self._owner._geecs_device_name

    @property
    def _column_headers(self) -> dict[str, str]:
        return self._owner._column_headers

    async def connect(
        self,
        mock: Any = False,
        timeout: float = DEFAULT_TIMEOUT,
        force_reconnect: bool = False,
    ) -> None:
        """Connect this (childless) view, then the owner whose signals it reads.

        Touched on its own (``connect_on_demand`` before a ``trigger`` /
        ``read`` of ``X.scalars``) it connects the owner; called *by* the
        owner's connect — a ``DeviceMock`` handed down in mock mode, a
        running connect task in real mode — it must not call back up.
        """
        await super().connect(
            mock=mock, timeout=timeout, force_reconnect=force_reconnect
        )
        if isinstance(mock, DeviceMock):
            return
        task = getattr(self._owner, "_connect_task", None)
        if task is not None and not task.done():
            return
        await self._owner.connect(
            mock=mock, timeout=timeout, force_reconnect=force_reconnect
        )

    def covers(self, obj: Any) -> bool:
        """Whether this view already reads *obj* (a child of the owner the owner logs).

        A view and the owner's Movable child are *siblings*, so
        ``separate_devices`` keeps both in a scan's read list; if the child
        is one of the owner's logged readables its readback would land twice
        in one event (the RunEngine refuses colliding data keys after the
        trigger box is armed).  The strict ``take_reading`` drops a listed
        object the view covers.
        """
        owner = self._owner
        signals = getattr(owner, "_scalar_signals", None)  # a GeecsDetector's columns
        if signals is not None:
            return any(obj is s for s in signals())
        read = getattr(obj, "read", None)  # a StandardReadable's registered readers
        return read is not None and read in getattr(owner, "_read_funcs", ())

    async def read(self) -> dict[str, Reading]:
        """The owner's scalar columns — same keys as the owner."""
        return await self._owner.read()

    async def describe(self) -> dict[str, DataKey]:
        """Data keys of the owner's scalar columns."""
        return await self._owner.describe()
