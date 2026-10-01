"""``ScalarsView`` — a device's scalars-only view, addressable as ``X.scalars``.

Every namespace device carries one so a preset's ``save_images: false``
expands to ``X.scalars`` whatever the device is: on
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


def owner_of(obj: Any) -> Any:
    """The device *obj* stands for: a view's owner, anything else itself.

    The one unwrapping rule: a ``.scalars`` view is its owner's columns,
    staged as its owner and named, in any stream or document key, as its
    owner (the non-essential stream ``<owner>_stream`` and the start
    document's ``non_essential`` list must agree on that name).
    """
    return obj._owner if isinstance(obj, ScalarsView) else obj


def geecs_device_name(obj: Any) -> str:
    """The GEECS device *obj* belongs to; its ophyd name when it belongs to none.

    One rule for every plan object: a device names itself, a ``.scalars``
    view names its owner, a detector's signal or a settable child names the
    nearest ancestor that is a GEECS device, and a plan-side object with no
    device behind it (the bin counter) gives its own name.  The plans'
    failure messages, the liveness gate and the background telemetry all
    name devices with it.
    """
    node: Any = owner_of(obj)
    for _ in range(8):
        name = getattr(node, "_geecs_device_name", None)
        if name is not None:
            return str(name)
        node = getattr(node, "parent", None)
        if node is None:
            break
    return str(getattr(obj, "name", obj))


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

    @property
    def connected_status(self) -> Any | None:
        """The owner's gateway liveness PV (``None`` for an owner without one).

        Read by the run's liveness gate and the strict refire gate: a
        ``.scalars`` view of a dead device must be named like the device
        (a scalar-only device's view would otherwise be invisible to the
        gate).
        """
        return getattr(self._owner, "connected_status", None)

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
