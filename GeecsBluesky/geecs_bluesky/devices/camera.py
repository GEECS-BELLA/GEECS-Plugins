"""GeecsCameraBase — base class for GEECS camera/detector devices.

Combines :class:`~geecs_bluesky.devices.triggerable.GeecsTriggerable` with
:class:`~geecs_bluesky.devices.geecs_device.GeecsDevice` to produce a device
that:

* Exposes one readable signal — the path to the last saved image file.
* Responds to ``trigger()`` by waiting for ``acq_timestamp`` to advance
  (i.e., the DG645 fires and the camera saves the next shot).

The hardware saves images natively via its SDK; GEECS reports the path through
a device variable (e.g. ``"SavedFile"``).  Each ``trigger_and_read`` call
waits for a new shot then reads the file-path signal, recording it in the
Bluesky event document.

Typical usage::

    from geecs_bluesky.devices.camera import GeecsCameraBase

    cam = GeecsCameraBase(
        device_name="U_ProbeCam",
        host="192.168.8.50",
        port=64000,
        name="probe_cam",
        filepath_variable="SavedFile",
    )
    await cam.connect()

    # In a plan:
    yield from bps.trigger_and_read([cam])
    # Bluesky emits an event with {"probe_cam-filepath": "/data/.../img001.tif"}

Subclassing::

    class ProbeCam(GeecsCameraBase):
        _acq_timestamp_variable = "acq_timestamp"   # default; override if needed

        def __init__(self, host, port, name="probe_cam"):
            super().__init__("U_ProbeCam", host, port,
                             name=name, filepath_variable="SavedFile")
"""

from __future__ import annotations

import logging
from typing import Any

from geecs_bluesky.devices.geecs_device import GeecsDevice
from geecs_bluesky.devices.triggerable import GeecsTriggerable
from geecs_bluesky.signals import geecs_signal_r
from geecs_bluesky.transport.udp_client import GeecsUdpClient

logger = logging.getLogger(__name__)


class GeecsCameraBase(GeecsTriggerable, GeecsDevice):
    """GEECS camera/detector: Triggerable + readable file-path signal.

    MRO: ``GeecsCameraBase → GeecsTriggerable → GeecsDevice → StandardReadable``

    Parameters
    ----------
    device_name:
        GEECS device name as it appears in the protocol (e.g. ``"U_ProbeCam"``).
    host:
        Device IP address.
    port:
        Device UDP/TCP port.
    name:
        ophyd-async device name (used to namespace signal keys).
    filepath_variable:
        GEECS variable that holds the path of the last saved file.
        Default: ``"SavedFile"``.
    acq_timestamp_variable:
        Variable used to detect shot completion.  Override the default
        ``"acq_timestamp"`` here or via the class attribute
        :attr:`~geecs_bluesky.devices.triggerable.GeecsTriggerable._acq_timestamp_variable`.
    """

    def __init__(
        self,
        device_name: str,
        host: str,
        port: int,
        name: str = "camera",
        filepath_variable: str = "SavedFile",
        acq_timestamp_variable: str | None = None,
    ) -> None:
        udp = GeecsUdpClient(host, port)
        with self.add_children_as_readables():
            self.filepath = geecs_signal_r(
                str,
                device_name,
                filepath_variable,
                host,
                port,
                shared_udp=udp,
            )
        super().__init__(name=name, shared_udp=udp)

        if acq_timestamp_variable is not None:
            # Instance-level override (not just class-level default).
            self._acq_timestamp_variable = acq_timestamp_variable

        self._geecs_device_name = device_name
        logger.debug(
            "GeecsCameraBase '%s': filepath=%s, acq_ts=%s",
            name,
            filepath_variable,
            self._acq_timestamp_variable,
        )

    # ------------------------------------------------------------------
    # DB constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_db(
        cls,
        device_name: str,
        name: str = "camera",
        **kwargs: Any,
    ) -> "GeecsCameraBase":
        """Construct by looking up ``(host, port)`` from the GEECS database.

        Parameters
        ----------
        device_name:
            GEECS device name to resolve.
        name:
            ophyd-async device name.
        **kwargs:
            Forwarded to :class:`GeecsCameraBase` (filepath_variable, …).
        """
        from geecs_bluesky.db.geecs_db import GeecsDb

        host, port = GeecsDb.find_device(device_name)
        logger.info("DB resolved %s → %s:%s", device_name, host, port)
        return cls(device_name, host, port, name=name, **kwargs)
