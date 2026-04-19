"""JetStage — ophyd-async device for the U_ESP_JetXYZ ESP301 motion controller.

Maps the three position axes of the Undulator jet XYZ stage into a single
:class:`GeecsDevice`.  Variable names are the exact strings sent over the
GEECS wire protocol (``Position.Axis 1``, ``Position.Axis 2``,
``Position.Axis 3``).

Typical usage::

    from geecs_bluesky.devices.jet_stage import JetStage

    # Explicit host/port (testing or when DB is unavailable):
    stage = JetStage("192.168.8.198", 65158, name="jet")
    await stage.connect()

    # Via GEECS database lookup (requires geecs-pythonapi):
    stage = JetStage.from_db("U_ESP_JetXYZ", name="jet")
    await stage.connect()

    reading = await stage.read()
"""

from __future__ import annotations

from geecs_bluesky.devices.geecs_device import GeecsDevice
from geecs_bluesky.signals import geecs_signal_rw
from geecs_bluesky.transport.udp_client import GeecsUdpClient

_DEVICE_NAME = "U_ESP_JetXYZ"


class JetStage(GeecsDevice):
    """XYZ jet stage driven by an ESP301 motion controller.

    Signals
    -------
    x : SignalRW[float]
        ``Position.Axis 1`` — lateral position, mm, range ±150 mm.
    y : SignalRW[float]
        ``Position.Axis 2`` — longitudinal position, mm, range ±6000 mm.
    z : SignalRW[float]
        ``Position.Axis 3`` — vertical position, mm, range ±100 mm.
    """

    def __init__(
        self,
        host: str,
        port: int,
        name: str = "jet_stage",
    ) -> None:
        """Initialise the jet stage device.

        Parameters
        ----------
        host:
            Device IP address (e.g. ``"192.168.8.198"``).
        port:
            Device UDP/TCP port (e.g. ``65158``).
        name:
            ophyd-async device name used to namespace signal keys.
        """
        udp = GeecsUdpClient(host, port)
        with self.add_children_as_readables():
            self.x = geecs_signal_rw(
                float,
                _DEVICE_NAME,
                "Position.Axis 1",
                host,
                port,
                units="mm",
                limits=(-150.0, 150.0),
                shared_udp=udp,
            )
            self.y = geecs_signal_rw(
                float,
                _DEVICE_NAME,
                "Position.Axis 2",
                host,
                port,
                units="mm",
                limits=(-6000.0, 6000.0),
                shared_udp=udp,
            )
            self.z = geecs_signal_rw(
                float,
                _DEVICE_NAME,
                "Position.Axis 3",
                host,
                port,
                units="mm",
                limits=(-100.0, 100.0),
                shared_udp=udp,
            )
        super().__init__(name=name, shared_udp=udp)
