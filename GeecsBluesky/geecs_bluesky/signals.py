"""Standalone signal factory functions for GEECS devices.

These mirror the ophyd-async pattern of ``epics_signal_rw``, ``epics_signal_r``, etc.
Use them when building device classes::

    from geecs_bluesky.signals import geecs_signal_rw, geecs_signal_r
    from geecs_bluesky.transport.udp_client import GeecsUdpClient

    class MyStage(GeecsDevice):
        def __init__(self, host: str, port: int, name: str = ""):
            dev = "U_MyStage"
            udp = GeecsUdpClient(host, port)   # shared, serialises concurrent reads
            with self.add_children_as_readables():
                self.x = geecs_signal_rw(float, dev, "Position.Axis 1", host, port,
                                         units="mm", shared_udp=udp)
                self.y = geecs_signal_rw(float, dev, "Position.Axis 2", host, port,
                                         units="mm", shared_udp=udp)
            super().__init__(name=name)
"""

from __future__ import annotations

from typing import Type

from ophyd_async.core import SignalR, SignalRW, SignalW

from geecs_bluesky.backends.geecs_signal_backend import GeecsSignalBackend
from geecs_bluesky.transport.udp_client import GeecsUdpClient


def geecs_signal_rw(
    datatype: Type,
    device_name: str,
    variable: str,
    host: str,
    port: int,
    units: str = "",
    limits: tuple[float, float] | None = None,
    shared_udp: GeecsUdpClient | None = None,
) -> SignalRW:
    """Create a read-write GEECS signal."""
    return SignalRW(
        GeecsSignalBackend(
            datatype=datatype,
            device_name=device_name,
            variable=variable,
            host=host,
            port=port,
            units=units,
            limits=limits,
            shared_udp=shared_udp,
        )
    )


def geecs_signal_r(
    datatype: Type,
    device_name: str,
    variable: str,
    host: str,
    port: int,
    units: str = "",
    shared_udp: GeecsUdpClient | None = None,
) -> SignalR:
    """Create a read-only GEECS signal."""
    return SignalR(
        GeecsSignalBackend(
            datatype=datatype,
            device_name=device_name,
            variable=variable,
            host=host,
            port=port,
            units=units,
            shared_udp=shared_udp,
        )
    )


def geecs_signal_w(
    datatype: Type,
    device_name: str,
    variable: str,
    host: str,
    port: int,
    shared_udp: GeecsUdpClient | None = None,
) -> SignalW:
    """Create a write-only GEECS signal."""
    return SignalW(
        GeecsSignalBackend(
            datatype=datatype,
            device_name=device_name,
            variable=variable,
            host=host,
            port=port,
            shared_udp=shared_udp,
        )
    )
