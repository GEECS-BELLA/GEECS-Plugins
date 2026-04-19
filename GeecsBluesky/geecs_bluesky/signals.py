"""Standalone signal factory functions for GEECS devices.

These mirror the ophyd-async pattern of ``epics_signal_rw``, ``epics_signal_r``, etc.
Use them when building device classes::

    from geecs_bluesky.signals import geecs_signal_rw, geecs_signal_r

    class JetStage(StandardReadable):
        def __init__(self, host: str, port: int, name: str = ""):
            dev = "U_ESP_JetXYZ"
            with self.add_children_as_readables():
                self.position_x = geecs_signal_rw(float, dev, "Jet_X (mm)", host, port, units="mm")
                self.position_y = geecs_signal_rw(float, dev, "Jet_Y (mm)", host, port, units="mm")
            super().__init__(name=name)
"""

from __future__ import annotations

from typing import Type

from ophyd_async.core import SignalR, SignalRW, SignalW

from geecs_bluesky.backends.geecs_signal_backend import GeecsSignalBackend


def geecs_signal_rw(
    datatype: Type,
    device_name: str,
    variable: str,
    host: str,
    port: int,
    units: str = "",
    limits: tuple[float, float] | None = None,
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
        )
    )


def geecs_signal_r(
    datatype: Type,
    device_name: str,
    variable: str,
    host: str,
    port: int,
    units: str = "",
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
        )
    )


def geecs_signal_w(
    datatype: Type,
    device_name: str,
    variable: str,
    host: str,
    port: int,
) -> SignalW:
    """Create a write-only GEECS signal."""
    return SignalW(
        GeecsSignalBackend(
            datatype=datatype,
            device_name=device_name,
            variable=variable,
            host=host,
            port=port,
        )
    )
