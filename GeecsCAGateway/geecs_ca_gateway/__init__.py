"""GEECS → EPICS Channel Access soft-IOC gateway.

Exposes GEECS devices as EPICS PVs via caproto, built on the GEECS transport
core (``geecs_bluesky.transport``).  See the package README for the design and
its place in the "standard access layer" plan.
"""

from .config import DeviceSpec, GatewayConfig, VariableSpec
from .gateway import GeecsCaGateway

__all__ = [
    "GeecsCaGateway",
    "GatewayConfig",
    "DeviceSpec",
    "VariableSpec",
]
