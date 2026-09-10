"""GEECS devices for Bluesky/ophyd-async — CA-backed, over the gateway PVs.

Two device shapes (``Planning/native_bluesky/03_clean_room_rebuild.md``
§4.A): :class:`~geecs_bluesky.devices.detector.GeecsDetector` for anything
that captures, the ``ca`` readables/movables for scalar-only devices and
settable children, and :class:`~geecs_bluesky.devices.shot_control.ShotControl`
for the trigger box.
"""

from .ca import (
    CaActionSignalFactory,
    CaConfirmSettable,
    CaMotor,
    CaPseudoMovable,
    CaSettable,
    CaSnapshotReadable,
)
from .detector import GeecsDetector
from .shot_control import ShotControl

__all__ = [
    "CaActionSignalFactory",
    "CaConfirmSettable",
    "CaMotor",
    "CaPseudoMovable",
    "CaSettable",
    "CaSnapshotReadable",
    "GeecsDetector",
    "ShotControl",
]
