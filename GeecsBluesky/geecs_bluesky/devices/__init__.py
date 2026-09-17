"""GEECS devices for Bluesky/ophyd-async — CA-backed, over the gateway PVs.

Two device shapes: :class:`~geecs_bluesky.devices.detector.GeecsDetector`
for anything that captures, the ``ca`` readables/movables for scalar-only
devices and settable children, and
:class:`~geecs_bluesky.devices.shot_control.ShotControl` for the trigger
box.
"""

from .ca import (
    CaConfirmSettable,
    CaMotor,
    CaPseudoPositioner,
    CaSettable,
    CaSnapshotReadable,
)
from .detector import GeecsDetector
from .shot_control import ShotControl

__all__ = [
    "CaConfirmSettable",
    "CaMotor",
    "CaPseudoPositioner",
    "CaSettable",
    "CaSnapshotReadable",
    "GeecsDetector",
    "ShotControl",
]
