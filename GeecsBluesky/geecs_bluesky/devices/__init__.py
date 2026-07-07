"""ophyd-async Device classes for GEECS hardware (CA-backed via the gateway)."""

from .ca import (
    CaAcqTimestampReadable,
    CaGenericDetector,
    CaMotor,
    CaSettable,
    CaSnapshotReadable,
    CaTimestampedReadable,
    CaTriggerable,
)
from .contributor import FreeRunContributorSupport
from .nonscalar_save import NonScalarSaveSupport
from .scan_context import ScanContext
from .shot_id import ShotIdSupport, ShotIdTracker

__all__ = [
    "CaAcqTimestampReadable",
    "CaGenericDetector",
    "CaMotor",
    "CaSettable",
    "CaSnapshotReadable",
    "CaTimestampedReadable",
    "CaTriggerable",
    "FreeRunContributorSupport",
    "NonScalarSaveSupport",
    "ScanContext",
    "ShotIdSupport",
    "ShotIdTracker",
]
