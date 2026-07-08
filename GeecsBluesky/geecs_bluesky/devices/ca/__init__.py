"""CA-backed ophyd-async devices: GEECS via the caproto gateway as an EPICS IOC.

These are the *stock*-EPICS presentation of GEECS devices — they consume the
:mod:`geecs_ca_gateway` PVs with plain ``epics_signal_r`` / ``epics_signal_rw``
and add only the GEECS-specific shot semantics (``acq_timestamp``-gated
trigger).  They are the CA counterpart of the direct UDP/TCP devices one level
up in ``geecs_bluesky/devices/``; the two are selected by backend, not by
divergent domain logic (shot-id / save-path / schema stay shared).

Requires the ``ca`` extra (``aioca``): ``poetry install --extras ca``.
"""

from geecs_bluesky.devices.ca.action_signals import CaActionSignalFactory
from geecs_bluesky.devices.ca.generic_detector import CaGenericDetector
from geecs_bluesky.devices.ca.motor import CaMotor
from geecs_bluesky.devices.ca.settable import CaSettable
from geecs_bluesky.devices.ca.snapshot import CaSnapshotReadable
from geecs_bluesky.devices.ca.telemetry import CaTelemetryReadable
from geecs_bluesky.devices.ca.timestamped_readable import CaTimestampedReadable
from geecs_bluesky.devices.ca.triggerable import CaAcqTimestampReadable, CaTriggerable

__all__ = [
    "CaAcqTimestampReadable",
    "CaActionSignalFactory",
    "CaGenericDetector",
    "CaMotor",
    "CaSettable",
    "CaSnapshotReadable",
    "CaTelemetryReadable",
    "CaTimestampedReadable",
    "CaTriggerable",
]
