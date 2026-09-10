"""CA-backed ophyd-async devices: GEECS via the caproto gateway as an EPICS IOC.

These are the *stock*-EPICS presentation of GEECS devices — they consume the
:mod:`geecs_ca_gateway` PVs with plain ``epics_signal_r`` / ``epics_signal_rw``
for the scalar-only devices and the settable children (motor, settable,
confirm, pseudo, snapshot) plus the gateway put primitive and the one-shot
reader.  Acquirers are :class:`~geecs_bluesky.devices.detector.GeecsDetector`
one level up.

Requires the ``ca`` extra (``aioca``): ``poetry install --extras ca``.
"""

from geecs_bluesky.devices.ca.action_signals import CaActionSignalFactory
from geecs_bluesky.devices.ca.confirm import CaConfirmSettable
from geecs_bluesky.devices.ca.motor import CaMotor
from geecs_bluesky.devices.ca.pseudo import CaPseudoMovable
from geecs_bluesky.devices.ca.settable import CaSettable
from geecs_bluesky.devices.ca.snapshot import CaSnapshotReadable

__all__ = [
    "CaActionSignalFactory",
    "CaConfirmSettable",
    "CaMotor",
    "CaPseudoMovable",
    "CaSettable",
    "CaSnapshotReadable",
]
