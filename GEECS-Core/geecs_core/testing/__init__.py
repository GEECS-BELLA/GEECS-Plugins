"""Testing utilities: fake GEECS devices speaking the real wire protocols.

Also the shared test doubles: ``patch_geecs_db`` (canned experiment-wide DB
queries) and ``hand_fed_udp_client`` (a UDP client whose replies the test
delivers). Test code only; nothing in a package's runtime imports this.
"""

from .fake_db import patch_geecs_db
from .fake_device_server import FakeGeecsDevice, FakeGeecsServer
from .fake_scan_mode_device import FakeScanModeDevice, FakeScanModeGroup
from .hand_fed_udp import RecordingTransport, hand_fed_udp_client, wait_for_command

__all__ = [
    "FakeGeecsDevice",
    "FakeGeecsServer",
    "FakeScanModeDevice",
    "FakeScanModeGroup",
    "RecordingTransport",
    "hand_fed_udp_client",
    "patch_geecs_db",
    "wait_for_command",
]
