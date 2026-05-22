"""Software interlocks for GEECS device monitoring.

A small TCP "permission" server: each registered interlock periodically
evaluates a condition against a GEECS device (or any other data
source); the server broadcasts the per-interlock safe/unsafe flags to
connected clients (typically Master Control).
"""

from .base_interlock import BaseInterlock
from .camera_threshold import CameraThresholdInterlock
from .config_loader import INTERLOCK_REGISTRY, load_interlock_server
from .geecs_interlock_server import InterlockServer

__all__ = [
    "BaseInterlock",
    "CameraThresholdInterlock",
    "INTERLOCK_REGISTRY",
    "InterlockServer",
    "load_interlock_server",
]
