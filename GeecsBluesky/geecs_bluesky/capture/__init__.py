"""Central PVA image capture (the ``capture`` extra: p4p + h5py).

Scan-gated capture daemon: subscribes to Point Grey camera NTNDArray PVs
during a scan, dedupes on ``acq_timestamp``, and trail-flushes one
frame-stack file per device per scan (``FORMAT.md`` — schema
``geecs-capture/1``) into the engine-created scan folders. Runs ALONGSIDE
the LV per-shot file save (dual-write doctrine); the arc's scope doc is
``Planning/data_capture/01_central_pva_capture_scope.md``.

Import surface is light: p4p/h5py load lazily inside the classes that need
them.
"""

from .daemon import CaptureDaemon, ScanCaptureSession
from .discovery import CAPTURE_DEVICE_TYPES, CameraTarget, discover_capture_cameras
from .subscriber import FrameSource, P4pFrameSource
from .writer import SCHEMA_ID, FrameStackWriter, Hdf5StackWriter

__all__ = [
    "CAPTURE_DEVICE_TYPES",
    "SCHEMA_ID",
    "CameraTarget",
    "CaptureDaemon",
    "FrameSource",
    "FrameStackWriter",
    "Hdf5StackWriter",
    "P4pFrameSource",
    "ScanCaptureSession",
    "discover_capture_cameras",
]
